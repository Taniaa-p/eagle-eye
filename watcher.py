import os
# Fix macOS PyTorch + OpenCV deadlocks
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import cv2
import argparse
import numpy as np
import easyocr
import re
import csv
from datetime import datetime
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

def fix_orientation(img):
    return cv2.rotate(img, cv2.ROTATE_180)

def read_plate_text(reader, image):
    """Extracts and cleans alphanumeric text from license plate images using EasyOCR."""
    if image is None or image.size == 0:
        return ""
        
    # Read text using easyocr
    results = reader.readtext(image, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if not results:
        return ""
        
    text = ""
    for (bbox, t, conf) in results:
        if conf >= 0.0:
            text += t
    
    # Keep only A-Z and 0-9
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return clean_text

def enhance_plate(image, frame_count, moto_id):
    """Enhances license plate readability for OCR and saves debug states."""
    if image is None or image.size == 0:
        return image
        
    debug_dir = f"debug_steps/plate_{frame_count}_{moto_id}"
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(f"{debug_dir}/1_original.jpg", image)
    
    # Pad the image heavily so edge characters (like T or N) aren't cropped out
    padded = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
    # Resize 2x
    img = cv2.resize(padded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"{debug_dir}/2_resized.jpg", img)
    
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{debug_dir}/3_gray.jpg", gray)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    cv2.imwrite(f"{debug_dir}/4_clahe.jpg", clahe_img)
    
    # Sharpen
    blur = cv2.GaussianBlur(clahe_img, (5,5), 0)
    sharp = cv2.addWeighted(clahe_img, 1.5, blur, -0.5, 0)
    cv2.imwrite(f"{debug_dir}/5_sharp.jpg", sharp)
    
    return sharp

def check_overlap_ratio(person_box, target_box, threshold=0.3):
    """
    Checks if the intersection area between person_box and target_box
    is greater than a given ratio of the person_box's area.
    Boxes should be in format (x1, y1, x2, y2).
    """
    px1, py1, px2, py2 = person_box
    tx1, ty1, tx2, ty2 = target_box

    # Calculate intersection coordinates
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)

    # Calculate intersection area
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    inter_area = inter_width * inter_height

    # Calculate person area
    person_area = max(0, px2 - px1) * max(0, py2 - py1)

    if person_area == 0:
        return False

    return (inter_area / person_area) > threshold

class HelmetWatcher:
    def __init__(self, helmet_model_path="runs/detect/helmet_detection7/weights/best.pt", base_model_path="yolov8n.pt"):
        print(f"Loading Base YOLO model from {base_model_path}...")
        self.base_model = YOLO(base_model_path)
        
        print(f"Loading Helmet YOLO model from {helmet_model_path}...")
        self.helmet_model = YOLO(helmet_model_path)
        
        self.helmet_names = self.helmet_model.names
        self.base_names = self.base_model.names
        
        # COCO class indices usually: 0 for person, 3 for motorcycle. Ensure this using names.
        self.person_class_id = list(self.base_names.keys())[list(self.base_names.values()).index('person')]
        self.motorcycle_class_id = list(self.base_names.keys())[list(self.base_names.values()).index('motorcycle')]

        self.total_violations = 0
        self.frame_count = 0
        
        self.output_dir = "runs/detect/final_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.plates_dir = "runs/plates"
        os.makedirs(self.plates_dir, exist_ok=True)
        
        self.enhanced_dir = "runs/detect/enhanced"
        os.makedirs(self.enhanced_dir, exist_ok=True)
        
        self.debug_dir = "debug_steps"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Roboflow Setup
        self.rf_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="o6ALAmEk1ZcuogAK6cnx"
        )
        
        # EasyOCR Setup
        print("Initializing EasyOCR reader (English) with MPS hardware acceleration...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=True)
        
        # CSV Logging Setup
        self.csv_file = "runs/detect/violations_log.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "plate_number", "violation_type"])
        
    def detect_plate_with_roboflow(self, frame, moto_box, moto_id):
        """
        Extract the license plate ROI from the motorcycle bounding box using Roboflow
        and save it to runs/plates.
        """
        x1, y1, x2, y2 = moto_box
        
        # Crop motorcycle from original frame
        moto_crop = frame[y1:y2, x1:x2]
        if moto_crop.size == 0:
            return
            
        try:
            # Send the cropped motorcycle image directly to Roboflow using targeted model inference
            result = self.rf_client.infer(moto_crop, model_id="license-plate-recognition-rxg4e/11")
            
            # Extract standard detection outputs
            outs = []
            if isinstance(result, dict) and 'predictions' in result:
                outs = result['predictions']
            elif hasattr(result, 'predictions'):
                # In case inference SDK returns a native Response object
                outs = [dict(p) for p in result.predictions] if not isinstance(result.predictions, list) else result.predictions
                
            for pred in outs:
                # The prediction could be named 'license-plate', '0', etc.
                # Just draw anything found since the model only finds plates
                if isinstance(pred, dict) and 'x' in pred and 'y' in pred and 'width' in pred and 'height' in pred:
                    cx = pred['x']
                    cy = pred['y']
                    w = pred['width']
                    h = pred['height']
                    
                    # Convert center x/y to bounding box for the plate in crop coordinates
                    px1 = int(cx - (w / 2))
                    py1 = int(cy - (h / 2))
                    px2 = int(cx + (w / 2))
                    py2 = int(cy + (h / 2))
                    
                    plate_crop = moto_crop[py1:py2, px1:px2]
                    
                    plate_text = ""
                    if plate_crop.size > 0:
                        plate_filename = os.path.join(self.plates_dir, f"plate_{self.frame_count}_{moto_id}.jpg")
                        cv2.imwrite(plate_filename, plate_crop)
                        
                        # Enhance plate for OCR
                        enhanced_plate = enhance_plate(plate_crop, self.frame_count, moto_id)
                        if enhanced_plate is not None and enhanced_plate.size > 0:
                            enhanced_filename = os.path.join(self.enhanced_dir, f"enhanced_{self.frame_count}_{moto_id}.jpg")
                            cv2.imwrite(enhanced_filename, enhanced_plate)
                            
                            # Read plate text
                            plate_text = read_plate_text(self.ocr_reader, enhanced_plate)
                            
                            # Log to CSV if valid (e.g. at least 4 characters long)
                            if plate_text and len(plate_text) >= 4:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                with open(self.csv_file, mode='a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([timestamp, plate_text, "No Helmet"])
                    
                    # Map the local coordinates back to the global frame context
                    gx1 = x1 + px1
                    gy1 = y1 + py1
                    gx2 = x1 + px2
                    gy2 = y1 + py2
                    
                    # Draw Yellow PLATE bounding box
                    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
                    
                    # Display the extracted text if available, otherwise just say "PLATE"
                    display_text = plate_text if plate_text else "PLATE"
                    cv2.putText(frame, display_text, (gx1, gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        except Exception as e:
            print(f"Roboflow API error: {e}")

    def process_frame(self, frame):
        """Processes a single frame: finds riders, performs helmet inference on them, draws boxes, extracts plates."""
        self.frame_count += 1
        
        # Stage 1: Base Detection (People and Motorcycles)
        base_results = self.base_model.predict(source=frame, device="cpu", conf=0.4, verbose=False)
        base_result = base_results[0]
        
        people_boxes = []
        motorcycle_boxes = []
        
        if base_result.boxes is not None:
            for box in base_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                if class_id == self.person_class_id:
                    people_boxes.append((x1, y1, x2, y2))
                elif class_id == self.motorcycle_class_id:
                    motorcycle_boxes.append((x1, y1, x2, y2))
        
        # Determine validated "riders" (people overlapping with a motorcycle)
        # Store as tuple: (person_box, moto_box)
        rider_mappings = []
        for person_box in people_boxes:
            for moto_box in motorcycle_boxes:
                if check_overlap_ratio(person_box, moto_box, threshold=0.3):
                    rider_mappings.append((person_box, moto_box))
                    break # Assign person as rider and stop checking other motorcycles
        
        # If no riders found, return the frame as-is (we ignore pedestrians)
        if not rider_mappings:
            return frame

        # Variables to track plate extractions this frame to prevent duplicates
        processed_motorcycles = set()
        
        # Stage 2: Helmet Inference
        helmet_results = self.helmet_model.predict(source=frame, device="cpu", conf=0.25, verbose=False)
        helmet_result = helmet_results[0]
        
        if helmet_result.boxes is not None:
            for box in helmet_result.boxes:
                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                helmet_box = (hx1, hy1, hx2, hy2)
                
                # Check if this helmet detection overlaps with a valid rider
                matched_moto_box = None
                for person_box, moto_box in rider_mappings:
                    # using somewhat higher threshold to match helmets strictly to the rider area
                    if check_overlap_ratio(helmet_box, person_box, threshold=0.1):
                        matched_moto_box = moto_box
                        break
                        
                if matched_moto_box is None:
                    continue # Ignore detections not tied to a valid rider
                    
                class_id = int(box.cls[0])
                label_name = self.helmet_names[class_id].lower()
                
                if label_name in ["no_helmet", "no-helmet"]:
                    # Mark as violation
                    self.total_violations += 1
                    color = (0, 0, 255) # Red in BGR
                    text = "VIOLATION"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 3)
                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (hx1, hy1 - 30), (hx1 + tw + 10, hy1), color, -1)
                    cv2.putText(frame, text, (hx1 + 5, hy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Prevent duplicate extractions per motorcycle in this frame
                    if matched_moto_box not in processed_motorcycles:
                        moto_id = len(processed_motorcycles)
                        self.detect_plate_with_roboflow(frame, matched_moto_box, moto_id)
                        processed_motorcycles.add(matched_moto_box)
                    
                elif label_name == "helmet":
                    color = (0, 255, 0) # Green in BGR
                    text = "helmet"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)
                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (hx1, hy1 - 25), (hx1 + tw + 10, hy1), color, -1)
                    cv2.putText(frame, text, (hx1 + 5, hy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        return frame

    def process_image(self, image_path):
        """Processes a single image file."""
        print(f"Processing image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return
            
        annotated_frame = self.process_frame(frame)
        
        # Display output
        cv2.imshow("Watcher: Helmet Detection", annotated_frame)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0) # Wait for key press
        cv2.destroyAllWindows()
        
        # Save output
        filename = os.path.basename(image_path)
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, annotated_frame)
        print(f"Saved annotated image to {save_path}")
        print(f"Total Violations Detected: {self.total_violations}")

    def process_video(self, video_path):
        """Processes a video file frame by frame."""
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties for saving
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Fallback if fps is missing/zero
        if not fps or fps != fps:
            fps = 30.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        filename = os.path.basename(video_path)
        save_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, int(fps), (width, height))
        
        print("Press 'q' in the video window to stop processing.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            annotated_frame = self.process_frame(frame)
            
            # Write out to video
            out.write(annotated_frame)
            
            # Display real-time output
            cv2.imshow("Watcher: Helmet Detection", annotated_frame)
            
            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved annotated video to {save_path}")
        print(f"Total Violations Detected: {self.total_violations}")

def main():
    parser = argparse.ArgumentParser(description="Helmet Detection System (Watcher)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video file.")
    parser.add_argument("--helmet-model", type=str, default="runs/detect/helmet_detection7/weights/best.pt", help="Path to YOLOv8 helmet model weights.")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Path to standard YOLOv8 model for general detection.")
    
    args = parser.parse_args()
    
    watcher = HelmetWatcher(helmet_model_path=args.helmet_model, base_model_path=args.base_model)
    
    # Simple check for video vs image extensions based on common formats
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ext = os.path.splitext(args.source)[-1].lower()
    
    if ext in video_exts:
        watcher.process_video(args.source)
    else:
        watcher.process_image(args.source)

if __name__ == "__main__":
    main()
