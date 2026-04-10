import os
import random
from pathlib import Path
from ultralytics import YOLO

# Paths
MODEL_PATH = "runs/detect/helmet_detection5/weights/best.pt"
VAL_DIR = Path("dataset/images/val")
CUSTOM_IMG = Path("no_helmet.jpeg")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load model
    model = YOLO(MODEL_PATH)
    
    # Gather images
    images_to_test = []
    
    # 5 random images from val
    if VAL_DIR.exists():
        all_val_imgs = list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.jpeg")) + list(VAL_DIR.glob("*.png"))
        if len(all_val_imgs) > 5:
            val_samples = random.sample(all_val_imgs, 5)
        else:
            val_samples = all_val_imgs
        images_to_test.extend(val_samples)
    else:
        print(f"Warning: Validation directory not found at {VAL_DIR}")
        
    # Custom image
    if CUSTOM_IMG.exists():
        images_to_test.append(CUSTOM_IMG)
    else:
        print(f"Warning: Custom image {CUSTOM_IMG} not found. Skipping it.")
        
    if not images_to_test:
        print("No images found to test!")
        return

    print("=" * 50)
    print(f"Testing {len(images_to_test)} images visually...")
    print("=" * 50)
    
    # Run inference
    results = model.predict(
        source=images_to_test,
        device="mps",
        save=True,
        project="runs/detect",
        name="test_results",
        exist_ok=True
    )
    
    # Process results
    for i, r in enumerate(results):
        img_path = images_to_test[i]
        print(f"\nImage: {img_path.name}")
        
        boxes = r.boxes
        if len(boxes) == 0:
            print("Model is not detecting properly")
        else:
            for box in boxes:
                # get class id
                cls_id = int(box.cls[0].item())
                # get class name
                cls_name = model.names[cls_id]
                # get confidence
                conf = box.conf[0].item()
                
                print(f"  - Detected: {cls_name} (Confidence: {conf:.2f})")
                
    print("\n" + "=" * 50)
    print("All output images with bounding boxes have been saved to runs/detect/test_results/")

if __name__ == '__main__':
    main()
