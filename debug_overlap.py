import cv2
from ultralytics import YOLO

# load base model
base_model = YOLO("yolov8n.pt")
helmet_model = YOLO("runs/detect/helmet_detection7/weights/best.pt")

image_path = "/Users/tania/Downloads/WhatsApp Image 2026-04-10 at 10.59.59.jpeg"
frame = cv2.imread(image_path)

if frame is None:
    print("Could not read image!")
    exit()

base_results = base_model.predict(source=frame, device="cpu", conf=0.2, verbose=False)[0]
people_boxes = []
motorcycle_boxes = []

print("--- Base Model Detections ---")
if base_results.boxes:
    for box in base_results.boxes:
        cls_id = int(box.cls[0])
        name = base_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if name == 'person':
            people_boxes.append((x1,y1,x2,y2))
            print(f"Person: {x1},{y1} to {x2},{y2} | conf={conf:.2f}")
        elif name == 'motorcycle':
            motorcycle_boxes.append((x1,y1,x2,y2))
            print(f"Motorcycle: {x1},{y1} to {x2},{y2} | conf={conf:.2f}")

print("\n--- Overlap Ratios ---")
def check_overlap_ratio(person_box, target_box):
    px1, py1, px2, py2 = person_box
    tx1, ty1, tx2, ty2 = target_box
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    person_area = max(0, px2 - px1) * max(0, py2 - py1)
    if person_area == 0: return 0
    return inter_area / person_area

for i, p in enumerate(people_boxes):
    for j, m in enumerate(motorcycle_boxes):
        ratio = check_overlap_ratio(p, m)
        print(f"Person {i} & Moto {j} overlap ratio: {ratio:.3f}")

print("\n--- Helmet Model Detections ---")
helmet_results = helmet_model.predict(source=frame, device="cpu", conf=0.2, verbose=False)[0]
if helmet_results.boxes:
    for box in helmet_results.boxes:
        cls_id = int(box.cls[0])
        name = helmet_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"Helmet Model - {name}: {x1},{y1} to {x2},{y2} | conf={conf:.2f}")

