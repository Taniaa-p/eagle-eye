from ultralytics import YOLO

model = YOLO("runs/detect/helmet_detection7/weights/best.pt")

results = model.predict(
    source="dataset/images/val/helme (8).jpeg", # your test image
    device="cpu",
    conf=0.4,
    save=True
)

print("Check runs/detect/predict/")


