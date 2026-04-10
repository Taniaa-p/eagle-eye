import os
from ultralytics import YOLO

def main():
    print("=== YOLOv8 Training Script for Smart Helmet Detection ===")
    
    # 1. Load the pre-trained YOLOv8 nano model
    print("\nLoading yolov8n.pt model...")
    # 'yolov8n.pt' will be automatically downloaded by ultralytics if it does not exist locally
    model = YOLO("yolov8n.pt") 
    
    # 2. Start training
    print("\nStarting training phase...")
    print("Hardware Target: MPS (Apple Silicon GPU)")
    
    # Retrieve absolute path for stability
    dataset_yaml = os.path.abspath("dataset/data.yaml")
    
    # Ultralytics natively handles progress reporting, saving results, maps, etc., automatically.
    # The training results (loss, mAP, precision, recall) will be heavily logged in the output directory.
#     results = model.train(
#     data=dataset_yaml,
#     epochs=30,
#     imgsz=512,
#     batch=8,
#     device="mps",
#     val=False, 
#     name="helmet_detection"
# )

    results = model.train(
        data=dataset_yaml,
        epochs=30,
        imgsz=512,
        batch=8,
        device="cpu",          # 👈 CRITICAL FIX: PyTorch MPS has bugs with YOLOv8 validation. Fallback to CPU!
        val=True,              # Change back to True!
        amp=False,             # 👈 CRITICAL FIX: Disables Mixed Precision to prevent NaN explosions
        name="helmet_detection"
    )

    
    # 3. Post-Training Info
    print("\n=== Training Completed ===")
    print("Detailed metrics, charts (mAP, Precision, Recall), and the final models have been saved automatically.")
    
    # Locate the best model dynamically inside the training run tracking folder
    best_weights = os.path.join(results.save_dir, "weights", "best.pt")
    
    print("\n" + "="*50)
    print(f"✅ YOUR BEST MODEL (best.pt) IS SAVED AT:")
    print(f"   --> {best_weights}")
    print("="*50 + "\n")
    
if __name__ == "__main__":
    main()
