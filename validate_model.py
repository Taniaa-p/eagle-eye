import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    python_path = sys.executable
    print("-" * 50)
    print(f"Python path being used: {python_path}")
    
    # Check if we are using the venv_arm environment
    if 'venv_arm' in python_path:
        print("Confirmed: Using venv_arm virtual environment")
    else:
        print("Warning: Not using venv_arm virtual environment")
        
    # The known model path
    model_path = Path("runs/detect/helmet_detection5/weights/best.pt")
    print(f"Found model path: {model_path}")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print("-" * 50)
    print("Running validation...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Run validation
    metrics = model.val(data="dataset/data.yaml", device="mps")
    
    # Print results neatly
    print("\n" + "=" * 50)
    print("Validation Results:")
    print("=" * 50)
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
