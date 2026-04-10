import torch
import torchvision
import numpy as np
import matplotlib
import cv2
import ultralytics

def run_checks():
    print("=== Environment Verification ===")
    
    # 1. Version checks
    print(f"Python versions & Dependencies:")
    print(f"- PyTorch:     {torch.__version__}")
    print(f"- Torchvision: {torchvision.__version__}")
    print(f"- Numpy:       {np.__version__}")
    print(f"- Matplotlib:  {matplotlib.__version__}")
    print(f"- OpenCV:      {cv2.__version__}")
    print(f"- Ultralytics: {ultralytics.__version__}")
    
    print("\n=== GPU Verification ===")
    
    # 2. CUDA check
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available (NVIDIA): {cuda_available}")
    
    # 3. MPS check (for Apple Silicon Macs)
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
        print(f"MPS available (Apple Silicon): {mps_available}")
        
        if mps_available:
            print("-> You can use device='mps' for GPU acceleration on your Mac!")
    
    print("\n✅ All imports and checks completed successfully!")

if __name__ == "__main__":
    run_checks()
