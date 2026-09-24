import sys
import time

def check_python_version():
    print(f"Checking Python version... {sys.version.split(' ')[0]}")
    if sys.version_info < (3, 8):
        print("WARNING: Python version < 3.8 is generally not recommended for YOLOv8/v11.")
    else:
        print("Python version OK.")

def check_numpy():
    print("Checking NumPy...", end=" ")
    try:
        import numpy as np
        print(f"OK ({np.__version__})")
    except ImportError:
        print("FAILED. Please install numpy.")
        return False
    return True

def check_opencv():
    print("Checking OpenCV...", end=" ")
    try:
        import cv2
        print(f"OK ({cv2.__version__})")
    except ImportError:
        print("FAILED. Please install opencv-python.")
        return False
    return True

def check_pytorch():
    print("Checking PyTorch...", end=" ")
    try:
        import torch
        print(f"OK ({torch.__version__})")
        
        print("Checking CUDA availability...", end=" ")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"OK ({device_name})")
        else:
            print("WARNING: CUDA is not available. YOLO will run on CPU, which may be too slow for real-time control.")
    except ImportError:
        print("FAILED. Please install torch.")
        return False
    return True

def check_ultralytics():
    print("Checking Ultralytics (YOLO)...", end=" ")
    try:
        import ultralytics
        print(f"OK ({ultralytics.__version__})")
    except ImportError:
        print("FAILED. Please install ultralytics.")
        return False
    return True

def check_carla_connection():
    print("Checking CARLA Server connection...", end=" ")
    try:
        import carla
        print("CARLA module OK")
    except ImportError:
        print("FAILED. Please install the CARLA python API (e.g., pip install carla==0.9.15).")
        return False

    print("Attempting to connect to CARLA server at localhost:2000...")
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(5.0)
        
        # Test connection by fetching the CARLA server version
        server_version = client.get_server_version()
        print(f"Connection OK! Connected to CARLA server version {server_version}.")
    except Exception as e:
        print("\n------------------------------------------------------------")
        print("FAILED: Could not connect to CARLA server.")
        print(f"Error: {e}")
        print("CARLA server is not running or has not finished initializing.")
        print("Start the CARLA server first using run_server.bat or CarlaUE4.exe.")
        print("------------------------------------------------------------")
        return False
    
    return True

def main():
    print("============================================================")
    print("CARLA Adaptive Control - Environment Validation")
    print("============================================================")
    
    check_python_version()
    print("-" * 60)
    
    all_ok = True
    all_ok &= check_numpy()
    all_ok &= check_opencv()
    all_ok &= check_pytorch()
    all_ok &= check_ultralytics()
    print("-" * 60)
    
    all_ok &= check_carla_connection()
    
    print("============================================================")
    if all_ok:
        print("Validation Successful! The environment is ready.")
    else:
        print("Validation Failed. Please address the errors above.")

if __name__ == "__main__":
    main()
