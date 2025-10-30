"""
Installation script for Virtual Board
"""
import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["models", "saves", "logs"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def check_camera():
    """Check if camera is available"""
    print("Checking camera availability...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✓ Camera is working")
                cap.release()
                return True
            else:
                print("✗ Camera found but cannot read frames")
        else:
            print("✗ Cannot open camera")
        cap.release()
    except ImportError:
        print("✗ OpenCV not installed")
    except Exception as e:
        print(f"✗ Camera check failed: {e}")
    
    return False

def check_system_requirements():
    """Check system requirements"""
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check for GPU support (optional)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU support available: {len(gpus)} GPU(s) found")
        else:
            print("ℹ No GPU found, using CPU (this is fine)")
    except ImportError:
        print("ℹ TensorFlow not yet installed")

def main():
    """Main installation process"""
    print("="*60)
    print("VIRTUAL BOARD - INSTALLATION SCRIPT")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\nInstallation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Check camera
    camera_ok = check_camera()
    
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    print("✓ Python version check passed")
    print("✓ Required directories created")
    print("✓ Python packages installed")
    
    if camera_ok:
        print("✓ Camera check passed")
    else:
        print("⚠ Camera check failed - please ensure your camera is connected")
    
    print("\nTo run the Virtual Board:")
    print("  python -m src.main")
    print("  or")
    print("  python demo.py")
    
    print("\nFor help:")
    print("  python -m src.main --help")
    
    print("\n" + "="*60)
    print("Installation completed!")
    print("="*60)

if __name__ == "__main__":
    main()