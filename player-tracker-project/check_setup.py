#!/usr/bin/env python3
"""
Diagnostic script to verify all dependencies are installed correctly.
Run: python check_setup.py
"""

import sys

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Need 3.9+")
        return False

def check_import(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name:25} - OK")
        return True
    except ImportError:
        print(f"❌ {package_name:25} - MISSING")
        return False

def main():
    print("="*50)
    print("⚽ Player Tracker - Dependency Check")
    print("="*50)
    print()
    
    # Python version
    print("System:")
    python_ok = check_python_version()
    print()
    
    # Required packages
    print("Required Packages:")
    checks = [
        ("streamlit", "streamlit"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("YOLO (ultralytics)", "ultralytics"),
        ("EasyOCR", "easyocr"),
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("DeepSort", "deep_sort_realtime"),
        ("SciPy", "scipy"),
        ("Pillow", "PIL"),
        ("StatsBombPy", "statsbombpy"),
        ("imageio", "imageio"),
        ("imageio-ffmpeg", "imageio_ffmpeg"),
    ]
    
    results = []
    for package, import_name in checks:
        results.append(check_import(package, import_name))
    
    print()
    print("="*50)
    
    # Summary
    all_ok = python_ok and all(results)
    
    if all_ok:
        print("✅ All dependencies installed! Ready to run.")
        print()
        print("Next step:")
        print("  streamlit run player_tracker_final.py")
    else:
        print("❌ Some packages missing. Install with:")
        print()
        print("  pip install -r requirements.txt")
    
    print("="*50)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
