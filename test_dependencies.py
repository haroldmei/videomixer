#!/usr/bin/env python3
"""
Test script for optimized video GPS overlay
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV not found")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy not found")
        return False
    
    try:
        import folium
        print("✓ Folium imported successfully")
    except ImportError:
        print("✗ Folium not found - install with: pip install folium")
        return False
    
    try:
        import tqdm
        print("✓ tqdm imported successfully")
    except ImportError:
        print("✗ tqdm not found - install with: pip install tqdm")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError:
        print("✗ Pillow not found - install with: pip install pillow")
        return False
    
    return True

def main():
    print("Testing optimized video GPS overlay dependencies...")
    print("=" * 50)
    
    if test_imports():
        print("\n✓ All dependencies are available!")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
    else:
        print("\n✗ Some dependencies are missing.")
        print("Please install them using:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
