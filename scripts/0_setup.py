#!/usr/bin/env python3
"""
Setup script for AI Game Automation Project
Automatically installs dependencies and sets up the environment.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up AI Game Automation Project")
    print("=" * 50)
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip not found. Please install pip first.")
        return False
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Install PyTorch first (it's large and might need special handling)
    if not run_command(f"{sys.executable} -m pip install torch torchvision", "Installing PyTorch"):
        print("⚠️  PyTorch installation failed. You may need to install it manually.")
        print("   Visit: https://pytorch.org/get-started/locally/")
    
    # Install other dependencies
    dependencies = [
        "opencv-python",
        "mss", 
        "pynput",
        "matplotlib",
        "numpy",
        "Pillow"
    ]
    
    for dep in dependencies:
        if not run_command(f"{sys.executable} -m pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Failed to install {dep}")
    
    # Test imports
    print("\n🔍 Testing imports...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("check_dependencies", "1_check_dependencies.py")
        check_dependencies = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(check_dependencies)
        check_dependencies.main()
    except ImportError:
        print("❌ Could not run import test")
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Record some gameplay data: python 3_record_data.py")
    print("2. Test your data quality: python 4_analyze_data_quality.py")
    print("3. Train the improved model: python 5_train_model.py")
    print("4. Run inference: python 6_run_inference.py")
    print("\n📖 For detailed instructions, see README.md")
    
    return True

if __name__ == "__main__":
    main()