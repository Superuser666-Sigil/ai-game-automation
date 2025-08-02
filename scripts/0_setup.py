#!/usr/bin/env python3
"""
Setup script for AI Game Automation
Installs dependencies and verifies system compatibility
"""

import subprocess
import sys


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version)
    return True


def install_requirements():
    """Install required packages."""
    print("\n📦 Installing dependencies...")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 AI Game Automation Setup")
    print("=" * 40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install requirements
    if not install_requirements():
        print("\n💡 Try running: pip install -r requirements.txt manually")
        sys.exit(1)

    print("\n✅ Setup complete! Run the following to get started:")
    print("   python scripts/2_verify_system_setup.py")
    print("   python scripts/3_record_data.py")


if __name__ == "__main__":
    main()
