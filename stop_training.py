#!/usr/bin/env python3
"""
Simple script to create a STOP_TRAINING file
This will gracefully stop the training after the current epoch completes
"""

import os
from pathlib import Path

def create_stop_file():
    """Create a STOP_TRAINING file to gracefully stop training."""
    stop_file = Path("STOP_TRAINING")
    stop_file.touch()
    print("🛑 Created STOP_TRAINING file")
    print("📝 Training will stop after completing the current epoch")
    print("💡 Delete the file to resume training: rm STOP_TRAINING")

def remove_stop_file():
    """Remove the STOP_TRAINING file to allow training to continue."""
    stop_file = Path("STOP_TRAINING")
    if stop_file.exists():
        stop_file.unlink()
        print("✅ Removed STOP_TRAINING file")
        print("🚀 Training can now continue")
    else:
        print("ℹ️  No STOP_TRAINING file found")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "remove":
        remove_stop_file()
    else:
        create_stop_file() 