#!/usr/bin/env python3
"""
System verification script for AI Game Automation
Verifies configuration and system compatibility
"""

import os
import sys

from config import (
    BATCH_SIZE,
    COMMON_KEYS,
    DATA_DIR,
    FRAME_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    SEQUENCE_LENGTH,
    TRAIN_IMG_HEIGHT,
    TRAIN_IMG_WIDTH,
    validate_config,
)


def check_directories():
    """Check and create required directories."""
    print("📁 Checking directories...")

    directories = [DATA_DIR, FRAME_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created: {directory}")
        else:
            print(f"✅ Exists: {directory}")


def check_configuration():
    """Check configuration settings."""
    print("\n⚙️  Checking configuration...")

    # Check image dimensions
    print(f"📐 Recording dimensions: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"📐 Training dimensions: {TRAIN_IMG_WIDTH}x{TRAIN_IMG_HEIGHT}")

    # Check key configuration
    print(f"⌨️  Keys configured: {len(COMMON_KEYS)} keys")
    if len(COMMON_KEYS) < 5:
        print("⚠️  Warning: Very few keys configured")

    # Check training parameters
    print(f"🧠 Batch size: {BATCH_SIZE}")
    print(f"🧠 Sequence length: {SEQUENCE_LENGTH}")

    # Validate configuration
    try:
        validate_config()
        return True
    except AssertionError as e:
        print(f"❌ Configuration error: {e}")
        return False


def check_memory_requirements():
    """Estimate memory requirements."""
    print("\n💾 Memory requirements:")

    # Calculate approximate memory usage
    frame_size = TRAIN_IMG_WIDTH * TRAIN_IMG_HEIGHT * 3  # RGB
    sequence_size = frame_size * SEQUENCE_LENGTH
    batch_size = sequence_size * BATCH_SIZE

    # Convert to MB
    batch_mb = batch_size / (1024 * 1024)

    print(f"📊 Estimated batch memory: {batch_mb:.1f} MB")

    if batch_mb > 1000:
        print("⚠️  High memory usage - consider reducing batch size")
    elif batch_mb < 50:
        print("✅ Low memory usage")
    else:
        print("✅ Reasonable memory usage")


def main():
    """Main verification function."""
    print("🔍 AI Game Automation - System Verification")
    print("=" * 50)

    # Check directories
    check_directories()

    # Check configuration
    if not check_configuration():
        print("\n❌ Configuration verification failed!")
        sys.exit(1)

    # Check memory requirements
    check_memory_requirements()

    print("\n✅ System verification complete!")
    print("🎉 Ready to start recording and training!")


if __name__ == "__main__":
    main()
