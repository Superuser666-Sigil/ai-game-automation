#!/usr/bin/env python3
"""
Quick test script to verify the AI game automation pipeline works on AMD systems.
"""

import torch
import numpy as np
import cv2
import mss
import sys
sys.path.append('.')
import importlib.util
spec = importlib.util.spec_from_file_location("train_model", "5_train_model.py")
train_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_model)
ImprovedBehaviorCloningCNNRNN = train_model.ImprovedBehaviorCloningCNNRNN

def verify_system_setup():
    print("🧪 Testing AI Game Automation Pipeline")
    print("=" * 50)
    
    # Test 1: PyTorch setup
    print("1. Testing PyTorch setup...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("   ✅ PyTorch setup OK")
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    try:
        model = ImprovedBehaviorCloningCNNRNN(output_dim=9)
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("   ✅ Model creation OK")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 3: Model forward pass
    print("\n3. Testing model forward pass...")
    try:
        # Create dummy input (batch_size=1, sequence_length=5, channels=3, height=960, width=540)
        dummy_input = torch.randn(1, 5, 3, 960, 540)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("   ✅ Forward pass OK")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test 4: Screen capture
    print("\n4. Testing screen capture...")
    try:
        sct = mss.mss()
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        print(f"   Screen resolution: {img.shape}")
        print("   ✅ Screen capture OK")
    except Exception as e:
        print(f"   ❌ Screen capture failed: {e}")
        return False
    
    # Test 5: Image processing
    print("\n5. Testing image processing...")
    try:
        # Resize and normalize
        img_resized = cv2.resize(img, (960, 540))
        img_normalized = img_resized.astype(np.float32) / 255.0
        print(f"   Resized shape: {img_resized.shape}")
        print(f"   Normalized range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        print("   ✅ Image processing OK")
    except Exception as e:
        print(f"   ❌ Image processing failed: {e}")
        return False
    
    # Test 6: Memory usage
    print("\n6. Testing memory usage...")
    try:
        # Create a larger batch to test memory
        large_batch = torch.randn(4, 5, 3, 960, 540)
        with torch.no_grad():
            large_output = model(large_batch)
        print(f"   Large batch shape: {large_batch.shape}")
        print(f"   Large output shape: {large_output.shape}")
        print("   ✅ Memory test OK")
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your AMD system is ready for AI game automation.")
    print("\n📋 Next steps:")
    print("1. Record some gameplay: python 3_record_data.py")
    print("2. Train the model: python 5_train_model.py")
    print("3. Run inference: python 6_run_inference.py")
    
    return True

if __name__ == "__main__":
    verify_system_setup()