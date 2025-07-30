#!/usr/bin/env python3
"""
Test script to check all required imports for the AI game automation project.
Run this to see what dependencies need to be installed.
Includes intelligent GPU detection for optimal PyTorch installation.
"""

import sys
import subprocess
import re

def detect_gpu_type():
    """Detect GPU type (NVIDIA, AMD, or None) on Windows."""
    print("\n🔍 Detecting GPU hardware...")
    
    nvidia_gpus = []
    amd_gpus = []
    
    # Method 1: Try wmic with simpler query first
    try:
        result = subprocess.run(
            ['wmic', 'path', 'win32_VideoController', 'get', 'Name'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line or 'Name' in line:
                    continue
                
                print(f"   🔍 Found: {line}")  # Debug output
                
                # Check for NVIDIA GPUs
                if re.search(r'NVIDIA|GeForce|RTX|GTX|Quadro|Tesla', line, re.IGNORECASE):
                    nvidia_gpus.append(line)
                
                # Check for AMD GPUs  
                elif re.search(r'AMD|Radeon|RX\s*\d+|RDNA', line, re.IGNORECASE):
                    amd_gpus.append(line)
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  Could not detect GPU via wmic: {e}")
    
    # Method 2: Try PowerShell Get-WmiObject (more reliable on modern Windows)
    if not nvidia_gpus and not amd_gpus:
        try:
            result = subprocess.run([
                'powershell', '-Command', 
                'Get-WmiObject -Class Win32_VideoController | Select-Object Name | Format-Table -HideTableHeaders'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    print(f"   🔍 Found: {line}")  # Debug output
                    
                    # Check for NVIDIA GPUs
                    if re.search(r'NVIDIA|GeForce|RTX|GTX|Quadro|Tesla', line, re.IGNORECASE):
                        nvidia_gpus.append(line)
                    
                    # Check for AMD GPUs
                    elif re.search(r'AMD|Radeon|RX\s*\d+|RDNA', line, re.IGNORECASE):
                        amd_gpus.append(line)
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"⚠️  Could not detect GPU via PowerShell: {e}")
    
    # Method 3: Alternative using DirectX diagnostic tool
    if not nvidia_gpus and not amd_gpus:
        try:
            result = subprocess.run(['dxdiag', '/t', 'dxdiag_temp.txt'], 
                                  capture_output=True, timeout=15)
            if result.returncode == 0:
                try:
                    with open('dxdiag_temp.txt', 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if re.search(r'NVIDIA|GeForce|RTX|GTX', content, re.IGNORECASE):
                            nvidia_gpus.append("NVIDIA GPU (detected via DirectX)")
                        elif re.search(r'AMD|Radeon|RX\s*\d+', content, re.IGNORECASE):
                            amd_gpus.append("AMD GPU (detected via DirectX)")
                except FileNotFoundError:
                    pass
                finally:
                    try:
                        import os
                        os.remove('dxdiag_temp.txt')
                    except:
                        pass
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass
    
    # Display results
    if nvidia_gpus:
        print("🟢 NVIDIA GPU(s) detected:")
        for gpu in nvidia_gpus[:3]:  # Limit to 3 for readability
            print(f"   • {gpu}")
        return "nvidia"
    elif amd_gpus:
        print("🔴 AMD GPU(s) detected:")
        for gpu in amd_gpus[:3]:
            print(f"   • {gpu}")
        return "amd"
    else:
        print("⚪ No dedicated GPU detected - will use CPU")
        return "cpu"

def get_pytorch_recommendation(gpu_type, amd_gpus=None):
    """Get PyTorch installation recommendation based on GPU type."""
    print(f"\n💡 PyTorch Installation Recommendation (GPU type: {gpu_type}):")
    print("=" * 60)
    
    if gpu_type == "nvidia":
        print("🟢 NVIDIA GPU detected - CUDA support recommended")
        print("📦 Recommended installation:")
        print("   pip uninstall torch torchvision -y")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\n🔧 Alternative CUDA versions:")
        print("   • CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121")
        print("   • CUDA 11.8: --index-url https://download.pytorch.org/whl/cu118")
        print("\n⚡ Benefits: 5-10x faster training and inference")
        
    elif gpu_type == "amd":
        print("🔴 AMD GPU detected - ROCm support evaluation")
        print("📦 ROCm compatibility check:")
        
        # Check if detected AMD GPUs support ROCm
        if amd_gpus:
            detected_gpu_names = ' '.join(amd_gpus).lower()
        else:
            detected_gpu_names = ""
        print(f"   🔍 Checking ROCm compatibility for: {detected_gpu_names}")
        
        # ROCm-compatible GPUs on Windows (RX 6000 series and newer)
        rocm_patterns = [
            r'rx\s*79\d\d',      # RX 7900 series
            r'rx\s*78\d\d',      # RX 7800 series  
            r'rx\s*77\d\d',      # RX 7700 series
            r'rx\s*76\d\d',      # RX 7600 series
            r'rx\s*69\d\d',      # RX 6950 series
            r'rx\s*68\d\d',      # RX 6800 series
            r'rx\s*67\d\d',      # RX 6700 series
            r'rx\s*66\d\d',      # RX 6600 series (including 6650M!)
            r'rx\s*65\d\d',      # RX 6500 series
            r'rx\s*64\d\d',      # RX 6400 series
        ]
        
        rocm_compatible = any(re.search(pattern, detected_gpu_names) for pattern in rocm_patterns)
        
        if rocm_compatible:
            print("   ✅ Your AMD GPU may support ROCm on Windows!")
            print("   📦 Try ROCm installation:")
            print("      pip uninstall torch torchvision -y")
            print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0")
            print("   ⚠️  ROCm on Windows is experimental - fallback to CPU if issues occur")
            print("   📖 Note: RX 6000+ series has limited but growing ROCm support")
        else:
            print("   ❌ Your AMD GPU likely doesn't support ROCm on Windows")
            print("   💭 Recommendation: Use CPU version for better stability")
            
        print("\n📦 Fallback CPU installation:")
        print("   pip uninstall torch torchvision -y")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        
    else:  # cpu
        print("⚪ No dedicated GPU detected - CPU version recommended")
        print("📦 Recommended installation:")
        print("   pip uninstall torch torchvision -y")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("\n💭 Note: CPU inference is slower but will work for smaller models")

def check_current_pytorch():
    """Check current PyTorch installation and GPU support."""
    try:
        import torch
        print(f"\n🔍 Current PyTorch: {torch.__version__}")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   • GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚪ CUDA not available - using CPU")
            
        # Check for ROCm (AMD)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"🔴 ROCm detected: {torch.version.hip}")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError:
        print(f"✗ {package_name} NOT FOUND - needs to be installed")
        return False
    except Exception as e:
        print(f"✗ {package_name} ERROR: {e}")
        return False



def main():
    print("🔍 AI Game Automation - Dependency & GPU Detection")
    print("=" * 60)
    
    # Step 1: Detect GPU type
    gpu_type = detect_gpu_type()
    
    # Step 2: Check current PyTorch installation
    pytorch_installed = check_current_pytorch()
    
    # Step 3: Provide recommendations  
    # Re-detect AMD GPUs for the recommendation function
    amd_gpus_for_recommendation = []
    if gpu_type == "amd":
        try:
            result = subprocess.run([
                'powershell', '-Command', 
                'Get-WmiObject -Class Win32_VideoController | Select-Object Name | Format-Table -HideTableHeaders'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line and re.search(r'AMD|Radeon|RX\s*\d+|RDNA', line, re.IGNORECASE):
                        amd_gpus_for_recommendation.append(line)
        except:
            pass
    
    get_pytorch_recommendation(gpu_type, amd_gpus_for_recommendation)
    
    # Model size calculation moved to 4.5_choose_model_configuration.py
    
    print("\n" + "=" * 60)
    print("🔍 Testing other dependencies...")
    
    # Core dependencies (excluding torch/torchvision as they need special handling)
    required_modules = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("mss", "mss"),
        ("pynput", "pynput"), 
        ("matplotlib", "matplotlib"),
        ("PIL", "Pillow"),
    ]
    
    missing_modules = []
    
    for module_name, package_name in required_modules:
        if not test_import(module_name, package_name):
            missing_modules.append(package_name)
    
    # Test PyTorch separately
    if not test_import("torch", "PyTorch"):
        missing_modules.append("torch")
    if not test_import("torchvision", "torchvision"):
        missing_modules.append("torchvision")
    
    print("\n" + "=" * 60)
    
    if missing_modules:
        print("❌ MISSING DEPENDENCIES:")
        non_torch_missing = [m for m in missing_modules if m not in ['torch', 'torchvision']]
        
        if non_torch_missing:
            print("📦 Install basic dependencies:")
            print(f"   pip install {' '.join(non_torch_missing)}")
        
        if 'torch' in missing_modules or 'torchvision' in missing_modules:
            print("🔧 For PyTorch, use the GPU-specific command above ☝️")
    else:
        print("✅ ALL DEPENDENCIES INSTALLED!")
        print("🎮 You're ready to run the AI game automation scripts!")
        
        # Additional performance check
        if pytorch_installed:
            import torch
            if gpu_type == "nvidia" and not torch.cuda.is_available():
                print("\n⚠️  PERFORMANCE WARNING:")
                print("   You have an NVIDIA GPU but PyTorch isn't using CUDA.")
                print("   Consider reinstalling PyTorch with CUDA support for better performance.")
            elif gpu_type == "amd" and not (torch.cuda.is_available() or hasattr(torch.version, 'hip')):
                print("\n💭 PERFORMANCE NOTE:")
                print("   You have an AMD GPU but PyTorch is using CPU.")
                print("   This is normal - ROCm support on Windows is limited.")
    
    print("\n📋 Available scripts (in recommended order):")
    print("• 1_check_dependencies.py - Check dependencies (this script)")
    print("• 2_verify_system_setup.py - Verify system setup")
    print("• 3_record_data.py - Record gameplay data")
    print("• 4_analyze_data_quality.py - Analyze data quality")
    print("• 4.5_choose_model_configuration.py - Choose optimal model size")
    print("• 5_train_model.py - Train improved model")
    print("• 6_run_inference.py - Run improved inference")
    print("• 7_debug_model_output.py - Debug model predictions")

if __name__ == "__main__":
    main()