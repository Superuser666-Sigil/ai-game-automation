#!/usr/bin/env python3
"""
4.5_choose_model_configuration.py - Model Configuration Optimizer
Analyzes your training data and system capabilities to recommend the optimal model configuration.
Run this after 4_analyze_data_quality.py to make informed decisions about model size.
"""

import os
import sys
import subprocess
import re
import numpy as np

def check_data_exists():
    """Check if training data exists and analyze basic properties."""
    print("🔍 Checking Training Data...")
    
    # Check both current directory and data_human directory
    data_paths = [
        ("actions.npy", "frames"),  # Current directory
        ("data_human/actions.npy", "data_human/frames")  # data_human directory
    ]
    
    actions_path = None
    frames_dir = None
    
    for actions_file, frames_folder in data_paths:
        if os.path.exists(actions_file) and os.path.exists(frames_folder):
            actions_path = actions_file
            frames_dir = frames_folder
            break
    
    if not actions_path:
        print("❌ No training data found!")
        print("   Please run 3_record_data.py first to record gameplay data.")
        return None
    
    # Load and analyze data
    try:
        actions = np.load(actions_path)
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')]) if os.path.exists(frames_dir) else 0
        
        data_info = {
            'total_frames': len(actions),
            'frame_files': frame_count,
            'data_duration_minutes': len(actions) / 600,  # Assuming 10 FPS
            'actions_shape': actions.shape
        }
        
        print(f"✅ Training data found:")
        print(f"   • {data_info['total_frames']:,} action records")
        print(f"   • {data_info['frame_files']:,} frame images")
        print(f"   • ~{data_info['data_duration_minutes']:.1f} minutes of gameplay")
        
        return data_info
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def analyze_data_complexity():
    """Analyze the complexity of the training data to recommend model size."""
    print("\n📊 Analyzing Data Complexity...")
    
    # Find the data path (same logic as check_data_exists)
    data_paths = [
        "actions.npy",  # Current directory
        "data_human/actions.npy"  # data_human directory
    ]
    
    actions_path = None
    for path in data_paths:
        if os.path.exists(path):
            actions_path = path
            break
    
    if not actions_path:
        print("❌ No actions.npy found for analysis")
        return None
    
    try:
        actions = np.load(actions_path)
        
        # Common keys analysis
        COMMON_KEYS_COUNT = 62
        key_actions = actions[:, :COMMON_KEYS_COUNT]
        mouse_actions = actions[:, COMMON_KEYS_COUNT:COMMON_KEYS_COUNT+2]
        click_actions = actions[:, COMMON_KEYS_COUNT+2:]
        
        # Calculate complexity metrics
        key_press_rate = np.mean(key_actions)
        unique_key_combinations = len(np.unique(key_actions, axis=0))
        mouse_movement_variance = np.var(mouse_actions)
        click_rate = np.mean(click_actions)
        
        # Temporal complexity (how much things change frame to frame)
        key_changes = np.mean(np.abs(np.diff(key_actions, axis=0)))
        mouse_changes = np.mean(np.abs(np.diff(mouse_actions, axis=0)))
        
        complexity_metrics = {
            'key_press_rate': key_press_rate,
            'unique_combinations': unique_key_combinations,
            'mouse_variance': mouse_movement_variance,
            'click_rate': click_rate,
            'temporal_key_changes': key_changes,
            'temporal_mouse_changes': mouse_changes,
            'total_frames': len(actions)
        }
        
        print(f"📈 Data Complexity Analysis:")
        print(f"   • Key press rate: {key_press_rate:.1%}")
        print(f"   • Unique key combinations: {unique_key_combinations:,}")
        print(f"   • Mouse movement variance: {mouse_movement_variance:.3f}")
        print(f"   • Click rate: {click_rate:.1%}")
        print(f"   • Frame-to-frame changes: {key_changes:.3f} (keys), {mouse_changes:.3f} (mouse)")
        
        return complexity_metrics
        
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return None

def detect_system_capabilities():
    """Detect system capabilities for model sizing recommendations."""
    print("\n💻 Detecting System Capabilities...")
    
    # Memory detection
    total_ram_gb = 8  # Default assumption
    try:
        if sys.platform == "win32":
            result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip() and line.strip() != 'TotalPhysicalMemory':
                        total_ram_gb = int(line.strip()) / (1024**3)
                        break
        print(f"   • System RAM: {total_ram_gb:.1f} GB")
    except:
        print(f"   • System RAM: ~{total_ram_gb} GB (estimated)")
    
    # GPU detection (simplified from main script)
    gpu_info = detect_gpu_simple()
    
    return {
        'ram_gb': total_ram_gb,
        'gpu_type': gpu_info['type'],
        'gpu_memory_estimate': gpu_info['memory_estimate']
    }

def detect_gpu_simple():
    """Simplified GPU detection for configuration recommendations."""
    try:
        result = subprocess.run([
            'powershell', '-Command', 
            'Get-WmiObject -Class Win32_VideoController | Select-Object Name | Format-Table -HideTableHeaders'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            gpu_text = result.stdout.lower()
            if re.search(r'nvidia|geforce|rtx|gtx', gpu_text):
                print("   • GPU: NVIDIA detected (excellent for training)")
                return {'type': 'nvidia', 'memory_estimate': 8}
            elif re.search(r'amd|radeon|rx\s*\d+', gpu_text):
                print("   • GPU: AMD detected (limited GPU training support)")
                return {'type': 'amd', 'memory_estimate': 4}
    except:
        pass
    
    print("   • GPU: None detected or CPU only")
    return {'type': 'cpu', 'memory_estimate': 0}

def calculate_model_size(img_width=960, img_height=540, lstm_hidden=256, lstm_layers=2, 
                        cnn_complexity="medium", batch_size=8):
    """Calculate model size - simplified version of the main calculator."""
    
    COMMON_KEYS_COUNT = 62
    
    # CNN parameters based on complexity
    if cnn_complexity == "small":
        cnn_params = 23808  # Pre-calculated
        cnn_output_size = 64 * 6 * 6
    elif cnn_complexity == "medium":
        cnn_params = 95232  # Pre-calculated
        cnn_output_size = 128 * 6 * 6
    else:  # large
        cnn_params = 1691648  # Pre-calculated
        cnn_output_size = 512 * 6 * 6
    
    # LSTM parameters
    lstm_params = lstm_layers * (4 * (cnn_output_size * lstm_hidden + lstm_hidden * lstm_hidden + lstm_hidden * 2))
    
    # Head parameters
    head_params = (lstm_hidden * 128 + 128) + (128 * COMMON_KEYS_COUNT + COMMON_KEYS_COUNT) + \
                  (lstm_hidden * 64 + 64) + (64 * 2 + 2) + \
                  (lstm_hidden * 32 + 32) + (32 * 2 + 2)
    
    total_params = cnn_params + lstm_params + head_params
    
    # File size (MB)
    file_size_mb = (total_params * 4 * 1.15) / (1024 * 1024)
    
    # Memory usage (MB)
    sequence_length = 5
    batch_memory_mb = (batch_size * 3 * img_height * img_width * sequence_length * 4) / (1024 * 1024)
    training_memory_mb = (file_size_mb * 3) + batch_memory_mb
    
    return {
        'total_params': total_params,
        'file_size_mb': file_size_mb,
        'training_memory_mb': training_memory_mb,
        'batch_memory_mb': batch_memory_mb
    }

def recommend_configuration(data_complexity, system_caps):
    """Recommend optimal model configuration based on data and system analysis."""
    print("\n🎯 Model Configuration Recommendations:")
    print("=" * 60)
    
    # Analyze requirements
    if data_complexity is None or system_caps is None:
        print("❌ Cannot make recommendations without data and system analysis")
        return None
    
    # Decision logic
    recommendations = []
    
    # Based on data complexity
    if data_complexity['key_press_rate'] < 0.05:  # < 5% key press rate
        complexity_rec = "small"
        print("📊 Data Analysis → Small model recommended:")
        print("   • Low key press rate suggests simple patterns")
    elif data_complexity['key_press_rate'] > 0.15:  # > 15% key press rate
        complexity_rec = "large"
        print("📊 Data Analysis → Large model recommended:")
        print("   • High key press rate suggests complex patterns")
    else:
        complexity_rec = "medium"
        print("📊 Data Analysis → Medium model recommended:")
        print("   • Moderate complexity patterns detected")
    
    # Based on system capabilities
    if system_caps['ram_gb'] < 4:
        system_rec = "small"
        print("\n💻 System Analysis → Small model required:")
        print("   • Limited RAM detected")
    elif system_caps['ram_gb'] > 16 and system_caps['gpu_type'] == 'nvidia':
        system_rec = "large"
        print("\n💻 System Analysis → Large model possible:")
        print("   • High RAM + NVIDIA GPU detected")
    else:
        system_rec = "medium"
        print("\n💻 System Analysis → Medium model suitable:")
        print("   • Moderate system resources")
    
    # Based on data amount
    if data_complexity['total_frames'] < 1000:
        data_amount_rec = "small"
        print("\n📈 Data Amount → Small model recommended:")
        print("   • Limited training data - larger models may overfit")
    elif data_complexity['total_frames'] > 5000:
        data_amount_rec = "large"
        print("\n📈 Data Amount → Large model possible:")
        print("   • Substantial training data available")
    else:
        data_amount_rec = "medium"
        print("\n📈 Data Amount → Medium model appropriate:")
        print("   • Adequate training data for balanced model")
    
    # Final recommendation (most conservative wins)
    rec_priority = {"small": 1, "medium": 2, "large": 3}
    final_rec = min([complexity_rec, system_rec, data_amount_rec], 
                   key=lambda x: rec_priority[x])
    
    print(f"\n🏆 FINAL RECOMMENDATION: {final_rec.upper()} MODEL")
    
    # Generate specific configurations
    configurations = {
        "small": {
            "name": "Small & Fast",
            "img_size": (640, 360),
            "lstm_hidden": 128,
            "lstm_layers": 1,
            "cnn_complexity": "small",
            "batch_size": 4,
            "description": "Fastest training, lowest memory, good for testing"
        },
        "medium": {
            "name": "Medium (Balanced)",
            "img_size": (960, 540),
            "lstm_hidden": 256,
            "lstm_layers": 2,
            "cnn_complexity": "medium",
            "batch_size": 8,
            "description": "Best balance of speed, memory, and accuracy"
        },
        "large": {
            "name": "Large & Accurate",
            "img_size": (1280, 720),
            "lstm_hidden": 512,
            "lstm_layers": 3,
            "cnn_complexity": "large",
            "batch_size": 4,
            "description": "Best accuracy, requires powerful hardware"
        }
    }
    
    recommended_config = configurations[final_rec]
    
    # Calculate size for recommended configuration
    size_info = calculate_model_size(
        recommended_config["img_size"][0],
        recommended_config["img_size"][1],
        recommended_config["lstm_hidden"],
        recommended_config["lstm_layers"],
        recommended_config["cnn_complexity"],
        recommended_config["batch_size"]
    )
    
    print(f"\n📊 {recommended_config['name']} Configuration:")
    print(f"   • Input Resolution: {recommended_config['img_size'][0]}x{recommended_config['img_size'][1]}")
    print(f"   • LSTM: {recommended_config['lstm_hidden']} hidden, {recommended_config['lstm_layers']} layers")
    print(f"   • CNN Complexity: {recommended_config['cnn_complexity']}")
    print(f"   • Batch Size: {recommended_config['batch_size']}")
    print(f"   • {recommended_config['description']}")
    
    print(f"\n💾 Resource Requirements:")
    print(f"   • Model file size: {size_info['file_size_mb']:.1f} MB")
    print(f"   • Training memory: {size_info['training_memory_mb']:.0f} MB")
    print(f"   • Estimated training time: {estimate_training_time(final_rec, system_caps['gpu_type'])}")
    
    # Memory check
    required_memory_gb = size_info['training_memory_mb'] / 1024
    if required_memory_gb > system_caps['ram_gb'] * 0.8:  # Don't use more than 80% of RAM
        print(f"\n⚠️  WARNING: Training may require {required_memory_gb:.1f} GB but you have {system_caps['ram_gb']:.1f} GB")
        print("   Consider reducing batch size or using smaller model")
    
    return recommended_config, size_info

def estimate_training_time(model_size, gpu_type):
    """Estimate training time based on model size and hardware."""
    base_times = {
        "small": {"nvidia": 5, "amd": 15, "cpu": 30},
        "medium": {"nvidia": 15, "amd": 45, "cpu": 90},
        "large": {"nvidia": 45, "amd": 120, "cpu": 240}
    }
    
    time_minutes = base_times[model_size][gpu_type]
    if time_minutes < 60:
        return f"{time_minutes} minutes"
    else:
        return f"{time_minutes // 60}h {time_minutes % 60}m"

def show_all_configurations():
    """Show comparison of all possible configurations."""
    print("\n📋 All Available Configurations:")
    print("=" * 80)
    
    configs = [
        ("Small & Fast", 640, 360, 128, 1, "small", 4),
        ("Medium (Balanced)", 960, 540, 256, 2, "medium", 8),
        ("Large & Accurate", 1280, 720, 512, 3, "large", 4),
    ]
    
    print(f"{'Configuration':<18} {'Resolution':<12} {'File Size':<10} {'RAM Usage':<11} {'Training Time':<15}")
    print("-" * 80)
    
    for name, w, h, hidden, layers, cnn, batch in configs:
        size_info = calculate_model_size(w, h, hidden, layers, cnn, batch)
        gpu_type = detect_gpu_simple()['type']
        model_size = "small" if "Small" in name else ("large" if "Large" in name else "medium")
        time_est = estimate_training_time(model_size, gpu_type)
        
        print(f"{name:<18} {w}x{h:<7} {size_info['file_size_mb']:<9.1f}MB "
              f"{size_info['training_memory_mb']:<10.0f}MB {time_est:<15}")

def main():
    print("🎯 AI Game Automation - Model Configuration Optimizer")
    print("=" * 70)
    
    # Step 1: Check if training data exists
    data_info = check_data_exists()
    if data_info is None:
        return
    
    # Step 2: Analyze data complexity
    data_complexity = analyze_data_complexity()
    
    # Step 3: Detect system capabilities
    system_caps = detect_system_capabilities()
    
    # Step 4: Make recommendations
    recommended_config, size_info = recommend_configuration(data_complexity, system_caps)
    
    # Step 5: Show all options
    show_all_configurations()
    
    print("\n💡 Next Steps:")
    print("   1. If you agree with the recommendation, proceed to:")
    print("      python 5_train_model.py")
    print("   2. To use a different configuration, edit 5_train_model.py")
    print("   3. To see detailed model size comparisons:")
    print("      python 8_estimate_model_size.py --compare")

if __name__ == "__main__":
    main() 