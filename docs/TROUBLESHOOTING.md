# 🛠️ Troubleshooting Guide

Common issues and solutions for AI Game Automation v2.0.

## 🔍 Installation Issues

### "pip not found" or "python not found"
**Problem**: Python not installed properly
**Solution**: 
```bash
# Download Python from python.org
# During installation, check "Add Python to PATH"
# Restart your computer after installation
```

### "Permission denied" or "Access denied"
**Problem**: Windows blocking the scripts
**Solutions**:
```bash
# Option 1: Run as Administrator
# Right-click Command Prompt → "Run as administrator"

# Option 2: Allow through Windows Security
# Windows Security → Virus Protection → Allow an app
```

### "Module not found" errors
**Problem**: Dependencies not installed
**Solution**:
```bash
# Run the setup script
python scripts/0_setup.py

# Or check dependencies manually
python scripts/1_check_dependencies.py

# Install missing packages
pip install -r requirements.txt
```

### "scikit-learn not found" error
**Problem**: New dependency missing (added in v2.0)
**Solution**:
```bash
pip install scikit-learn>=1.3.0
```

## 🎮 Recording Issues

### "No frames captured" or "Empty recording"
**Problem**: Screen capture not working
**Solutions**:
```bash
# Check if running as administrator
# Try different games (some block screen capture)
# Ensure game is in windowed or borderless mode
```

### "Recording too slow" or "Frames dropping"
**Problem**: Computer too slow for 10 FPS capture
**Solutions**:
```bash
# Close other programs during recording
# Lower screen resolution in game
# Record shorter sessions (2-3 minutes)
```

### Recording file too large
**Problem**: Large video files filling disk
**Solutions**:
```bash
# Record shorter sessions
# Delete old recordings: data_human/frames/ and data_human/actions.npy
# Use lower game resolution
```

## 🧠 Training Problems

### Training extremely slow
**Problem**: Using CPU instead of GPU
**Solutions**:
```bash
# Run GPU detection
python scripts/1_check_dependencies.py

# Install GPU version of PyTorch (follow the specific commands shown)
# For NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu121
# For AMD/Intel: pip install torch-directml (Windows DirectML acceleration)
# For AMD Linux: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

### "CUDA out of memory"
**Problem**: GPU memory full
**Solutions**:
```bash
# Close other programs using GPU (games, browsers)
# Reduce batch size in config.py: BATCH_SIZE = 8
# Use CPU training instead
```

### "Loss not decreasing" or "Bad training results"
**Problem**: Poor data quality or wrong settings
**Solutions**:
```bash
# Check data quality first
python scripts/4_analyze_data_quality.py

# Verify configuration
python scripts/2_verify_system_setup.py

# Adjust oversampling if needed
# In config.py: OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 20
```

### "Model not learning keys"
**Problem**: Class imbalance (common issue)
**Solutions**:
```bash
# The new oversampling should fix this automatically
# If still having issues, increase oversampling:
# In config.py: OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 25

# Lower the key threshold for inference:
# In config.py: KEY_THRESHOLD = 0.15

# Record more data with key presses
```

## 🎯 Inference Issues

### "AI not pressing keys"
**Problem**: Threshold too high or model not trained well
**Solutions**:
```bash
# Lower the key threshold in config.py
KEY_THRESHOLD = 0.15  # Try 0.1-0.2 range

# Check model predictions
python scripts/7_debug_model_output.py

# Retrain with better data or more oversampling
```

### "Jittery mouse movement"
**Problem**: Mouse smoothing settings
**Solutions**:
```bash
# Adjust smoothing in config.py
MOUSE_SMOOTHING_ALPHA = 0.1  # Lower = smoother
SMOOTH_FACTOR = 0.8          # Higher = smoother
```

### "AI pressing wrong keys"
**Problem**: Model confusion or threshold issues
**Solutions**:
```bash
# Increase threshold to be more selective
KEY_THRESHOLD = 0.25

# Check which keys are being confused
python scripts/7_debug_model_output.py

# Retrain with cleaner data (fewer accidental key presses)
```

## ⚙️ Configuration Issues

### "Configuration validation failed"
**Problem**: Settings conflict or missing values
**Solutions**:
```bash
# Run configuration validation
python scripts/2_verify_system_setup.py

# Check config.py for missing or invalid settings
# Ensure all required variables are defined
```

### "Model architecture mismatch"
**Problem**: Different model definitions between scripts
**Solutions**:
```bash
# This should be fixed in v2.0 with centralized config
# If still having issues, ensure all scripts use the same config.py
# Check that model class definitions match across scripts
```

## 📊 Data Quality Issues

### "Very low key press rate"
**Problem**: Not enough action data for training
**Solutions**:
```bash
# Record more active gameplay
# Include more key presses in your recording
# The oversampling will help, but you still need some key presses

# Check your data:
python scripts/4_analyze_data_quality.py
```

### "Actions and frames count mismatch"
**Problem**: Recording was interrupted or corrupted
**Solutions**:
```bash
# The system will use the smaller count automatically
# For better results, record a new clean session
# Ensure recording completes properly (press F2 to stop)
```

## 🚀 Performance Issues

### "Training takes too long"
**Problem**: Inefficient settings or hardware
**Solutions**:
```bash
# Enable GPU acceleration
python scripts/1_check_dependencies.py

# Optimize settings in config.py:
BATCH_SIZE = 32          # Larger batches
EPOCHS = 5               # Fewer epochs
TRAIN_IMG_WIDTH = 160    # Smaller images
TRAIN_IMG_HEIGHT = 160
```

### "Inference is laggy"
**Problem**: Too high FPS or resolution
**Solutions**:
```bash
# Lower inference FPS in config.py:
INFERENCE_FPS = 5        # Reduce from 10

# Lower resolution:
IMG_WIDTH = 640          # Reduce from 960
IMG_HEIGHT = 360         # Reduce from 540
```

## 🔧 Advanced Issues

### "Validation F1-score not improving"
**Problem**: Overfitting or poor data quality
**Solutions**:
```bash
# Check validation split is working
# In config.py: VALIDATION_SPLIT = 0.15

# Record more diverse data
# Try different oversampling values
# Check for data leakage between train/val sets
```

### "Model saves but performance is poor"
**Problem**: Model saved based on wrong metric
**Solutions**:
```bash
# The new system saves based on F1-score
# Check that validation data is representative
# Try different thresholds for inference
```

### "Memory usage too high"
**Problem**: Batch size or image size too large
**Solutions**:
```bash
# Reduce memory usage in config.py:
BATCH_SIZE = 8           # Smaller batches
TRAIN_IMG_WIDTH = 160    # Smaller images
TRAIN_IMG_HEIGHT = 160
SEQUENCE_LENGTH = 3      # Shorter sequences
```

## 📋 Quick Fix Checklist

Before asking for help, try these steps:

1. **✅ Run system verification**:
   ```bash
   python scripts/2_verify_system_setup.py
   ```

2. **✅ Check data quality**:
   ```bash
   python scripts/4_analyze_data_quality.py
   ```

3. **✅ Validate configuration**:
   ```bash
   python scripts/1_check_dependencies.py
   ```

4. **✅ Debug model output**:
   ```bash
   python scripts/7_debug_model_output.py
   ```

5. **✅ Check documentation**:
   - [Configuration Guide](CONFIGURATION.md)
   - [Refactor Summary](../REFACTOR_SUMMARY.md)
   - [Project Cleanup](../PROJECT_CLEANUP.md)

## 🆘 Getting Help

If you're still having issues:

1. **Check the logs**: Look for error messages in the terminal output
2. **Verify your setup**: Run all verification scripts
3. **Check your data**: Ensure you have good quality training data
4. **Review configuration**: Make sure settings match your hardware
5. **Open an issue**: Include error messages and system information

**Common solutions for v2.0:**
- Use the new oversampling feature for better key detection
- Adjust thresholds based on your game type
- Enable GPU acceleration for faster training
- Use the centralized configuration system

---

**Still stuck?** The refactored system should be much more reliable. If you're having issues, it's likely a configuration or data quality problem that can be easily fixed! 