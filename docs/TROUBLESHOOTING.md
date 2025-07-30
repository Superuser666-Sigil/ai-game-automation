# 🛠️ Troubleshooting Guide

Common issues and solutions for AI Game Automation.

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
# Navigate to scripts directory
cd scripts

# Run the dependency checker
python 1_check_dependencies.py

# Follow the installation commands it provides
# For example: pip install opencv-python numpy torch
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
# Delete old recordings: frames_*.png and actions.npy
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
# Reduce batch size in train_model.py (change BATCH_SIZE = 8 to BATCH_SIZE = 4)
# Use CPU training instead
```

### "Loss not decreasing" or "Bad training results"
**Problem**: Poor data quality or wrong settings
**Solutions**:
```bash
# Check data quality first
python scripts/4_analyze_data_quality.py

# If key press rate < 5%, record more active gameplay
# If data looks good, try longer training (more epochs)

# If loss is stagnating (not changing between epochs):
# - Training parameters have been optimized for better convergence
# - Learning rate reduced from 5e-4 to 1e-4 in config.py
# - Loss weights balanced for better learning
# - Try recording more diverse gameplay data

# Configuration issues:
# - Check if config.py imports correctly
# - Verify all required variables are defined
# - Run validation: python -c "from config import validate_config; validate_config()"
```

## 🎯 AI Performance Issues

### AI doesn't press any keys
**Problem**: Model too conservative
**Solutions**:
```bash
# Check model outputs first
python scripts/7_debug_model_output.py

# If predictions are very low, lower the threshold:
# Edit scripts/6_run_inference.py, change KEY_THRESHOLD = 0.15 to 0.1
```

### Mouse movement is jerky or jumpy
**Problem**: Movement not smooth enough
**Solutions**:
```bash
# Increase mouse smoothing
# Edit scripts/6_run_inference.py, change MOUSE_SMOOTHING_ALPHA = 0.2 to 0.1
# (Lower values = more smoothing)
```

### AI presses keys constantly
**Problem**: Model too aggressive
**Solutions**:
```bash
# Increase confidence threshold
# Edit scripts/6_run_inference.py, change KEY_THRESHOLD = 0.15 to 0.25
# Check training data - might have too many key presses
```

### AI mouse stuck in center of screen
**Problem**: Mouse position prediction failing
**Solutions**:
```bash
# Check if model predicting mouse movement
python scripts/7_debug_model_output.py

# Retrain with more varied mouse movement data
# Ensure game resolution matches training resolution
```

## 💻 GPU and Performance Issues

### "GPU detected but PyTorch using CPU"
**Problem**: Wrong PyTorch installation
**Solution**:
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Reinstall GPU version (follow commands from dependency checker)
python scripts/1_check_dependencies.py
```

### "ROCm not working" (AMD users)
**Problem**: ROCm experimental on Windows
**Solutions**:
```bash
# ROCm on Windows is experimental, try:
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# If still issues, use CPU version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### "DirectML installation issues" (AMD/Intel Windows users)
**Problem**: torch-directml not installing or working
**Solutions**:
```bash
# Ensure Python 3.12 (DirectML requirement)
python --version  # Should show 3.12.x

# Install DirectML in fresh virtual environment:
python -m venv ai_directml_venv
ai_directml_venv\Scripts\activate
pip install torch-directml

# If Python 3.13+, downgrade to 3.12 or use CPU version
```

### "High CPU usage with DirectML" (Normal behavior)
**Problem**: Seeing 80% CPU usage during DirectML training
**Solution**: This is **normal and expected**! DirectML uses both GPU and CPU:
- 55-60% GPU usage = GPU acceleration working
- 80% CPU usage = Full system utilization (good!)
- This indicates optimal performance, not an issue

## ⚙️ Configuration Issues

### "Config import errors" or "Missing variables"
**Problem**: Training script fails to import or find config variables
**Solutions**:
```bash
# Test config import
python -c "from config import *; print('Config works!')"

# If import fails, check file location:
# config.py should be in project root, not scripts/ folder

# Test validation
python -c "from config import validate_config; validate_config()"

# Check for typos in variable names in config.py
```

### "Training crashes with undefined variables"
**Problem**: Script references variables not in config
**Solutions**:
```bash
# Recent fixes added error handling for:
# - Missing IMG_HEIGHT/IMG_WIDTH definitions
# - Undefined all_datasets/all_actions variables
# - Missing SEQUENCE_LENGTH

# Update to latest version or check scripts/5_train_model.py for fixes
```

## 🔧 Game-Specific Issues

### "AI doesn't work in [specific game]"
**Possible causes**:
1. **Anti-cheat blocking**: Some games block automation
2. **Different resolution**: Train and run at same resolution
3. **Game too fast**: Record slower-paced gameplay first
4. **Complex UI**: Train on simple scenarios first

**Solutions**:
```bash
# Start with simpler games (puzzle, strategy games)
# Train on specific game scenes (combat, exploration separately)
# Ensure consistent resolution and graphics settings
```

### Recording doesn't capture game
**Problem**: Game in exclusive fullscreen
**Solution**:
```bash
# Change game to "Windowed" or "Borderless Window" mode
# This allows screen capture to work properly
```

## 📞 Getting Help

### If nothing works:
1. **Run diagnostics**: `python scripts/1_check_dependencies.py` and `python scripts/2_verify_system_setup.py`
2. **Check data quality**: `python scripts/4_analyze_data_quality.py`
3. **Debug model**: `python scripts/7_debug_model_output.py`
4. **Start simple**: Try with a simple game first

### Common error solutions:
- **"File not found"**: Make sure you're in the right folder
- **"Permission denied"**: Run as administrator
- **"Out of memory"**: Close other programs, reduce batch size
- **"No GPU detected"**: Follow PyTorch GPU installation instructions

### Performance expectations:
- **Training time**: 10 minutes (GPU) to 3 hours (CPU)
- **AI response time**: Under 100ms
- **Accuracy**: 70-85% for a well-trained model
- **Memory usage**: 2-4GB during training, 1GB during inference

---

**Still having issues?** Open an issue on GitHub with:
- Your system specifications
- Error messages (copy the full text)
- Steps you've already tried
- Screenshots/videos if helpful 