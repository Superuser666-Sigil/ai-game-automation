# 🎮 AI Game Automation - Complete Setup Guide

Train an AI to play games by learning from your gameplay recordings. This project captures your screen, keyboard, and mouse actions, then trains a neural network to mimic your behavior.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/ai-game-automation.git
cd ai-game-automation/scripts

# Follow the numbered workflow
python 1_check_dependencies.py     # Check your system
python 2_verify_system_setup.py    # Verify setup
python 3_record_data.py             # Record gameplay
python 4_analyze_data_quality.py    # Analyze data quality
python 4.5_choose_model_configuration.py  # Choose model size
python 5_train_model.py             # Train the AI
python 6_run_inference.py           # Run the AI
```

## 📋 What You Need

- **Windows 10/11** (currently Windows-only)
- **Python 3.8+**
- **4GB+ RAM** (8GB+ recommended)
- **GPU** (NVIDIA/AMD) for faster training, or CPU works too

## 🔧 Installation

### Step 1: Clone and Setup
```bash
# Clone from GitHub
git clone https://github.com/your-username/ai-game-automation.git
cd ai-game-automation

# Navigate to scripts directory
cd scripts

# Run the setup script (installs dependencies automatically)
python 0_setup.py
```

### Step 2: Check Your Hardware
```bash
# This script will detect your GPU and recommend the best PyTorch installation
python 1_check_dependencies.py
```

**What happens**: The script automatically detects if you have:
- **NVIDIA GPU**: Recommends CUDA installation for 5-10x faster training
- **AMD GPU**: Checks ROCm compatibility for potential GPU acceleration  
- **CPU only**: Provides CPU-optimized installation (works fine, just slower)

### Step 3: Verify Everything Works
```bash
python 2_verify_system_setup.py
```

## 🎮 Recording Your Gameplay

### Basic Recording
```bash
python 3_record_data.py
```

**What to do**:
1. **Start the script** - it will count down 5 seconds
2. **Switch to your game** 
3. **Play normally** for 5-10 minutes
4. **Press Ctrl+C** to stop recording

**Recording Tips**:
- **Play naturally** - don't try to press keys constantly
- **Include variety** - different scenarios, speeds, actions
- **Use the keys you want the AI to learn** (WASD, space, mouse clicks)
- **Good lighting** - avoid very dark or very bright scenes

### What Gets Recorded
- **Screen captures** at 10 frames per second
- **All keyboard presses** (with timing)
- **Mouse movement and clicks** (with positions)
- **Automatic synchronization** between video and actions

## 📊 Checking Your Data

### Analyze Data Quality
```bash
python 4_analyze_data_quality.py
```

**What you'll see**:
- **Key press frequency** (should be 5-15% for good training)
- **Mouse movement patterns** (should show natural movement)
- **Data visualization** (timeline graphs, heatmaps)
- **Quality recommendations**

**Good data signs**:
- Key press rate: 5-15%
- Smooth mouse movement patterns
- Variety in actions and scenarios
- 1000+ frames recorded

**Bad data signs**:
- Key press rate < 2% (too little action)
- Mouse stuck in one area
- Very repetitive patterns
- < 500 frames recorded

## 🧠 Training Your AI

### Start Training
```bash
python 5_train_model.py
```

**What happens during training**:
1. **Data loading** - loads your recorded gameplay
2. **GPU detection** - automatically uses your best hardware
3. **Smart training** - uses advanced techniques to handle rare events (like key presses)
4. **Progress monitoring** - shows real-time training progress
5. **Model saving** - saves the trained AI as `model_improved.pt`

**Training progress example**:
```
Epoch 1/50
  Total Loss: 2.34
  Key Loss: 1.23 (learning when to press keys)
  Mouse Loss: 0.89 (learning mouse movement)
  Click Loss: 0.22 (learning when to click)

Epoch 25/50
  Total Loss: 0.67
  Key Loss: 0.34
  Mouse Loss: 0.21
  Click Loss: 0.12

Epoch 50/50 ✅ TRAINING COMPLETE
  Total Loss: 0.45
  Key Loss: 0.23
  Mouse Loss: 0.15
  Click Loss: 0.07
```

**Training times**:
- **With GPU**: 10-30 minutes
- **CPU only**: 1-3 hours

## 🎯 Running Your AI

### Start AI Gameplay
```bash
python 6_run_inference.py
```

**What happens**:
1. **Model loading** - loads your trained AI
2. **5-second countdown** - time to switch to your game
3. **AI takes control** - captures screen and controls keyboard/mouse
4. **Press Ctrl+C** to stop

**Features included**:
- **Smart mouse smoothing** - prevents jerky movement
- **Intelligent key pressing** - only presses keys when confident
- **Real-time performance** - responds within 100ms
- **Safe operation** - easy to stop with Ctrl+C

## 🛠️ Troubleshooting

### 🔍 Dependencies and Installation Issues

#### "pip not found" or "python not found"
**Problem**: Python not installed properly
**Solution**: 
```bash
# Download Python from python.org
# During installation, check "Add Python to PATH"
# Restart your computer after installation
```

#### "Permission denied" or "Access denied"
**Problem**: Windows blocking the scripts
**Solutions**:
```bash
# Option 1: Run as Administrator
# Right-click Command Prompt → "Run as administrator"

# Option 2: Allow through Windows Security
# Windows Security → Virus Protection → Allow an app
```

#### "Module not found" errors
**Problem**: Dependencies not installed
**Solution**:
```bash
# Run the dependency checker
python 1_check_dependencies.py

# Follow the installation commands it provides
# For example: pip install opencv-python numpy torch
```

### 🎮 Recording Issues

#### "No frames captured" or "Empty recording"
**Problem**: Screen capture not working
**Solutions**:
```bash
# Check if running as administrator
# Try different games (some block screen capture)
# Ensure game is in windowed or borderless mode
```

#### "Recording too slow" or "Frames dropping"
**Problem**: Computer too slow for 10 FPS capture
**Solutions**:
```bash
# Close other programs during recording
# Lower screen resolution in game
# Record shorter sessions (2-3 minutes)
```

#### Recording file too large
**Problem**: Large video files filling disk
**Solutions**:
```bash
# Record shorter sessions
# Delete old recordings: frames_*.png and actions.npy
# Use lower game resolution
```

### 🧠 Training Problems

#### Training extremely slow
**Problem**: Using CPU instead of GPU
**Solutions**:
```bash
# Run GPU detection
python 1_check_dependencies.py

# Install GPU version of PyTorch (follow the specific commands shown)
# For NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu121
# For AMD: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
```

#### "CUDA out of memory"
**Problem**: GPU memory full
**Solutions**:
```bash
# Close other programs using GPU (games, browsers)
# Reduce batch size in train_model.py (change BATCH_SIZE = 8 to BATCH_SIZE = 4)
# Use CPU training instead: add --device cpu to training command
```

#### "Loss not decreasing" or "Bad training results"
**Problem**: Poor data quality or wrong settings
**Solutions**:
```bash
# Check data quality first
python 4_analyze_data_quality.py

# If key press rate < 5%, record more active gameplay
# If data looks good, try longer training (more epochs)
```

### 🎯 AI Performance Issues

#### AI doesn't press any keys
**Problem**: Model too conservative
**Solutions**:
```bash
# Check model outputs first
python 7_debug_model_output.py

# If predictions are very low, lower the threshold:
# Edit run_inference.py, change KEY_THRESHOLD = 0.15 to 0.1
```

#### Mouse movement is jerky or jumpy
**Problem**: Movement not smooth enough
**Solutions**:
```bash
# Increase mouse smoothing
# Edit run_inference.py, change MOUSE_SMOOTHING_ALPHA = 0.2 to 0.1
# (Lower values = more smoothing)
```

#### AI presses keys constantly
**Problem**: Model too aggressive
**Solutions**:
```bash
# Increase confidence threshold
# Edit run_inference.py, change KEY_THRESHOLD = 0.15 to 0.25
# Check training data - might have too many key presses
```

#### AI mouse stuck in center of screen
**Problem**: Mouse position prediction failing
**Solutions**:
```bash
# Check if model predicting mouse movement
python 7_debug_model_output.py

# Retrain with more varied mouse movement data
# Ensure game resolution matches training resolution
```

### 💻 GPU and Performance Issues

#### "GPU detected but PyTorch using CPU"
**Problem**: Wrong PyTorch installation
**Solution**:
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision -y

# Reinstall GPU version (follow commands from dependency checker)
python 1_check_dependencies.py
```

#### "ROCm not working" (AMD users)
**Problem**: ROCm experimental on Windows
**Solutions**:
```bash
# ROCm on Windows is experimental, try:
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# If still issues, use CPU version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 🔧 Game-Specific Issues

#### "AI doesn't work in [specific game]"
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

#### Recording doesn't capture game
**Problem**: Game in exclusive fullscreen
**Solution**:
```bash
# Change game to "Windowed" or "Borderless Window" mode
# This allows screen capture to work properly
```

## 📈 Performance Optimization

### For Faster Training
```bash
# Use GPU (biggest speedup)
python 1_check_dependencies.py  # Follow GPU setup instructions

# Reduce model size (in train_model.py)
BATCH_SIZE = 4  # Smaller batches
EPOCHS = 25     # Fewer epochs for testing
```

### For Better AI Performance
```bash
# Record higher quality data
# - More varied scenarios
# - Longer recording sessions
# - Consistent gameplay speed

# Fine-tune thresholds (in run_inference.py)
KEY_THRESHOLD = 0.15      # Adjust based on testing
MOUSE_SMOOTHING_ALPHA = 0.2  # Adjust mouse smoothness
```

### For Lower Resource Usage
```bash
# Reduce capture resolution (in record_data.py)
IMG_WIDTH = 320   # Smaller images
IMG_HEIGHT = 240

# Lower capture rate
FPS = 5  # Instead of 10 FPS
```

## 🎮 Game-Specific Tips

### Strategy Games (Age of Empires, StarCraft)
- **Record**: Base building, unit management, combat
- **Focus**: Mouse clicks, hotkeys (1-9), precise positioning
- **Tips**: Train on one game mode first

### FPS Games (Counter-Strike, Valorant) 
- **Record**: Aiming, movement, shooting
- **Focus**: Mouse precision, WASD movement, shooting timing
- **Tips**: Record different maps and situations

### RPG Games (Skyrim, Witcher)
- **Record**: Exploration, combat, inventory management
- **Focus**: Movement, interaction keys, combat sequences
- **Tips**: Train on specific activities separately

### Puzzle Games (Portal, Tetris)
- **Record**: Problem-solving sequences
- **Focus**: Precise timing, specific key combinations
- **Tips**: Great for beginners - simpler patterns

## 📞 Getting Help

### If nothing works:
1. **Run diagnostics**: `python 1_check_dependencies.py` and `python 2_verify_system_setup.py`
2. **Check data quality**: `python 4_analyze_data_quality.py`
3. **Debug model**: `python 7_debug_model_output.py`
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

### Advanced users:
- **Model size estimation**: Use `4.5_choose_model_configuration.py` after analyzing your data
- **Configuration comparison**: The 4.5 script provides detailed size and performance comparisons
- **Custom model sizes**: Edit parameters in `5_train_model.py` based on recommendations

## 📁 Project Structure

```
ai-game-automation/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── scripts/                    # Main Python scripts
│   ├── 0_setup.py             # Initial setup
│   ├── 1_check_dependencies.py # Hardware detection
│   ├── 2_verify_system_setup.py # System verification
│   ├── 3_record_data.py       # Data recording
│   ├── 4_analyze_data_quality.py # Data analysis
│   ├── 4.5_choose_model_configuration.py # Model sizing
│   ├── 5_train_model.py       # Training script
│   ├── 6_run_inference.py     # AI execution
│   ├── 7_debug_model_output.py # Debugging tools
│   └── requirements.txt       # Python dependencies
├── docs/                      # Documentation
│   ├── TROUBLESHOOTING.md     # Problem-solving guide
│   ├── CONFIGURATION.md       # Advanced configuration
│   └── CONTRIBUTING.md        # Contribution guidelines
└── examples/                  # Example files and datasets
    ├── README.md             # Examples guide
    ├── sample_recordings/    # Sample training data
    └── trained_models/       # Pre-trained models
```

## 🤝 Contributing

We welcome contributions! Whether you're:
- 🐛 **Reporting bugs** or requesting features
- 📝 **Improving documentation** or tutorials  
- 🎮 **Sharing example datasets** from different games
- 🧠 **Contributing code** improvements or new features
- 🎯 **Testing on different hardware** configurations

See our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to get started.

## 📖 Documentation

- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Solutions for common issues
- **[Configuration Guide](docs/CONFIGURATION.md)** - Advanced customization options
- **[Examples](examples/README.md)** - Sample data and pre-trained models
- **[Contributing](docs/CONTRIBUTING.md)** - How to contribute to the project

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch for deep learning
- Uses OpenCV for computer vision
- Inspired by behavior cloning research
- Thanks to the open-source community

---

**Remember**: Training an AI to play games is an iterative process. Start with simple games and short recordings, then gradually increase complexity as you get better results. The key is good training data and patience! 🎮✨