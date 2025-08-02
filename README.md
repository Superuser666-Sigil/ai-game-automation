# 🤖 AI Game Automation

Train an AI to play games by learning from your gameplay! This project uses behavior cloning with advanced oversampling techniques to teach neural networks to mimic human gameplay patterns.

## ✨ Features

- **🎯 Easy Setup**: Numbered scripts guide you through the entire process
- **🧠 Advanced Training**: Oversampling and validation for better model performance
- **⚙️ Centralized Configuration**: Single `config.py` file controls all settings
- **📊 Data Quality Analysis**: Ensures your training data is good before training
- **🤖 Intelligent Model Architecture**: LSTM-based neural networks for temporal learning
- **🎮 Real-time Inference**: Smooth mouse movement with temporal smoothing
- **🛠️ Comprehensive Debugging**: Tools to understand and improve your AI's performance
- **⚡ Optimized Training**: Class balancing, validation splits, and adaptive learning rates
- **🚀 GPU Acceleration**: Support for CUDA, DirectML, and CPU training

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/ai-game-automation.git
cd ai-game-automation

# Follow the numbered workflow
python scripts/0_setup.py                           # Install dependencies
python scripts/2_verify_system_setup.py             # Verify everything works
python scripts/3_record_data.py                     # Record 5-10 minutes of gameplay
python scripts/4_analyze_data_quality.py            # Check your recording quality
python scripts/5_train_model.py                     # Train your AI (10-60 minutes)
python scripts/6_run_inference.py                   # Watch your AI play!
```

## 📋 Requirements

- **Windows 10/11** (primary platform)
- **Python 3.8+** (3.12 recommended)
- **4GB+ RAM** (8GB+ recommended for training)
- **Optional**: NVIDIA GPU for CUDA acceleration
- **Optional**: AMD/Intel GPU for DirectML acceleration

## 🎯 How It Works

1. **📹 Record**: Capture your screen and input actions while playing
2. **📊 Analyze**: Check data quality and get optimization recommendations  
3. **🧠 Train**: Neural network learns to map screen images to your actions with oversampling (10-60 minutes with GPU)
4. **🎮 Play**: AI takes control with smooth mouse movement and responsive key presses

## 🔧 Key Improvements in v2.0

### **Advanced Training Features**
- **Dataset Oversampling**: Frames with actions are repeated 15x more to fix class imbalance
- **Validation Split**: 15% of data used for validation to prevent overfitting
- **F1-Score Based Saving**: Models are saved based on validation performance
- **Simplified Architecture**: Consistent LSTM-based model across all scripts

### **Better Configuration**
- **Centralized Settings**: All parameters in `config.py`
- **Automatic Validation**: Configuration consistency checking
- **Improved Thresholds**: Optimized key detection and mouse smoothing

### **Enhanced Debugging**
- **Model Output Analysis**: Detailed prediction analysis
- **Data Quality Metrics**: Comprehensive data validation
- **Performance Monitoring**: Training progress tracking

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| **[🛠️ Troubleshooting](docs/TROUBLESHOOTING.md)** | Solutions for common issues |
| **[⚙️ Configuration](docs/CONFIGURATION.md)** | Advanced settings and game-specific configs |
| **[📁 Examples](examples/README.md)** | Sample datasets and pre-trained models |
| **[🤝 Contributing](docs/CONTRIBUTING.md)** | How to contribute to the project |
| **[🔄 Refactor Summary](REFACTOR_SUMMARY.md)** | Detailed refactoring documentation |
| **[🧹 Project Cleanup](PROJECT_CLEANUP.md)** | Cleanup and modernization summary |

## 🎮 Supported Games

**Works well with:**
- Turn-based strategy games
- Puzzle games  
- Slower-paced RPGs
- Any game in windowed/borderless mode

**Experimental:**
- Fast-paced FPS games
- Real-time strategy games
- Action games

## 🤝 Contributing

We welcome all types of contributions:
- 🐛 Bug reports and feature requests
- 📝 Documentation improvements
- 🎮 Sample datasets from different games
- 🧠 Code improvements and new features

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

## 📊 Project Structure

```
ai-game-automation/
├── config.py                    # Centralized configuration (EDIT THIS TO CUSTOMIZE)
├── scripts/                     # Main Python scripts (numbered workflow)
│   ├── 0_setup.py              # Install dependencies
│   ├── 2_verify_system_setup.py # Verify system compatibility
│   ├── 3_record_data.py        # Record gameplay data
│   ├── 4_analyze_data_quality.py # Analyze data quality
│   ├── 5_train_model.py        # Train AI model (with oversampling)
│   ├── 6_run_inference.py      # Run AI inference
│   └── 7_debug_model_output.py # Debug model predictions
├── docs/                       # Detailed documentation and guides
├── examples/                   # Sample data and pre-trained models
├── data_human/                 # Your recorded gameplay data
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── REFACTOR_SUMMARY.md         # Refactoring documentation
└── PROJECT_CLEANUP.md          # Cleanup summary
```

## ⚖️ License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Credits

Built with PyTorch, OpenCV, and inspired by behavior cloning research. Thanks to the open-source community!

---

**Ready to get started?** Run `python scripts/0_setup.py` and follow the numbered workflow above! 

## ⚙️ Quick Configuration

**Edit `config.py` to customize your setup:**

### **Key Training Parameters**
```python
# Dataset balancing (fixes class imbalance)
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15  # Frames with actions repeated 15x
VALIDATION_SPLIT = 0.15                   # 15% of data for validation

# Training settings
BATCH_SIZE = 16                           # Memory vs speed tradeoff
EPOCHS = 10                               # Reduced due to oversampling
LEARNING_RATE = 3e-4                      # Optimized for new architecture

# Inference thresholds
KEY_THRESHOLD = 0.2                       # Key press sensitivity
CLICK_THRESHOLD = 0.3                     # Mouse click sensitivity
```

### **Game-Specific Keys**
```python
# Modify COMMON_KEYS for your game
COMMON_KEYS = [
    'w', 'a', 's', 'd',                   # Movement
    'space', 'shift', 'ctrl',             # Actions
    '1', '2', '3', '4', '5',              # Hotkeys
    # Add more keys your game uses...
]
```

**🚀 GPU Acceleration**: The setup will automatically detect and configure GPU acceleration for faster training!

**Need help?** Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions to common issues.