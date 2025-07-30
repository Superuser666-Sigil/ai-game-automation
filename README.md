# �� AI Game Automation

Train an AI to play games by learning from your gameplay! This project uses behavior cloning to teach neural networks to mimic human gameplay patterns.

## ✨ Features

- **🎯 Easy Setup**: Numbered scripts guide you through the entire process
- **🚀 DirectML GPU Acceleration**: Native AMD/Intel GPU support on Windows (3-6x faster training)
- **🧠 Smart GPU Detection**: Automatically detects and configures NVIDIA/AMD/CPU with fallback
- **⚙️ Centralized Configuration**: Single `config.py` file controls all settings with validation
- **📊 Data Quality Analysis**: Ensures your training data is good before training
- **🤖 Intelligent Model Sizing**: Recommends optimal configurations for your hardware
- **🎮 Real-time Inference**: Smooth mouse movement with temporal smoothing
- **🛠️ Comprehensive Debugging**: Tools to understand and improve your AI's performance
- **⚡ Optimized Training**: Balanced loss functions, robust error handling, and adaptive learning rates

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/ai-game-automation.git
cd ai-game-automation/scripts

# Follow the numbered workflow
python 0_setup.py                           # Install dependencies
python 1_check_dependencies.py              # Check your system
python 2_verify_system_setup.py             # Verify everything works
python 3_record_data.py                     # Record 5-10 minutes of gameplay
python 4_analyze_data_quality.py            # Check your recording quality
python 4.5_choose_model_configuration.py    # Choose optimal model size
python 5_train_model.py                     # Train your AI (10-60 minutes)
python 6_run_inference.py                   # Watch your AI play!
```

## 📋 Requirements

- **Windows 10/11** (primary platform)
- **Python 3.12** (recommended for DirectML support)
- **4GB+ RAM** (8GB+ recommended for training)
- **Optional**: AMD/Intel GPU for DirectML acceleration (3-6x faster)
- **Optional**: NVIDIA GPU for CUDA acceleration

## 🎯 How It Works

1. **📹 Record**: Capture your screen and input actions while playing
2. **📊 Analyze**: Check data quality and get optimization recommendations  
3. **🧠 Train**: Neural network learns to map screen images to your actions (15-60 minutes with GPU)
4. **🎮 Play**: AI takes control with smooth mouse movement and responsive key presses

## 📖 Documentation

| Guide | Description |
|-------|-------------|
| **[🛠️ Troubleshooting](docs/TROUBLESHOOTING.md)** | Solutions for common issues |
| **[⚙️ Configuration](docs/CONFIGURATION.md)** | Advanced settings and game-specific configs |
| **[📁 Examples](examples/README.md)** | Sample datasets and pre-trained models |
| **[🤝 Contributing](docs/CONTRIBUTING.md)** | How to contribute to the project |

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
├── config.py          # Centralized configuration (EDIT THIS TO CUSTOMIZE)
├── scripts/           # Main Python scripts (0-7, numbered workflow)
├── docs/             # Detailed documentation and guides
├── examples/         # Sample data and pre-trained models
├── data_human/       # Your recorded gameplay data
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## ⚖️ License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Credits

Built with PyTorch, OpenCV, and inspired by behavior cloning research. Thanks to the open-source community!

---

**Ready to get started?** Run `python scripts/0_setup.py` and follow the numbered workflow above! 

## ⚙️ Quick Configuration

**Edit `config.py` to customize your setup:**
- **Keys to learn**: Modify `COMMON_KEYS` list for your game
- **Training speed**: Adjust `BATCH_SIZE` and `LEARNING_RATE`
- **Model size**: Change `TRAIN_IMG_WIDTH/HEIGHT` for performance vs accuracy
- **Data location**: Set `DATA_DIR` for your recordings

**🚀 GPU Acceleration**: For AMD/Intel GPUs, the setup will automatically configure DirectML for 3-6x faster training!

**Need help?** Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions to common issues.