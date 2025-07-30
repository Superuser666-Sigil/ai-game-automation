# �� AI Game Automation

Train an AI to play games by learning from your gameplay! This project uses behavior cloning to teach neural networks to mimic human gameplay patterns.

## ✨ Features

- **🎯 Easy Setup**: Numbered scripts guide you through the entire process
- **🧠 Smart GPU Detection**: Automatically detects and configures NVIDIA/AMD/CPU
- **📊 Data Quality Analysis**: Ensures your training data is good before training
- **⚙️ Intelligent Model Sizing**: Recommends optimal configurations for your hardware
- **🎮 Real-time Inference**: Smooth mouse movement and responsive key presses
- **🛠️ Comprehensive Debugging**: Tools to understand and improve your AI's performance

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
- **Python 3.8+**
- **4GB+ RAM** (8GB+ recommended for training)
- **Optional**: NVIDIA/AMD GPU for faster training

## 🎯 How It Works

1. **📹 Record**: Capture your screen and input actions while playing
2. **📊 Analyze**: Check data quality and get optimization recommendations  
3. **🧠 Train**: Neural network learns to map screen images to your actions
4. **🎮 Play**: AI takes control and mimics your gameplay style

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
├── scripts/           # Main Python scripts (0-7, numbered workflow)
├── docs/             # Detailed documentation and guides
├── examples/         # Sample data and pre-trained models
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

## ⚖️ License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Credits

Built with PyTorch, OpenCV, and inspired by behavior cloning research. Thanks to the open-source community!

---

**Ready to get started?** Run `python scripts/0_setup.py` and follow the numbered workflow above! 

**Need help?** Check the [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions to common issues.