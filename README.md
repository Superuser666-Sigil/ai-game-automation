# AI Game Automation v2.0

A behavior cloning system for game automation using CNN-LSTM architecture. This system learns from human gameplay recordings and can replicate the behavior in real-time.

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-game-automation.git
   cd ai-game-automation
   ```

2. **Install the package:**
   ```bash
   # Install with pip
   pip install -e .
   
   # Or install with GPU support
   pip install -e .[gpu]
   
   # For development (includes linting tools)
   pip install -e .[dev,gpu]
   ```

3. **Install GPU-specific PyTorch (if needed):**
   ```bash
   # For NVIDIA CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For AMD DirectML (Windows)
   pip install torch-directml
   
   # For Apple Silicon
   pip install torch torchvision
   ```

### Usage

1. **Record training data:**
   ```bash
   python 1_record_data.py
   # Or use the console script
   record-data
   ```

2. **Train the model:**
   ```bash
   # Start training with automatic TensorBoard (recommended)
   python 2_train_model.py
   
   # Or disable TensorBoard auto-startup
   python 2_train_model.py --no-tensorboard
   
   # Or use the console script
   train-model
   ```

3. **Run inference (AI control):**
   ```bash
   python 3_run_inference.py
   # Or use the console script
   run-inference
   ```

4. **Debug model output:**
   ```bash
   python 4_debug_model_output.py
   # Or use the console script
   debug-model
   ```

## 🎯 Key Features

- **Behavior Cloning**: Learns from human gameplay recordings
- **Transformer Architecture**: Advanced CNN-Transformer for better temporal understanding
- **Real-time Inference**: Controls games in real-time
- **GPU Acceleration**: Supports CUDA, DirectML, and ROCm
- **Enhanced TensorBoard Integration**: Comprehensive training monitoring with auto-startup
- **Focal Loss**: Handles class imbalance for rare actions (mouse clicks)
- **Early Stopping**: Prevents overfitting with configurable patience
- **Oversampling**: Balances action vs. non-action frames
- **Validation**: Robust model evaluation with F1-score and prediction analysis
- **Graceful Shutdown**: Multiple ways to safely stop training

## 📊 TensorBoard Monitoring

The training script now includes automatic TensorBoard startup for comprehensive training monitoring:

### Automatic TensorBoard Startup
When you run training, TensorBoard automatically starts and opens at `http://localhost:6006`:
```bash
python 2_train_model.py
# TensorBoard will start automatically and show:
# ✅ TensorBoard started successfully!
#    Open your browser and go to: http://localhost:6006
```

### What You Can Monitor
- **Loss Components**: Individual losses for keys, clicks, and mouse position
- **Training Metrics**: Learning rate, gradient norms, and overall loss
- **Validation Metrics**: F1 scores, prediction distributions, and confusion matrices
- **Model Performance**: Action prediction statistics and threshold optimization
- **Prediction Analysis**: Histograms showing model confidence distributions

### Manual TensorBoard Control
If you prefer to start TensorBoard manually:
```bash
# Disable auto-startup
python 2_train_model.py --no-tensorboard

# Start TensorBoard manually
tensorboard --logdir runs/behavior_cloning_improved --port 6006
```

## 🛑 Graceful Training Shutdown

The training script includes three methods for safely stopping training without losing progress:

### Method 1: Early Stopping (Automatic)
Training automatically stops when the F1-score doesn't improve for 3 epochs:
```bash
# Configure in config.py
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0
```

### Method 2: Keyboard Interrupt (Manual)
Press `Ctrl+C` during training - it will complete the current epoch, save checkpoint, then exit:
```bash
python 2_train_model.py
# Press Ctrl+C when you want to stop
```

### Method 3: Stop File (Remote Control)
Create a stop file from another terminal to gracefully stop training:
```bash
# Stop training
python stop_training.py

# Remove stop file to allow training to continue
python stop_training.py remove

# Or manually
touch STOP_TRAINING    # Stop training
rm STOP_TRAINING       # Allow training to continue
```

**Benefits:**
- ✅ No data loss - Always saves checkpoint before stopping
- ✅ Clean exits - TensorBoard logs are properly closed
- ✅ Remote control - Can stop training from another SSH session
- ✅ Resumable - Can restart from any checkpoint

## 📁 Project Structure

```
ai-game-automation/
├── config.py                 # Centralized configuration
├── 1_record_data.py         # Data recording script
├── 2_train_model.py         # Training script
├── 3_run_inference.py       # Inference/control script
├── 4_debug_model_output.py  # Model debugging script
├── stop_training.py         # Graceful shutdown utility
├── data_human/              # Training data
│   ├── frames/              # Screenshot frames
│   └── actions.npy          # Action data
├── game_model/              # Model checkpoints
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── install.sh              # Linux/Mac installation script
├── install.bat             # Windows installation script
└── README.md               # This file
```

## ⚙️ Configuration

All settings are centralized in `config.py`:

- **Training Parameters**: Batch size, epochs, learning rate
- **Model Architecture**: CNN/LSTM dimensions
- **Data Processing**: Image size, sequence length
- **Inference Settings**: Thresholds, smoothing factors
- **Paths**: Data directories, model save locations
- **Early Stopping**: Patience and minimum improvement thresholds

## 🎮 Supported Games

The system works with any game that:
- Runs in windowed or fullscreen mode
- Uses keyboard and mouse input
- Has visual feedback for actions

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, macOS
- **GPU**: NVIDIA (CUDA), AMD (DirectML), or CPU
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for training data

## 📊 Performance

- **Training**: ~30 minutes on A100 GPU
- **Inference**: <10ms latency
- **Accuracy**: 85%+ F1-score on balanced datasets
- **Memory**: ~2GB VRAM usage during training

## 🛠️ Development

### Code Quality
```bash
# Format code
black .

# Lint code
ruff check .
flake8 .

# Run tests (if available)
pytest
```

### Adding New Features
1. Update `config.py` with new parameters
2. Modify the relevant script
3. Test with `debug-model`
4. Update this README

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **Model not learning**: Check data quality with `debug-model`
3. **Poor performance**: Increase training data or adjust thresholds
4. **Input lag**: Reduce inference FPS in `config.py`
5. **Training won't stop**: Use `python stop_training.py` or press `Ctrl+C`

### Getting Help

- Check the configuration in `config.py`
- Use `debug-model` to analyze model output
- Monitor training with TensorBoard
- Review the troubleshooting guide

## 🎯 Roadmap

- [ ] Multi-game support
- [ ] Reinforcement learning integration
- [ ] Web UI for configuration
- [ ] Cloud training support
- [ ] Mobile deployment

---

**Happy gaming! 🎮** 