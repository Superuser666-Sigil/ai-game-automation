# Troubleshooting Guide - AI Game Automation v2.0

This guide helps you resolve common issues with the AI Game Automation system.

## 🚨 Common Issues

### Training Issues

#### **Training Won't Stop**
**Problem:** Training continues indefinitely or you can't stop it safely.

**Solutions:**
1. **Use graceful shutdown methods:**
   ```bash
   # Method 1: Press Ctrl+C (safest)
   # Training will complete current epoch, save checkpoint, then exit
   
   # Method 2: Create stop file from another terminal
   python stop_training.py
   
   # Method 3: Manual stop file
   touch STOP_TRAINING
   ```

2. **Check early stopping configuration:**
   ```python
   # In config.py
   EARLY_STOPPING_PATIENCE = 3  # Stop after 3 epochs without improvement
   EARLY_STOPPING_MIN_DELTA = 0.0  # Minimum improvement threshold
   ```

#### **CUDA Out of Memory**
**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. **Reduce batch size:**
   ```python
   # In config.py
   BATCH_SIZE = 32  # Reduce from 64
   ```

2. **Reduce image size:**
   ```python
   # In config.py
   TRAIN_IMG_WIDTH = 160   # Reduce from 224
   TRAIN_IMG_HEIGHT = 160  # Reduce from 224
   ```

3. **Use CPU training:**
   ```python
   # Force CPU usage
   device = torch.device("cpu")
   ```

#### **Model Not Learning**
**Problem:** Loss doesn't decrease or F1-score stays low.

**Solutions:**
1. **Check data quality:**
   ```bash
   python 4_debug_model_output.py
   ```

2. **Increase training data:**
   - Record more gameplay sessions
   - Ensure diverse actions are captured

3. **Adjust learning rate:**
   ```python
   # In config.py
   LEARNING_RATE = 5e-4  # Increase from 1e-4
   ```

4. **Check class imbalance:**
   ```python
   # In config.py
   OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 20  # Increase from 15
   ```

#### **Poor F1-Score**
**Problem:** Validation F1-score is low (<0.5).

**Solutions:**
1. **Increase oversampling:**
   ```python
   # In config.py
   OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 25
   ```

2. **Adjust thresholds:**
   ```python
   # In config.py
   KEY_THRESHOLD = 0.15  # Lower from 0.2
   ```

3. **Train longer:**
   ```python
   # In config.py
   EPOCHS = 50  # Increase from 30
   ```

### Inference Issues

#### **Input Lag**
**Problem:** AI responses are delayed or sluggish.

**Solutions:**
1. **Reduce inference FPS:**
   ```python
   # In config.py
   INFERENCE_FPS = 5  # Reduce from 10
   ```

2. **Use GPU for inference:**
   ```python
   # Ensure CUDA is available
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

3. **Reduce image processing:**
   ```python
   # In config.py
   IMG_WIDTH = 640   # Reduce from 960
   IMG_HEIGHT = 360  # Reduce from 540
   ```

#### **Model Not Responding**
**Problem:** AI doesn't press keys or move mouse.

**Solutions:**
1. **Check thresholds:**
   ```python
   # In config.py
   KEY_THRESHOLD = 0.1  # Lower threshold
   CLICK_THRESHOLD = 0.2  # Lower threshold
   ```

2. **Debug model output:**
   ```bash
   python 4_debug_model_output.py
   ```

3. **Check model loading:**
   ```bash
   # Ensure model file exists
   ls -la trained_model.pth
   ```

### Installation Issues

#### **PyTorch Installation Fails**
**Problem:** `pip install torch` fails or wrong version installed.

**Solutions:**
1. **Use correct PyTorch version:**
   ```bash
   # For NVIDIA CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For AMD DirectML (Windows)
   pip install torch-directml
   
   # For CPU only
   pip install torch torchvision
   ```

2. **Check CUDA compatibility:**
   ```bash
   nvidia-smi  # Check CUDA version
   python -c "import torch; print(torch.version.cuda)"
   ```

#### **Missing Dependencies**
**Problem:** Import errors for required packages.

**Solutions:**
1. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Use installation scripts:**
   ```bash
   # Linux/Mac
   ./install.sh
   
   # Windows
   install.bat
   ```

3. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

### Data Issues

#### **No Training Data Found**
**Problem:** `No valid datasets found!`

**Solutions:**
1. **Record training data:**
   ```bash
   python 1_record_data.py
   ```

2. **Check data directory structure:**
   ```
   data_human/
   ├── frames/          # Screenshot images
   └── actions.npy      # Action data
   ```

3. **Verify file permissions:**
   ```bash
   ls -la data_human/
   ```

#### **Frame Files Missing**
**Problem:** Training can't find frame images.

**Solutions:**
1. **Check file extensions:**
   ```bash
   # Ensure frames are .jpg or .png
   ls data_human/frames/*.jpg
   ls data_human/frames/*.png
   ```

2. **Re-record data:**
   ```bash
   python 1_record_data.py
   ```

## 🔧 Advanced Troubleshooting

### Performance Optimization

#### **Slow Training**
**Solutions:**
1. **Use GPU acceleration:**
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Increase batch size (if memory allows):**
   ```python
   # In config.py
   BATCH_SIZE = 128  # Increase from 64
   ```

3. **Use multiple workers:**
   ```python
   # In DataLoader
   num_workers=8  # Increase from 4
   ```

#### **Memory Issues**
**Solutions:**
1. **Monitor memory usage:**
   ```bash
   # Linux
   htop
   
   # Windows
   Task Manager
   ```

2. **Reduce model size:**
   ```python
   # In config.py
   CNN_CHANNELS = [16, 32, 64]  # Reduce from [32, 64, 128]
   ```

### Debugging Tools

#### **TensorBoard Issues**
**Problem:** TensorBoard won't start or shows no data.

**Solutions:**
1. **Check port availability:**
   ```bash
   # Find free port
   python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
   ```

2. **Manual TensorBoard start:**
   ```bash
   tensorboard --logdir runs/behavior_cloning_experiment --port 6006
   ```

#### **Model Debugging**
**Problem:** Need to understand model predictions.

**Solutions:**
1. **Use debug script:**
   ```bash
   python 4_debug_model_output.py
   ```

2. **Analyze checkpoints:**
   ```bash
   # Compare different epochs
   ls -la game_model/
   ```

## 📞 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Review the configuration in `config.py`**
3. **Run the debug script: `python 4_debug_model_output.py`**
4. **Check TensorBoard logs for training progress**

### Useful Commands

```bash
# Check system info
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU memory
nvidia-smi

# Monitor training
tensorboard --logdir runs/

# Test model loading
python -c "import torch; model = torch.load('trained_model.pth'); print('Model loaded successfully')"
```

### Common Error Messages

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch size or image size |
| `No module named 'torch'` | Install PyTorch correctly |
| `No valid datasets found` | Record training data first |
| `Model file not found` | Train model or check path |
| `Permission denied` | Check file permissions |

---

**Still having issues?** Check the GitHub issues page or create a new issue with:
- Your operating system and Python version
- The exact error message
- Steps to reproduce the problem
- Your configuration settings 