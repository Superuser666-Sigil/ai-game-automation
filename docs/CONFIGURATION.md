# ⚙️ Configuration Guide

**New in v2.0**: All configuration is now centralized in `config.py` with advanced oversampling and validation features!

## 🎯 Quick Configuration

**All settings are in one file: `config.py`**

### Essential Settings

```python
# === TRAINING PARAMETERS ===
BATCH_SIZE = 16                    # Optimized for oversampling and memory efficiency
EPOCHS = 10                        # Reduced due to oversampling effectiveness
LEARNING_RATE = 3e-4               # Optimized for the improved architecture

# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15  # Frames with actions repeated 15x
VALIDATION_SPLIT = 0.15                   # 15% of data for validation

# === IMAGE SETTINGS ===
# Recording dimensions (full resolution for accuracy)
IMG_WIDTH, IMG_HEIGHT = 960, 540
# Training dimensions (smaller for speed)
TRAIN_IMG_WIDTH, TRAIN_IMG_HEIGHT = 224, 224

# === KEYS TO LEARN ===
COMMON_KEYS = ['w', 'a', 's', 'd', ...]  # Customize for your game
```

## 🧠 Advanced Training Features

### Dataset Oversampling

The new training system automatically handles class imbalance by oversampling frames with actions:

```python
# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15  # Key setting for better training
VALIDATION_SPLIT = 0.15                   # Prevents overfitting
```

**How it works:**
- Frames with key presses or mouse clicks are repeated 15x more than normal frames
- This ensures the model learns from rare but important action moments
- Validation split prevents the model from overfitting to the training data

### Training Parameters

```python
# === TRAINING PARAMETERS ===
BATCH_SIZE = 16          # Optimized for oversampling (was 32)
EPOCHS = 10              # Reduced due to oversampling (was 50)
LEARNING_RATE = 3e-4     # Optimized for new architecture (was 1e-4)
```

**Why these changes:**
- **Smaller batch size**: Better for oversampling and memory efficiency
- **Fewer epochs**: Oversampling reduces the need for many training iterations
- **Higher learning rate**: The new architecture can handle faster learning

## 📊 Model Configuration

### Automated Model Sizing

Get intelligent recommendations:
```bash
python scripts/4.5_choose_model_configuration.py
```

This analyzes your:
- Training data complexity
- System hardware capabilities  
- Available memory and storage

### Model Architecture

The refactored system uses a consistent LSTM-based architecture:

```python
# === MODEL ARCHITECTURE ===
CNN_CHANNELS = [32, 64, 128]  # Channel progression
TEMPORAL_HIDDEN_SIZE = 256    # LSTM hidden size
KEY_HEAD_SIZE = 128           # Key prediction head
MOUSE_POS_HEAD_SIZE = 64      # Mouse position head
MOUSE_CLICK_HEAD_SIZE = 32    # Mouse click head
```

### Model Size Comparison

| Configuration | File Size | Training RAM | Training Time | Accuracy |
|---------------|-----------|--------------|---------------|----------|
| Current (v2.0) | ~15 MB    | ~200 MB      | 10-60 min     | Excellent |
| Large         | ~50 MB    | ~500 MB      | 30-120 min    | Best     |

## 🎮 Game-Specific Configuration

### Key Mapping for Your Game

Edit `COMMON_KEYS` in `config.py`:

```python
# Example: MOBA/RTS Games
COMMON_KEYS = [
    'q', 'w', 'e', 'r',           # Abilities
    '1', '2', '3', '4', '5', '6',  # Items/Hotkeys
    'a', 's', 'd',                # Attack/Stop/Hold
    'tab', 'space', 'alt'         # UI toggles
]

# Example: FPS Games  
COMMON_KEYS = [
    'w', 'a', 's', 'd',           # Movement
    'space', 'shift', 'ctrl',     # Jump/Run/Crouch
    'r', 'f', 'g',                # Reload/Use/Grenade
    '1', '2', '3', '4', '5'       # Weapon switching
]

# Example: RPG Games
COMMON_KEYS = [
    'w', 'a', 's', 'd',           # Movement
    '1', '2', '3', '4', '5',      # Hotbar skills
    'i', 'm', 'c', 'b',          # Inventory/Map/Character
    'tab', 'enter', 'space'      # Target/Chat/Interact
]
```

### Recording Settings

All in `config.py`:

```python
# === RECORDING & INFERENCE ===
RECORDING_FPS = 10      # Frames per second (10-15 recommended)
IMG_WIDTH = 960         # Recording resolution width
IMG_HEIGHT = 540        # Recording resolution height

# Performance vs Quality tradeoffs:
# Higher resolution = better accuracy, more memory
# Higher FPS = smoother gameplay, larger files
```

## 🎯 Inference Configuration

### Sensitivity Tuning

```python
# === INFERENCE PARAMETERS ===
KEY_THRESHOLD = 0.2               # Key press sensitivity (0.1-0.3 range)
CLICK_THRESHOLD = 0.3             # Mouse click sensitivity (0.2-0.4 range)
MOUSE_SMOOTHING_ALPHA = 0.2       # Mouse smoothing (0.1-0.3 range)
SMOOTH_FACTOR = 0.7               # Movement smoothing (0.5-0.8 range)
```

**Tuning Guidelines:**
- **Lower thresholds** = More sensitive, more false positives
- **Higher thresholds** = Less sensitive, fewer false positives
- **Lower smoothing** = Smoother but less responsive movement
- **Higher smoothing** = More responsive but potentially jittery

### Threshold Recommendations by Game Type

```python
# Fast-paced games (FPS, Action)
KEY_THRESHOLD = 0.15              # More sensitive
CLICK_THRESHOLD = 0.25            # Faster clicks
MOUSE_SMOOTHING_ALPHA = 0.3       # More responsive

# Slower games (Strategy, RPG)
KEY_THRESHOLD = 0.25              # Less sensitive
CLICK_THRESHOLD = 0.35            # More deliberate clicks
MOUSE_SMOOTHING_ALPHA = 0.15      # Smoother movement
```

## 🔧 Advanced Configuration

### Loss Function Weights

```python
# === LOSS FUNCTION WEIGHTS ===
KEY_LOSS_WEIGHT = 1.0             # Key press prediction weight
POS_LOSS_WEIGHT = 1.0             # Mouse position weight
CLICK_LOSS_WEIGHT = 1.0           # Mouse click weight
SMOOTHNESS_LOSS_WEIGHT = 0.05     # Movement smoothness weight
```

### Device Settings

```python
# === DEVICE SETTINGS ===
PREFER_DIRECTML = True            # AMD/Intel GPU acceleration
PREFER_CUDA = True                # NVIDIA GPU acceleration
PREFER_ROCm = True                # AMD Linux GPU acceleration
```

## 📈 Performance Optimization

### Memory Management

```python
# Reduce memory usage
BATCH_SIZE = 8                    # Smaller batches
SEQUENCE_LENGTH = 3               # Shorter sequences
TRAIN_IMG_WIDTH = 160             # Smaller training images
TRAIN_IMG_HEIGHT = 160
```

### Training Speed

```python
# Faster training
BATCH_SIZE = 32                   # Larger batches (if memory allows)
EPOCHS = 5                        # Fewer epochs
LEARNING_RATE = 5e-4              # Higher learning rate
```

### Accuracy vs Speed Tradeoffs

| Setting | Faster Training | Better Accuracy |
|---------|----------------|-----------------|
| `BATCH_SIZE` | Higher (32) | Lower (8-16) |
| `TRAIN_IMG_WIDTH/HEIGHT` | Lower (160) | Higher (224) |
| `SEQUENCE_LENGTH` | Lower (3) | Higher (5) |
| `EPOCHS` | Lower (5) | Higher (10-15) |

## 🛠️ Configuration Validation

The system automatically validates your configuration:

```bash
python scripts/2_verify_system_setup.py
```

This checks:
- ✅ Configuration consistency
- ✅ Model architecture compatibility
- ✅ Memory requirements
- ✅ GPU availability
- ✅ Data accessibility

## 🎯 Troubleshooting Configuration

### Common Issues

**Training too slow:**
```python
BATCH_SIZE = 32                   # Increase batch size
LEARNING_RATE = 5e-4              # Increase learning rate
EPOCHS = 5                        # Reduce epochs
```

**Poor key detection:**
```python
KEY_THRESHOLD = 0.15              # Lower threshold
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 20  # Increase oversampling
```

**Memory errors:**
```python
BATCH_SIZE = 8                    # Reduce batch size
TRAIN_IMG_WIDTH = 160             # Reduce image size
TRAIN_IMG_HEIGHT = 160
```

**Jittery mouse movement:**
```python
MOUSE_SMOOTHING_ALPHA = 0.1       # Lower smoothing
SMOOTH_FACTOR = 0.8               # Higher smoothing
```

## 📋 Configuration Checklist

Before training, verify these settings:

- [ ] `COMMON_KEYS` includes all keys your game uses
- [ ] `OVERSAMPLE_ACTION_FRAMES_MULTIPLIER` is set to 15-20
- [ ] `VALIDATION_SPLIT` is set to 0.15
- [ ] `BATCH_SIZE` fits your available memory
- [ ] `KEY_THRESHOLD` and `CLICK_THRESHOLD` are appropriate for your game
- [ ] GPU acceleration is properly configured

Run `python scripts/2_verify_system_setup.py` to validate your configuration! 