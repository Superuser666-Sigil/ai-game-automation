# ⚙️ Configuration Guide

**New in v2.0**: All configuration is now centralized in `config.py` for easy customization!

## 🎯 Quick Configuration

**All settings are in one file: `config.py`**

### Essential Settings

```python
# === TRAINING PARAMETERS ===
BATCH_SIZE = 32          # Higher = faster training, more memory
LEARNING_RATE = 1e-4     # Lower = more stable training
EPOCHS = 50              # More epochs = better learning (usually)

# === IMAGE SETTINGS ===
# Recording dimensions (full resolution for accuracy)
IMG_WIDTH, IMG_HEIGHT = 960, 540
# Training dimensions (smaller for speed)
TRAIN_IMG_WIDTH, TRAIN_IMG_HEIGHT = 224, 224

# === KEYS TO LEARN ===
COMMON_KEYS = ['w', 'a', 's', 'd', ...]  # Customize for your game
```

## 📊 Model Configuration

### Automated Model Sizing

Get intelligent recommendations:
```bash
cd scripts
python 4.5_choose_model_configuration.py
```

This analyzes your:
- Training data complexity
- System hardware capabilities  
- Available memory and storage

### Model Size Comparison

| Configuration | File Size | Training RAM | Training Time | Accuracy |
|---------------|-----------|--------------|---------------|----------|
| Small & Fast  | 5.7 MB    | 70 MB        | 5-30 min      | Good     |
| Medium        | 44.4 MB   | 371 MB       | 15-90 min     | Better   |
| Large         | 518.6 MB  | 1767 MB      | 45-240 min    | Best     |

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

Edit `scripts/6_run_inference.py`:

```python
# Key press sensitivity
KEY_THRESHOLD = 0.15        # Lower = more key presses (0.05-0.3)
CLICK_THRESHOLD = 0.3       # Mouse click sensitivity (0.1-0.5)

# Mouse movement smoothing
MOUSE_SMOOTHING_ALPHA = 0.2 # Lower = smoother (0.1-0.3)
```

### Performance Optimization

```python
# Inference speed vs accuracy
SEQUENCE_LENGTH = 5         # Shorter = faster, less context
IMG_WIDTH = 640            # Lower resolution = faster
IMG_HEIGHT = 360           

# GPU memory management
torch.cuda.empty_cache()   # Clear GPU memory between batches
```

## 🧠 Training Configuration

### Learning Parameters

Edit `scripts/5_train_model.py`:

```python
# Learning rate schedule
LEARNING_RATE = 5e-4       # Base learning rate
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Loss function weights
def improved_loss(outputs, targets, num_keys, device, key_weights):
    # Adjust these weights based on importance:
    total_loss = key_loss * 2.0 + pos_loss * 1.0 + click_loss * 1.5
    return total_loss
```

### Data Augmentation

```python
# Image transforms for training robustness
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Add variation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

## 🎮 Game-Specific Configurations

### FPS Games (Counter-Strike, Valorant)
```python
# High precision mouse, fast reactions
IMG_WIDTH, IMG_HEIGHT = 1280, 720    # Higher resolution for aiming
KEY_THRESHOLD = 0.2                  # More sensitive for quick reactions
MOUSE_SMOOTHING_ALPHA = 0.3          # Less smoothing for precise aim
FPS = 15                             # Higher capture rate

# Focus on movement and aiming keys
COMMON_KEYS = ['w', 'a', 's', 'd', 'space', 'shift', 'ctrl', 'r', 'g', 'b']
```

### Strategy Games (Age of Empires, StarCraft)
```python
# Lower resolution OK, focus on clicks and hotkeys
IMG_WIDTH, IMG_HEIGHT = 960, 540     # Standard resolution
KEY_THRESHOLD = 0.15                 # Standard sensitivity
CLICK_THRESHOLD = 0.25               # More click sensitivity
FPS = 10                             # Standard capture rate

# Include number keys for hotkeys
COMMON_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'q', 'w', 'e', 'r']
```

### RPG Games (Skyrim, Witcher)
```python
# Balanced settings, include interaction keys
IMG_WIDTH, IMG_HEIGHT = 960, 540     
KEY_THRESHOLD = 0.15                 
SEQUENCE_LENGTH = 7                  # Longer context for complex actions
FPS = 10

# Include interaction and inventory keys
COMMON_KEYS = ['w', 'a', 's', 'd', 'space', 'e', 'tab', 'i', 'j', 'k', 'm']
```

### Puzzle Games (Portal, Tetris)
```python
# Simple settings, focus on precise timing
IMG_WIDTH, IMG_HEIGHT = 640, 360     # Lower resolution OK
KEY_THRESHOLD = 0.1                  # Very sensitive for precise moves
SEQUENCE_LENGTH = 3                  # Short context
FPS = 10

# Minimal key set
COMMON_KEYS = ['w', 'a', 's', 'd', 'space', 'up', 'down', 'left', 'right']
```

## 💾 Storage and Memory Management

### Disk Space Optimization

```python
# Reduce storage requirements
IMG_WIDTH, IMG_HEIGHT = 640, 360     # Smaller images
FPS = 8                              # Lower frame rate
SEQUENCE_LENGTH = 3                  # Shorter sequences

# Compress training data
import cv2
cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
```

### Memory Management

```python
# For systems with limited RAM
BATCH_SIZE = 4                       # Smaller batches
del variables_not_needed             # Explicit cleanup
torch.cuda.empty_cache()             # GPU memory management

# Enable gradient checkpointing for large models
model.gradient_checkpointing = True
```

## 🔧 Advanced Features

### Multi-GPU Training

```python
# For systems with multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    BATCH_SIZE = BATCH_SIZE * torch.cuda.device_count()
```

### Mixed Precision Training

```python
# Faster training with minimal accuracy loss
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(sequences)
    loss = criterion(outputs, targets)
```

### Custom Loss Functions

```python
# Add your own loss components
def custom_loss(outputs, targets):
    # Standard losses
    key_loss = focal_loss(key_outputs, key_targets)
    mouse_loss = mse_loss(mouse_outputs, mouse_targets)
    
    # Custom: penalize inconsistent mouse movement
    mouse_smoothness = torch.mean(torch.abs(torch.diff(mouse_outputs, dim=1)))
    
    # Custom: reward correct key combinations
    combo_bonus = reward_key_combinations(key_outputs, key_targets)
    
    return key_loss + mouse_loss + 0.1 * mouse_smoothness - 0.05 * combo_bonus
```

---

**Need help with configuration?** Check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on GitHub. 