# config.py - Shared configuration for all scripts
import os

# === DATA COLLECTION & DIRECTORIES ===
DATA_DIR = "data_human"
FRAME_DIR = os.path.join(DATA_DIR, "frames")
ACTIONS_FILE = os.path.join(DATA_DIR, "actions.npy")
MODEL_PATH = "model_improved.pt"

# === IMAGE & SEQUENCE SETTINGS ===
# Recording/Inference dimensions (full resolution for accuracy)
IMG_WIDTH, IMG_HEIGHT = 960, 540
# Training dimensions (resized for efficiency and memory)
TRAIN_IMG_WIDTH, TRAIN_IMG_HEIGHT = 224, 224
SEQUENCE_LENGTH = 5

# === RECORDING & INFERENCE ===
RECORDING_FPS = 10
INFERENCE_FPS = 10

# === KEYS TO MONITOR ===
# Add or remove keys you want the AI to learn.
# Examples: 'w', 'a', 's', 'd', 'space', 'shift', 'e', 'q', '1', '2', '3', '4'
COMMON_KEYS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '0', 'space', 'shift', 'ctrl', 'alt', 'tab', 'enter', 'backspace',
    'up', 'down', 'left', 'right', 'f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
    'f10', 'f11', 'f12', '-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/'
]

# === TRAINING PARAMETERS ===
BATCH_SIZE = 32  # Increased from 16 to 32 to improve GPU utilization and training efficiency, at the cost of higher memory usage
EPOCHS = 50
LEARNING_RATE = 1e-4  # Reduced from 5e-4 for better convergence and stability
WEIGHT_DECAY = 1e-5
SCHEDULER_FACTOR = 0.7
SCHEDULER_PATIENCE = 10
GRADIENT_CLIP_NORM = 1.0

# === LOSS FUNCTION WEIGHTS ===
KEY_LOSS_WEIGHT = 1.0
POS_LOSS_WEIGHT = 1.0
CLICK_LOSS_WEIGHT = 1.0
SMOOTHNESS_LOSS_WEIGHT = 0.05

# === INFERENCE PARAMETERS ===
KEY_THRESHOLD = 0.15
CLICK_THRESHOLD = 0.3
MOUSE_SMOOTHING_ALPHA = 0.2  # Lower = more smoothing (0.1-0.3 range)

# === DEVICE SETTINGS ===
# Priority order: DirectML -> CUDA -> ROCm -> CPU
PREFER_DIRECTML = True
PREFER_CUDA = True
PREFER_ROCm = True

# === MODEL ARCHITECTURE ===
CNN_CHANNELS = [32, 64, 128]  # Channel progression
TEMPORAL_HIDDEN_SIZE = 256
KEY_HEAD_SIZE = 128
MOUSE_POS_HEAD_SIZE = 64
MOUSE_CLICK_HEAD_SIZE = 32

def validate_config():
    """Validate configuration consistency and provide warnings"""
    warnings = []
    
    # Check image dimension consistency
    if IMG_WIDTH != 960 or IMG_HEIGHT != 540:
        warnings.append(f"Recording dimensions ({IMG_WIDTH}x{IMG_HEIGHT}) differ from standard (960x540)")
    
    if TRAIN_IMG_WIDTH != 224 or TRAIN_IMG_HEIGHT != 224:
        warnings.append(f"Training dimensions ({TRAIN_IMG_WIDTH}x{TRAIN_IMG_HEIGHT}) differ from standard (224x224)")
    
    # Check learning rate
    if LEARNING_RATE > 1e-3:
        warnings.append(f"Learning rate {LEARNING_RATE} may be too high for stable training")
    
    # Check batch size
    if BATCH_SIZE > 64:
        warnings.append(f"Batch size {BATCH_SIZE} may cause memory issues on some GPUs")
    
    # Check key list
    if len(COMMON_KEYS) < 10:
        warnings.append(f"Only {len(COMMON_KEYS)} keys configured - consider adding more for better gameplay coverage")
    
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    return len(warnings) == 0