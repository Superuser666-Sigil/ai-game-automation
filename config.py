# Configuration for AI Game Automation
# All settings centralized here for easy customization

import os

# === PATHS ===
DATA_DIR = "data_human"
FRAME_DIR = os.path.join(DATA_DIR, "frames")
ACTIONS_FILE = os.path.join(DATA_DIR, "actions.npy")
MODEL_FILE = "trained_model.pth"

# === RECORDING & INFERENCE ===
RECORDING_FPS = 10
INFERENCE_FPS = 10
IMG_WIDTH = 960
IMG_HEIGHT = 540

# === TRAINING PARAMETERS ===
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 3e-4

# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15
VALIDATION_SPLIT = 0.15

# === IMAGE SETTINGS ===
TRAIN_IMG_WIDTH = 224
TRAIN_IMG_HEIGHT = 224

# === MODEL ARCHITECTURE ===
CNN_CHANNELS = [32, 64, 128]
TEMPORAL_HIDDEN_SIZE = 256
KEY_HEAD_SIZE = 128
MOUSE_POS_HEAD_SIZE = 64
MOUSE_CLICK_HEAD_SIZE = 32

# === SEQUENCE SETTINGS ===
SEQUENCE_LENGTH = 5

# === KEYS TO LEARN ===
COMMON_KEYS = [
    "w",
    "a",
    "s",
    "d",  # Movement
    "space",
    "shift",
    "ctrl",  # Actions
    "1",
    "2",
    "3",
    "4",
    "5",  # Hotkeys
    "q",
    "e",
    "r",
    "f",  # Additional actions
    "tab",
    "enter",
    "escape",  # UI keys
]

# === KEY MAPPING ===
KEY_MAPPING = {
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "space": "space",
    "shift": "shift",
    "ctrl": "ctrl",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "q": "q",
    "e": "e",
    "r": "r",
    "f": "f",
    "tab": "tab",
    "enter": "enter",
    "escape": "escape",
}

# === INFERENCE PARAMETERS ===
KEY_THRESHOLD = 0.2
CLICK_THRESHOLD = 0.3
MOUSE_SMOOTHING_ALPHA = 0.2
SMOOTH_FACTOR = 0.7

# === DEVICE SETTINGS ===
PREFER_DIRECTML = True
PREFER_CUDA = True
PREFER_ROCm = True

# === LOSS FUNCTION WEIGHTS ===
KEY_LOSS_WEIGHT = 1.0
POS_LOSS_WEIGHT = 1.0
CLICK_LOSS_WEIGHT = 1.0
SMOOTHNESS_LOSS_WEIGHT = 0.05


def validate_config():
    """Validate configuration settings."""
    # Check required directories exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(FRAME_DIR):
        os.makedirs(FRAME_DIR)

    # Validate numeric parameters
    assert 0 < VALIDATION_SPLIT < 1, "VALIDATION_SPLIT must be between 0 and 1"
    assert (
        OVERSAMPLE_ACTION_FRAMES_MULTIPLIER > 0
    ), "OVERSAMPLE_ACTION_FRAMES_MULTIPLIER must be positive"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert EPOCHS > 0, "EPOCHS must be positive"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"

    # Validate image dimensions
    assert (
        TRAIN_IMG_WIDTH > 0 and TRAIN_IMG_HEIGHT > 0
    ), "Training image dimensions must be positive"
    assert (
        IMG_WIDTH > 0 and IMG_HEIGHT > 0
    ), "Recording image dimensions must be positive"

    # Validate thresholds
    assert 0 <= KEY_THRESHOLD <= 1, "KEY_THRESHOLD must be between 0 and 1"
    assert 0 <= CLICK_THRESHOLD <= 1, "CLICK_THRESHOLD must be between 0 and 1"
    assert (
        0 <= MOUSE_SMOOTHING_ALPHA <= 1
    ), "MOUSE_SMOOTHING_ALPHA must be between 0 and 1"

    print("✅ Configuration validation passed!")


if __name__ == "__main__":
    validate_config()
