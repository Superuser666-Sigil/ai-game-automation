import os
from pynput import keyboard

# === PATHS ===
DATA_DIR = "data_human"
DATA_DIRS = [DATA_DIR]  # You can add more data directories here
FRAME_DIR = os.path.join(DATA_DIR, "frames")
ACTIONS_FILE = "actions.npy"

MODEL_FILE = "trained_model.pth"  # Final trained model
MODEL_SAVE_DIR = "game_model_checkpoints"  # Directory for epoch checkpoints
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH_TEMPLATE = os.path.join(MODEL_SAVE_DIR, "model_epoch_{}.pth")

# === FPS-OPTIMIZED IMAGE & SEQUENCE SETTINGS ===
# Optimized for FPS gaming with better detail retention
# Options: 360x240 (fast), 480x320 (balanced), 640x480 (detailed)
IMG_WIDTH = 1920  # Increased from 360 for better FPS detail
IMG_HEIGHT = 1080  # Increased from 240 for better FPS detail
SEQUENCE_LENGTH = 15  # Number of frames the model sees at once

# === FPS-OPTIMIZED RECORDING & INFERENCE FPS ===
# Higher FPS for better responsiveness in FPS games
RECORDING_FPS = 60  # Increased from 30 for smoother FPS gameplay
INFERENCE_FPS = 60  # Increased from 30 for more responsive AI control

# === FPS-OPTIMIZED TRAINING PARAMETERS ===
# Adjusted for higher resolution and FPS
BATCH_SIZE = 16  # Reduced from 32 due to higher resolution
EPOCHS = 50  # Increased epochs for better convergence
LEARNING_RATE = 2e-4  # Slightly higher learning rate for better learning

# === TRANSFORMER MODEL PARAMETERS ===
# These are used for the Transformer-based model architecture.
D_MODEL = 256  # The dimension of the transformer model (embedding size)
N_HEAD = 8     # Number of attention heads in the multi-head attention models
N_LAYERS = 3   # Number of sub-encoder-layers in the transformer encoder
DROPOUT = 0.1  # Dropout value

# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 20  # Increased for better balance
VALIDATION_SPLIT = 0.15
VALIDATION_WINDOW = 3  # Timesteps to aggregate for validation metrics
THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # Extended range

# === FPS-OPTIMIZED INFERENCE THRESHOLDS ===
KEY_THRESHOLD = 0.5  # Stricter threshold after validation
CLICK_THRESHOLD = 0.5
WHEEL_THRESHOLD = 0.5  # Mouse wheel threshold
# Mouse smoothing parameters (optimized for FPS)
MOUSE_SMOOTHING_ALPHA = 0.4  # More responsive for FPS (was 0.3)
SMOOTH_FACTOR = 0.7  # Faster movement for FPS (was 0.6)
MOUSE_DEADZONE = 1  # Smaller deadzone for precision (was 2)

# === EARLY STOPPING & TENSORBOARD ===
EARLY_STOPPING_PATIENCE = 8  # Increased patience for better convergence
EARLY_STOPPING_MIN_DELTA = 0.003  # Reduced minimum improvement threshold
TENSORBOARD_LOG_DIR = "runs/behavior_cloning_improved"  # New experiment name

# === FPS-OPTIMIZED KEY MAPPING ===
# This list defines the order and size of the keyboard action space for the model.
# Optimized for FPS gaming with primary movement and action keys.
COMMON_KEYS = [
    "w","a","s","d",                # Primary movement (WASD)
    "space","shift","ctrl",         # Actions (jump, sprint, crouch)
    "1","2","3","4","5","6",        # Weapon hotkeys
    "e","r","tab",                  # Interact, reload, menu
]

# This dictionary maps the string representation to pynput key objects for inference.
KEY_MAPPING = {
    # Alphanumeric
    **{char: keyboard.KeyCode.from_char(char) for char in "abcdefghijklmnopqrstuvwxyz1234567890"},
    # Function keys
    **{f'f{i}': getattr(keyboard.Key, f'f{i}') for i in range(1, 13)},
    # Modifier keys
    'shift': keyboard.Key.shift,
    'ctrl': keyboard.Key.ctrl,
    'alt': keyboard.Key.alt,
    # Special keys
    'space': keyboard.Key.space,
    'enter': keyboard.Key.enter,
    'backspace': keyboard.Key.backspace,
    'tab': keyboard.Key.tab,
    'escape': keyboard.Key.esc,
    'insert': keyboard.Key.insert,
    'delete': keyboard.Key.delete,
    'home': keyboard.Key.home,
    'end': keyboard.Key.end,
    'page_up': keyboard.Key.page_up,
    'page_down': keyboard.Key.page_down,
    # Arrow keys
    'up': keyboard.Key.up,
    'down': keyboard.Key.down,
    'left': keyboard.Key.left,
    'right': keyboard.Key.right,
    # Symbol keys
    '`': keyboard.KeyCode.from_char('`'),
    '-': keyboard.KeyCode.from_char('-'),
    '=': keyboard.KeyCode.from_char('='),
    '[': keyboard.KeyCode.from_char('['),
    ']': keyboard.KeyCode.from_char(']'),
    '\\': keyboard.KeyCode.from_char('\\'),
    ';': keyboard.KeyCode.from_char(';'),
    "'": keyboard.KeyCode.from_char("'"),
    ',': keyboard.KeyCode.from_char(','),
    '.': keyboard.KeyCode.from_char('.'),
    '/': keyboard.KeyCode.from_char('/'),
}
