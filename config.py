import os

# === PATHS ===
DATA_DIR       = "data_human"
DATA_DIRS      = [DATA_DIR]
FRAME_DIR      = os.path.join(DATA_DIR, "frames")
ACTIONS_FILE   = "actions.npy"

MODEL_FILE                 = "trained_model.pth"
MODEL_SAVE_DIR             = "game_model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH_TEMPLATE   = os.path.join(MODEL_SAVE_DIR, "model_epoch{}.pt")

# === RECORDING & INFERENCE ===
RECORDING_FPS          = 10
INFERENCE_FPS          = 10
IMG_WIDTH              = 960
IMG_HEIGHT             = 540

# === TRAINING PARAMETERS ===
BATCH_SIZE             = 64
EPOCHS                 = 30
LEARNING_RATE          = 1e-4

# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15
VALIDATION_SPLIT                   = 0.15

# Number of timesteps to aggregate in validation
VALIDATION_WINDOW                  = 3

# Thresholds to sweep when binarizing outputs
THRESHOLD_SWEEP                    = [0.3, 0.4, 0.5, 0.6, 0.7]

# === IMAGE SETTINGS (for model input) ===
TRAIN_IMG_WIDTH       = 224
TRAIN_IMG_HEIGHT      = 224

# === SEQUENCE & ACTION SETTINGS ===
SEQUENCE_LENGTH       = 5

COMMON_KEYS = [
    "w","a","s","d",           # Movement
    "space","shift","ctrl",    # Actions
    "1","2","3","4","5",       # Hotkeys
    "q","e","r","f",           # Utility
    "tab","enter","escape",    # UI
]

KEY_THRESHOLD          = 0.2
CLICK_THRESHOLD        = 0.3
MOUSE_SMOOTHING_ALPHA  = 0.2
SMOOTH_FACTOR          = 0.7

# === LOSS WEIGHTS ===
KEY_LOSS_WEIGHT         = 1.0
POS_LOSS_WEIGHT         = 1.0
CLICK_LOSS_WEIGHT       = 1.0
SMOOTHNESS_LOSS_WEIGHT  = 0.05

# === EARLY STOPPING & TENSORBOARD ===
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.0

# Directory for TensorBoard logs
TENSORBOARD_LOG_DIR     = "runs/behavior_cloning_experiment"
