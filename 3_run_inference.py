# 3_run_inference.py - REFACTORED
import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import time
from collections import deque
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button
from pynput import keyboard
from torchvision import transforms
from config import (
    MODEL_FILE, COMMON_KEYS, KEY_THRESHOLD,
    CLICK_THRESHOLD, MOUSE_SMOOTHING_ALPHA, SMOOTH_FACTOR,
    IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH, INFERENCE_FPS
)  # Import specific settings from the config file

# === CONFIG (Loaded from config.py) ===
MODEL_PATH = MODEL_FILE  # Final trained model is in root directory

# --- Main script ---
with mss.mss() as sct: 
    monitor = sct.monitors[1]
SCREEN_WIDTH, SCREEN_HEIGHT = monitor["width"], monitor["height"]

class ImprovedBehaviorCloningCNNRNN(nn.Module):
    # ... (Class definition is the same as in the training script) ...
    def __init__(self, output_dim):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=256, num_layers=2, batch_first=True, dropout=0.1)
        self.key_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(COMMON_KEYS)), nn.Sigmoid())
        self.mouse_pos_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())
        self.mouse_click_head = nn.Sequential(nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 2), nn.Sigmoid())
    
    def forward(self, x):
        b, s, c, h, w = x.shape
        cnn_out = self.cnn(x.view(b * s, c, h, w))
        lstm_in = cnn_out.view(b, s, -1)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_flat = lstm_out.reshape(b * s, -1)
        key_out = self.key_head(lstm_flat)
        mouse_pos_out = self.mouse_pos_head(lstm_flat)
        mouse_click_out = self.mouse_click_head(lstm_flat)
        combined = torch.cat([key_out, mouse_pos_out, mouse_click_out], dim=1)
        return combined.view(b, s, -1)


output_dim = len(COMMON_KEYS) + 4
model = ImprovedBehaviorCloningCNNRNN(output_dim)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("✅ Improved model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}"); exit(1)

model.eval()
keyboard_controller = KeyboardController()
mouse_controller = MouseController()
running = True

# ... (Rest of the functions and main loop remain the same) ...
# They will now use variables like KEY_THRESHOLD, SMOOTH_FACTOR, etc., from the config.
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_pressed_keys = set()
current_mouse_buttons = set()
target_mouse_pos = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
prev_mouse_pos = None

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def on_key_press(key):
    global running
    if key == keyboard.Key.f2: running = False

def apply_output(output):
    global target_mouse_pos, prev_mouse_pos
    raw_output = output.detach().cpu().numpy()
    
    key_preds = raw_output[:len(COMMON_KEYS)]
    mouse_x = raw_output[len(COMMON_KEYS)] * SCREEN_WIDTH
    mouse_y = raw_output[len(COMMON_KEYS) + 1] * SCREEN_HEIGHT
    left_click = raw_output[len(COMMON_KEYS) + 2] > CLICK_THRESHOLD
    right_click = raw_output[len(COMMON_KEYS) + 3] > CLICK_THRESHOLD
    
    current_mouse_pred = (mouse_x, mouse_y)
    if prev_mouse_pos is None:
        smoothed_mouse_pos = current_mouse_pred
    else:
        smoothed_mouse_pos = (
            MOUSE_SMOOTHING_ALPHA * current_mouse_pred[0] + (1 - MOUSE_SMOOTHING_ALPHA) * prev_mouse_pos[0],
            MOUSE_SMOOTHING_ALPHA * current_mouse_pred[1] + (1 - MOUSE_SMOOTHING_ALPHA) * prev_mouse_pos[1]
        )
    prev_mouse_pos = smoothed_mouse_pos
    target_mouse_pos = smoothed_mouse_pos
    
    for i, key_str in enumerate(COMMON_KEYS):
        pynput_key = KEY_MAPPING.get(key_str)
        if not pynput_key: continue
        is_pressed = key_preds[i] > KEY_THRESHOLD
        if is_pressed and key_str not in current_pressed_keys:
            keyboard_controller.press(pynput_key); current_pressed_keys.add(key_str)
        elif not is_pressed and key_str in current_pressed_keys:
            keyboard_controller.release(pynput_key); current_pressed_keys.remove(key_str)
    
    if left_click and Button.left not in current_mouse_buttons: mouse_controller.press(Button.left); current_mouse_buttons.add(Button.left)
    elif not left_click and Button.left in current_mouse_buttons: mouse_controller.release(Button.left); current_mouse_buttons.remove(Button.left)
    if right_click and Button.right not in current_mouse_buttons: mouse_controller.press(Button.right); current_mouse_buttons.add(Button.right)
    elif not right_click and Button.right in current_mouse_buttons: mouse_controller.release(Button.right); current_mouse_buttons.remove(Button.right)

def smooth_mouse_movement():
    current_pos = mouse_controller.position
    diff_x, diff_y = target_mouse_pos[0] - current_pos[0], target_mouse_pos[1] - current_pos[1]
    if abs(diff_x) > 3 or abs(diff_y) > 3:
        new_x = int(current_pos[0] + diff_x * SMOOTH_FACTOR)
        new_y = int(current_pos[1] + diff_y * SMOOTH_FACTOR)
        mouse_controller.position = (max(0, min(SCREEN_WIDTH - 1, new_x)), max(0, min(SCREEN_HEIGHT - 1, new_y)))

def capture_and_process_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2RGB)
        return transform(img)

listener = keyboard.Listener(on_press=on_key_press); listener.start()
print("🟢 Starting AI inference in 3 seconds... Press [F2] to quit.")
time.sleep(3)

for _ in range(SEQUENCE_LENGTH): frame_sequence.append(capture_and_process_frame()); time.sleep(0.1)

try:
    frame_interval = 1.0 / INFERENCE_FPS
    last_inference_time = time.time()
    while running:
        current_time = time.time()
        if current_time - last_inference_time >= frame_interval:
            frame_sequence.append(capture_and_process_frame())
            with torch.no_grad():
                output = model(torch.stack(list(frame_sequence)).unsqueeze(0))
                apply_output(output[:, -1, :].squeeze())
            last_inference_time = current_time
        smooth_mouse_movement()
        time.sleep(0.005)
finally:
    for key_str in list(current_pressed_keys):
        if (pynput_key := KEY_MAPPING.get(key_str)): keyboard_controller.release(pynput_key)
    for button in list(current_mouse_buttons): mouse_controller.release(button)
    listener.stop()
    print("✅ Inference stopped cleanly.")