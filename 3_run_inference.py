import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import time
import math
import os
from collections import deque
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button
from pynput import keyboard
from torchvision import transforms
from config import * # Import all settings from the config file

# --- SCREEN DIMENSIONS ---
with mss.mss() as sct:
    monitor = sct.monitors[1]
SCREEN_WIDTH, SCREEN_HEIGHT = monitor["width"], monitor["height"]

# === POSITIONAL ENCODING ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# === TRANSFORMER MODEL DEFINITION ===
class BehaviorCloningTransformer(nn.Module):
    def __init__(self, output_dim, d_model, nhead, nlayers, dropout):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_out_size = self.cnn(dummy_input).shape[1]
        
        self.input_proj = nn.Linear(cnn_out_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQUENCE_LENGTH)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # Enhanced Output heads with better architecture (matching training)
        self.key_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(COMMON_KEYS))
        )
        
        self.mouse_pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Sigmoid()
        )
        
        # Enhanced click head with more capacity for rare events
        self.mouse_click_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # Mouse wheel head for scroll up/down
        self.mouse_wheel_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x_reshaped = x.view(b * s, c, h, w)
        feat = self.cnn(x_reshaped)
        feat_reshaped = feat.view(b, s, -1)
        
        projected_feat = self.input_proj(feat_reshaped) * math.sqrt(self.d_model)
        pos_encoded_feat = self.pos_encoder(projected_feat)
        
        transformer_out = self.transformer_encoder(pos_encoded_feat)
        
        key_out = self.key_head(transformer_out)
        pos_out = self.mouse_pos_head(transformer_out)
        click_out = self.mouse_click_head(transformer_out)
        wheel_out = self.mouse_wheel_head(transformer_out)
        return torch.cat([key_out, pos_out, click_out, wheel_out], dim=2)

# --- SETUP ---
device = torch.device("cpu")
output_dim = len(COMMON_KEYS) + 4
model = BehaviorCloningTransformer(output_dim, D_MODEL, N_HEAD, N_LAYERS, DROPOUT)
action_threshold = 0.5

try:
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model '{MODEL_FILE}' loaded successfully (trained for {checkpoint.get('epoch', 'N/A')} epochs).")

    threshold_path = os.path.join(os.path.dirname(MODEL_FILE) or ".", "best_threshold.txt")
    try:
        with open(threshold_path, 'r') as f:
            action_threshold = float(f.read().strip())
        print(f"   Dynamically loaded action threshold: {action_threshold:.2f}")
    except FileNotFoundError:
        print(f"   Threshold file not found. Using defaults: KEY={KEY_THRESHOLD}, CLICK={CLICK_THRESHOLD}")
        action_threshold = None
    except Exception as e:
        print(f"   Error loading threshold file: {e}. Using defaults.")
        action_threshold = None

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Ensure you have a trained model file at the correct path.")
    exit(1)

model.eval()
keyboard_controller = KeyboardController()
mouse_controller = MouseController()
running = True
ai_enabled = False

# --- STATE MANAGEMENT ---
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_pressed_keys = set()
current_mouse_buttons = set()
current_mouse_wheel = {'up': False, 'down': False}
target_mouse_pos = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- CONTROL FUNCTIONS ---
def on_press(key):
    global running, ai_enabled
    if key == keyboard.Key.f12:
        running = False
        print("🛑 Quit key (F12) pressed.")
    elif key == keyboard.Key.f10:
        ai_enabled = not ai_enabled
        if not ai_enabled: release_all_inputs()
        status = "ENABLED" if ai_enabled else "DISABLED"
        print(f"🤖 AI control is now {status}")

# 🐛 FIXED: This function now correctly handles the model's output
def apply_output(output):
    global target_mouse_pos
    output = output.detach().cpu()
    
    # Separate the output tensor into its components
    num_keys = len(COMMON_KEYS)
    key_logits = output[:num_keys]
    mouse_pos = output[num_keys : num_keys+2]
    click_logits = output[num_keys+2:num_keys+4]
    wheel_logits = output[num_keys+4:num_keys+6]

    # Apply sigmoid ONLY to the logits for keys, clicks, and wheel
    key_probs = torch.sigmoid(key_logits).numpy()
    click_probs = torch.sigmoid(click_logits).numpy()
    wheel_probs = torch.sigmoid(wheel_logits).numpy()
    
    # The mouse position is already a probability, so we use it directly
    mouse_x, mouse_y = mouse_pos[0].item(), mouse_pos[1].item()
    target_mouse_pos = (mouse_x * SCREEN_WIDTH, mouse_y * SCREEN_HEIGHT)
    
    key_thresh_to_use = action_threshold if action_threshold is not None else KEY_THRESHOLD
    click_thresh_to_use = action_threshold if action_threshold is not None else CLICK_THRESHOLD
    
    for i, key_str in enumerate(COMMON_KEYS):
        if not (pynput_key := KEY_MAPPING.get(key_str)): continue
        
        is_pressed = key_probs[i] > key_thresh_to_use
        if is_pressed and key_str not in current_pressed_keys:
            keyboard_controller.press(pynput_key)
            current_pressed_keys.add(key_str)
        elif not is_pressed and key_str in current_pressed_keys:
            keyboard_controller.release(pynput_key)
            current_pressed_keys.remove(key_str)
            
    left_click = click_probs[0] > click_thresh_to_use
    right_click = click_probs[1] > click_thresh_to_use
    
    if left_click and Button.left not in current_mouse_buttons:
        mouse_controller.press(Button.left)
        current_mouse_buttons.add(Button.left)
    elif not left_click and Button.left in current_mouse_buttons:
        mouse_controller.release(Button.left)
        current_mouse_buttons.remove(Button.left)
        
    if right_click and Button.right not in current_mouse_buttons:
        mouse_controller.press(Button.right)
        current_mouse_buttons.add(Button.right)
    elif not right_click and Button.right in current_mouse_buttons:
        mouse_controller.release(Button.right)
        current_mouse_buttons.remove(Button.right)
        
    # Handle mouse wheel scrolling
    wheel_thresh_to_use = action_threshold if action_threshold is not None else WHEEL_THRESHOLD
    
    wheel_up = wheel_probs[0] > wheel_thresh_to_use
    wheel_down = wheel_probs[1] > wheel_thresh_to_use
    
    if wheel_up and not current_mouse_wheel['up']:
        mouse_controller.scroll(0, 1)  # Scroll up
        current_mouse_wheel['up'] = True
        current_mouse_wheel['down'] = False
    elif wheel_down and not current_mouse_wheel['down']:
        mouse_controller.scroll(0, -1)  # Scroll down
        current_mouse_wheel['up'] = False
        current_mouse_wheel['down'] = True
    elif not wheel_up and not wheel_down:
        current_mouse_wheel['up'] = False
        current_mouse_wheel['down'] = False

def smooth_mouse_movement():
    current_pos = mouse_controller.position
    diff_x, diff_y = target_mouse_pos[0] - current_pos[0], target_mouse_pos[1] - current_pos[1]
    
    if abs(diff_x) > MOUSE_DEADZONE or abs(diff_y) > MOUSE_DEADZONE:
        new_x = int(current_pos[0] + diff_x * SMOOTH_FACTOR)
        new_y = int(current_pos[1] + diff_y * SMOOTH_FACTOR)
        mouse_controller.position = (max(0, min(SCREEN_WIDTH - 1, new_x)), max(0, min(SCREEN_HEIGHT - 1, new_y)))

def capture_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2RGB)
        return transform(img)

def release_all_inputs():
    for key_str in list(current_pressed_keys):
        if (pynput_key := KEY_MAPPING.get(key_str)):
            keyboard_controller.release(pynput_key)
    current_pressed_keys.clear()
    
    for button in list(current_mouse_buttons):
        mouse_controller.release(button)
    current_mouse_buttons.clear()
    print("All inputs released.")

# --- MAIN LOOP ---
if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("\n" + "="*50)
    print("🟢 Starting AI inference...")
    print("   Press [F10] to ENABLE/DISABLE AI control.")
    print("   Press [F12] to quit.")
    print("="*50 + "\n")

    for _ in range(SEQUENCE_LENGTH):
        frame_sequence.append(capture_frame())
        time.sleep(0.05)
    
    try:
        frame_interval = 1.0 / INFERENCE_FPS
        last_inference_time = time.time()
        
        while running:
            if ai_enabled:
                current_time = time.time()
                if current_time - last_inference_time >= frame_interval:
                    last_inference_time = current_time
                    frame_sequence.append(capture_frame())
                    
                    with torch.no_grad():
                        input_tensor = torch.stack(list(frame_sequence)).unsqueeze(0).to(device)
                        output = model(input_tensor)
                        apply_output(output[:, -1, :].squeeze())
                
                smooth_mouse_movement()
            
            time.sleep(0.001)
            
    finally:
        release_all_inputs()
        listener.stop()
        print("\n✅ Inference stopped cleanly.")