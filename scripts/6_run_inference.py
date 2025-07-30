# 5_run_improved_inference.py - FIXED VERSION
import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import time
from collections import deque
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button
from pynput import keyboard
from torchvision import transforms

# === CONFIG ===
with mss.mss() as sct: 
    monitor = sct.monitors[1]
SCREEN_WIDTH, SCREEN_HEIGHT = monitor["width"], monitor["height"]
MODEL_PATH = "model_improved.pt"  # Use improved model
IMG_WIDTH, IMG_HEIGHT = 960, 540
KEY_THRESHOLD = 0.15  # Much lower threshold for key presses
CLICK_THRESHOLD = 0.3  # Lower click threshold too
SEQUENCE_LENGTH = 5
SMOOTH_FACTOR = 0.7  # Less aggressive mouse smoothing
INFERENCE_FPS = 10

COMMON_KEYS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '0', 'space', 'shift', 'ctrl', 'alt', 'tab', 'enter', 'backspace',
    'up', 'down', 'left', 'right', 'f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
    'f10', 'f11', 'f12', '-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/'
]

KEY_MAPPING = {
    'space': Key.space, 'shift': Key.shift, 'ctrl': Key.ctrl, 'alt': Key.alt, 'tab': Key.tab,
    'enter': Key.enter, 'backspace': Key.backspace, 'up': Key.up, 'down': Key.down,
    'left': Key.left, 'right': Key.right, 'f1': Key.f1, 'f3': Key.f3, 'f4': Key.f4,
    'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8, 'f9': Key.f9,
    'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12
}

for k in COMMON_KEYS:
    if len(k) == 1: 
        KEY_MAPPING[k] = k

class ImprovedBehaviorCloningCNNRNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        # Match improved training architecture exactly
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
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size, 
            hidden_size=256,
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
        # Separate heads for different outputs
        self.key_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(COMMON_KEYS)),
            nn.Sigmoid()
        )
        
        self.mouse_pos_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )
        
        self.mouse_click_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
    
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

# === Load Model and Controllers ===
output_dim = len(COMMON_KEYS) + 4
model = ImprovedBehaviorCloningCNNRNN(output_dim)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("✅ Improved model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Make sure to train with 3_train_improved_model.py first")
    exit(1)

model.eval()

keyboard_controller = KeyboardController()
mouse_controller = MouseController()
running = True

# State tracking
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_pressed_keys = set()
current_mouse_buttons = set()
target_mouse_pos = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

# Temporal smoothing variables for mouse movement
prev_mouse_pos = None
MOUSE_SMOOTHING_ALPHA = 0.2  # Lower = more smoothing (0.1-0.3 range recommended)
                              # 0.1 = very smooth, 0.3 = more responsive

# Improved transforms to match training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def on_key_press(key):
    global running
    if key == keyboard.Key.f2: 
        running = False

def apply_output(output):
    global target_mouse_pos, prev_mouse_pos
    
    raw_output = output.detach().cpu().numpy()
    
    # Extract predictions
    key_preds = raw_output[:len(COMMON_KEYS)]
    mouse_x = raw_output[len(COMMON_KEYS)] * SCREEN_WIDTH
    mouse_y = raw_output[len(COMMON_KEYS) + 1] * SCREEN_HEIGHT
    left_click = raw_output[len(COMMON_KEYS) + 2] > CLICK_THRESHOLD
    right_click = raw_output[len(COMMON_KEYS) + 3] > CLICK_THRESHOLD
    
    # Apply temporal smoothing (Exponential Moving Average)
    current_mouse_pred = (mouse_x, mouse_y)
    
    if prev_mouse_pos is None:
        smoothed_mouse_pos = current_mouse_pred
    else:
        # EMA: smoothed = alpha * current + (1 - alpha) * previous
        smoothed_mouse_pos = (
            MOUSE_SMOOTHING_ALPHA * current_mouse_pred[0] + (1 - MOUSE_SMOOTHING_ALPHA) * prev_mouse_pos[0],
            MOUSE_SMOOTHING_ALPHA * current_mouse_pred[1] + (1 - MOUSE_SMOOTHING_ALPHA) * prev_mouse_pos[1]
        )
    
    prev_mouse_pos = smoothed_mouse_pos
    target_mouse_pos = smoothed_mouse_pos
    
    # Handle key presses with much lower threshold
    active_keys = []
    for i, key_str in enumerate(COMMON_KEYS):
        pynput_key = KEY_MAPPING.get(key_str)
        if not pynput_key: 
            continue
            
        confidence = key_preds[i]
        is_pressed = confidence > KEY_THRESHOLD
        
        if is_pressed and key_str not in current_pressed_keys:
            print(f"🔑 AI PRESS: Key '{key_str}' (Confidence: {confidence:.3f})")
            try:
                keyboard_controller.press(pynput_key)
                current_pressed_keys.add(key_str)
                active_keys.append(key_str)
            except Exception as e:
                print(f"Error pressing key {key_str}: {e}")
                
        elif not is_pressed and key_str in current_pressed_keys:
            try:
                keyboard_controller.release(pynput_key)
                current_pressed_keys.remove(key_str)
            except Exception as e:
                print(f"Error releasing key {key_str}: {e}")
    
    # Handle mouse clicks
    if left_click and Button.left not in current_mouse_buttons:
        print(f"🖱️ AI LEFT CLICK (Confidence: {raw_output[len(COMMON_KEYS) + 2]:.3f})")
        try:
            mouse_controller.press(Button.left)
            current_mouse_buttons.add(Button.left)
        except Exception as e:
            print(f"Error with left click: {e}")
            
    elif not left_click and Button.left in current_mouse_buttons:
        try:
            mouse_controller.release(Button.left)
            current_mouse_buttons.remove(Button.left)
        except Exception as e:
            print(f"Error releasing left click: {e}")
    
    if right_click and Button.right not in current_mouse_buttons:
        print(f"🖱️ AI RIGHT CLICK (Confidence: {raw_output[len(COMMON_KEYS) + 3]:.3f})")
        try:
            mouse_controller.press(Button.right)
            current_mouse_buttons.add(Button.right)
        except Exception as e:
            print(f"Error with right click: {e}")
            
    elif not right_click and Button.right in current_mouse_buttons:
        try:
            mouse_controller.release(Button.right)
            current_mouse_buttons.remove(Button.right)
        except Exception as e:
            print(f"Error releasing right click: {e}")

def smooth_mouse_movement():
    """Apply smooth mouse movement towards target position"""
    try:
        current_pos = mouse_controller.position
        
        # Calculate smooth movement
        diff_x = target_mouse_pos[0] - current_pos[0]
        diff_y = target_mouse_pos[1] - current_pos[1]
        
        # Only move if difference is significant
        if abs(diff_x) > 3 or abs(diff_y) > 3:
            new_x = int(current_pos[0] + diff_x * SMOOTH_FACTOR)
            new_y = int(current_pos[1] + diff_y * SMOOTH_FACTOR)
            
            # Clamp to screen bounds
            new_x = max(0, min(SCREEN_WIDTH - 1, new_x))
            new_y = max(0, min(SCREEN_HEIGHT - 1, new_y))
            
            mouse_controller.position = (new_x, new_y)
            
    except Exception as e:
        print(f"Error moving mouse: {e}")

def capture_and_process_frame():
    """Capture and preprocess a frame"""
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2RGB)
            return transform(img)
    except Exception as e:
        print(f"Error capturing frame: {e}")
        return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)

# === Main Loop ===
listener = keyboard.Listener(on_press=on_key_press)
listener.start()

print("🟢 Starting IMPROVED AI inference in 3 seconds... Press [F2] to quit.")
print(f"🔧 Using thresholds: Key={KEY_THRESHOLD}, Click={CLICK_THRESHOLD}")
time.sleep(3)

# Initialize frame sequence
print("📸 Initializing frame sequence...")
for i in range(SEQUENCE_LENGTH):
    frame = capture_and_process_frame()
    frame_sequence.append(frame)
    print(f"  Frame {i+1}/{SEQUENCE_LENGTH} captured")
    time.sleep(0.1)

print("🤖 IMPROVED AI is now active!")

try:
    frame_interval = 1.0 / INFERENCE_FPS
    last_inference_time = time.time()
    frame_count = 0
    
    while running:
        current_time = time.time()
        
        # Run inference at consistent intervals
        if current_time - last_inference_time >= frame_interval:
            # Capture new frame
            frame = capture_and_process_frame()
            frame_sequence.append(frame)
            
            # Prepare input tensor
            input_tensor = torch.stack(list(frame_sequence)).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output_sequence = model(input_tensor)
                # Use the last timestep's output
                last_output = output_sequence[:, -1, :].squeeze()
            
            # Apply the predicted actions
            apply_output(last_output)
            
            frame_count += 1
            if frame_count % 50 == 0:  # Print status every 5 seconds
                print(f"🔄 Processed {frame_count} frames...")
            
            last_inference_time = current_time
        
        # Always apply smooth mouse movement
        smooth_mouse_movement()
        
        # Small sleep to prevent excessive CPU usage
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")
except Exception as e:
    print(f"❌ Error during inference: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup: Release all pressed keys and buttons
    print("🧹 Cleaning up...")
    
    for key_str in list(current_pressed_keys):
        pynput_key = KEY_MAPPING.get(key_str)
        if pynput_key:
            try:
                keyboard_controller.release(pynput_key)
            except:
                pass
    
    for button in list(current_mouse_buttons):
        try:
            mouse_controller.release(button)
        except:
            pass
    
    listener.stop()
    print("✅ Improved inference stopped cleanly.")