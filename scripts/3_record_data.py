# 3_record_data.py - Human Data Recording
import os
import sys
import cv2
import mss
import numpy as np
import time
from pynput import keyboard, mouse
import threading

# Import shared configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# === SETUP ===
os.makedirs(FRAME_DIR, exist_ok=True)

with mss.mss() as sct:
    monitor = sct.monitors[1]
    SCREEN_WIDTH = monitor["width"]
    SCREEN_HEIGHT = monitor["height"]

print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# Global state
pressed_keys = set()
mouse_buttons = {'left': 0, 'right': 0}
mouse_position = (0.5, 0.5)  # Start at center
running = True
data_lock = threading.Lock()

def get_key_str(key):
    if hasattr(key, 'char') and key.char:
        return key.char.lower()
    elif hasattr(key, 'name'):
        return key.name.replace('_l', '').replace('_r', '')
    return None

def on_key_press(key):
    global running
    if key == keyboard.Key.f2:
        print("🛑 Quit key (F2) pressed.")
        running = False
        return
    
    key_str = get_key_str(key)
    if key_str in COMMON_KEYS:
        with data_lock:
            pressed_keys.add(key_str)

def on_key_release(key):
    key_str = get_key_str(key)
    if key_str in COMMON_KEYS:
        with data_lock:
            pressed_keys.discard(key_str)

def on_click(x, y, button, pressed):
    with data_lock:
        if button == mouse.Button.left:
            mouse_buttons['left'] = int(pressed)
        elif button == mouse.Button.right:
            mouse_buttons['right'] = int(pressed)

def on_move(x, y):
    global mouse_position
    with data_lock:
        # Normalize mouse position to [0, 1]
        mouse_position = (x / SCREEN_WIDTH, y / SCREEN_HEIGHT)

def capture_frame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img

def get_current_action():
    with data_lock:
        key_vector = [int(k in pressed_keys) for k in COMMON_KEYS]
        action = key_vector + list(mouse_position) + [mouse_buttons['left'], mouse_buttons['right']]
        return action.copy(), pressed_keys.copy()

if __name__ == "__main__":
    print("🟢 Starting HUMAN data recording in 5 seconds...")
    print("Play the game normally. Press [F2] to quit.")
    time.sleep(5)
    
    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    key_listener.start()
    mouse_listener.start()
    
    actions = []
    frame_interval = 1.0 / RECORDING_FPS
    i = 0
    
    try:
        last_capture_time = time.time()
        
        while running:
            current_time = time.time()
            
            # Capture at consistent intervals
            if current_time - last_capture_time >= frame_interval:
                frame = capture_frame()
                action, current_keys = get_current_action()
                
                frame_path = os.path.join(FRAME_DIR, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                actions.append(action)
                
                # Log key presses for debugging
                if current_keys:
                    print(f"Frame {i}: keys={list(current_keys)}, mouse=({action[-4]:.2f}, {action[-3]:.2f})")
                
                i += 1
                last_capture_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            
    finally:
        key_listener.stop()
        mouse_listener.stop()
        
        if actions:
            actions_array = np.array(actions, dtype=np.float32)
            np.save(ACTIONS_FILE, actions_array)
            print(f"✅ Saved {len(actions)} actions and frames to {DATA_DIR}.")
            
            # Print statistics for debugging
            key_actions = actions_array[:, :len(COMMON_KEYS)]
            key_press_rates = np.sum(key_actions, axis=0) / len(actions)
            active_keys = [(COMMON_KEYS[i], rate) for i, rate in enumerate(key_press_rates) if rate > 0.01]
            print(f"Key press rates: {active_keys}")
        else:
            print("No actions recorded!")