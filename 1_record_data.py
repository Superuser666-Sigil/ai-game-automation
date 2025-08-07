import cv2
import mss
import numpy as np
import os
import time
import re # <-- ADDED
import signal
import sys
from pynput import keyboard, mouse
import threading
from config import * # Import all settings from the config file

# Create directories if they don't exist
os.makedirs(FRAME_DIR, exist_ok=True)

# --- Main script ---
# Get screen dimensions for mouse coordinate normalization
with mss.mss() as sct:
    monitor = sct.monitors[1]
    SCREEN_WIDTH = monitor["width"]
    SCREEN_HEIGHT = monitor["height"]

print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"Recording at {IMG_WIDTH}x{IMG_HEIGHT} @ {RECORDING_FPS} FPS")
print(f"Keys being recorded: {len(COMMON_KEYS)}")


# --- Global State Management ---
pressed_keys = set()
mouse_buttons = {'left': 0, 'right': 0}
# Store mouse position as normalized coordinates (0.0 to 1.0)
mouse_position = (0.5, 0.5)
# Use a smoothed position for recording to reduce jitter from high-frequency mouse movements
smoothed_mouse_position = (0.5, 0.5)
# Mouse wheel state (up/down scrolling)
mouse_wheel = {'up': 0, 'down': 0}
running = True
data_lock = threading.Lock() # Thread-safe access to shared state

def get_key_str(key):
    """Converts a pynput key object to a standardized string."""
    if hasattr(key, 'char') and key.char:
        return key.char.lower()
    elif hasattr(key, 'name'):
        # Standardize names (e.g., 'shift_r' -> 'shift')
        return key.name.replace('_l', '').replace('_r', '')
    return None

def on_key_press(key):
    """Handles key press events."""
    global running
    # Use a dedicated quit key (F12) that is not in COMMON_KEYS
    if key == keyboard.Key.f12:
        print("🛑 Quit key (F12) pressed. Stopping recording...")
        running = False
        return
    # Also allow Ctrl+C equivalent with Escape key
    elif key == keyboard.Key.esc:
        print("🛑 Escape key pressed. Stopping recording...")
        running = False
        return
    
    key_str = get_key_str(key)
    if key_str in COMMON_KEYS:
        with data_lock:
            pressed_keys.add(key_str)

def on_key_release(key):
    """Handles key release events."""
    key_str = get_key_str(key)
    if key_str in COMMON_KEYS:
        with data_lock:
            pressed_keys.discard(key_str)

def on_click(x, y, button, pressed):
    """Handles mouse click events."""
    with data_lock:
        if button == mouse.Button.left:
            mouse_buttons['left'] = int(pressed)
        elif button == mouse.Button.right:
            mouse_buttons['right'] = int(pressed)

def on_scroll(x, y, dx, dy):
    """Handles mouse wheel scroll events."""
    with data_lock:
        if dy > 0:  # Scroll up
            mouse_wheel['up'] = 1
            mouse_wheel['down'] = 0
        elif dy < 0:  # Scroll down
            mouse_wheel['up'] = 0
            mouse_wheel['down'] = 1
        else:
            mouse_wheel['up'] = 0
            mouse_wheel['down'] = 0

def on_move(x, y):
    """Handles mouse movement events and applies smoothing."""
    global mouse_position, smoothed_mouse_position
    with data_lock:
        # Normalize coordinates
        raw_position = (x / SCREEN_WIDTH, y / SCREEN_HEIGHT)
        mouse_position = raw_position
        
        # Apply exponential moving average for smoothing
        alpha = 0.3
        smoothed_mouse_position = (
            alpha * raw_position[0] + (1 - alpha) * smoothed_mouse_position[0],
            alpha * raw_position[1] + (1 - alpha) * smoothed_mouse_position[1]
        )

def capture_frame():
    """Captures and resizes a single screen frame."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        # Convert from BGRA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # Resize to the standard dimensions for the model
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img

def get_current_action():
    """Constructs the action vector for the current state."""
    with data_lock:
        # Create a binary vector for key presses
        key_vector = [int(k in pressed_keys) for k in COMMON_KEYS]
        # Use the smoothed mouse position for more stable training data
        # Add mouse wheel state to the action vector
        action = (key_vector + list(smoothed_mouse_position) + 
                 [mouse_buttons['left'], mouse_buttons['right']] +
                 [mouse_wheel['up'], mouse_wheel['down']])
        return action.copy(), pressed_keys.copy()

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    global running
    print(f"\n🛑 Received signal {signum}. Stopping recording gracefully...")
    running = False

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    actions_path = os.path.join(DATA_DIR, ACTIONS_FILE)
    actions = []
    start_index = 0

    # === MODIFIED SECTION: Check for existing data to continue recording ===
    if os.path.exists(FRAME_DIR) and len(os.listdir(FRAME_DIR)) > 0:
        existing_frames = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
        if existing_frames:
            last_frame_num = -1
            for frame_file in existing_frames:
                match = re.search(r'frame_(\d+).jpg', frame_file)
                if match:
                    last_frame_num = max(last_frame_num, int(match.group(1)))
            
            if last_frame_num != -1:
                start_index = last_frame_num + 1
                print(f"✅ Found existing data. Resuming recording from frame index {start_index}.")
                if os.path.exists(actions_path):
                    try:
                        actions = np.load(actions_path).tolist()
                        # Sanity check: ensure actions match frame count
                        if len(actions) > start_index:
                            print(f"⚠️ Action file has more entries ({len(actions)}) than frames ({start_index}). Truncating actions to match.")
                            actions = actions[:start_index]
                        print(f"   Loaded {len(actions)} existing actions.")
                    except Exception as e:
                        print(f"❌ Could not load existing actions file at {actions_path}, starting fresh. Error: {e}")
                        actions = []
                        start_index = 0 # Reset if actions are corrupt

    # ======================================================================

    print("\n" + "="*50)
    print(f"🟢 Starting HUMAN data recording in 5 seconds...")
    print("   Play the game normally.")
    print("   Press [F12] or [ESC] to quit gracefully.")
    print("   Or use [Ctrl+C] in terminal for emergency stop.")
    print("="*50 + "\n")
    time.sleep(5)
    
    # Start listeners in separate threads
    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
    key_listener.start()
    mouse_listener.start()
    
    frame_interval = 1.0 / RECORDING_FPS
    i = start_index # Use the calculated start_index
    
    try:
        last_capture_time = time.time()
        while running:
            current_time = time.time()
            if current_time - last_capture_time >= frame_interval:
                last_capture_time = current_time
                
                try:
                    frame = capture_frame()
                    action, current_keys = get_current_action()
                    
                    # Validate action vector length
                    expected_length = len(COMMON_KEYS) + 2 + 2 + 2  # keys + mouse_pos + clicks + wheel
                    if len(action) != expected_length:
                        print(f"⚠️ Invalid action length: {len(action)} (expected {expected_length}). Skipping frame.")
                        continue
                    
                    # Save frame and action
                    frame_path = os.path.join(FRAME_DIR, f"frame_{i:06d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    actions.append(action)
                    
                    # Log output to console
                    if current_keys or action[-1] or action[-2] or action[-3] or action[-4]:
                        print(f"Frame {i}: keys={list(current_keys)}, mouse=({action[-6]:.2f}, {action[-5]:.2f}), "
                              f"click_L={action[-4]}, click_R={action[-3]}, wheel_up={action[-2]}, wheel_down={action[-1]}")
                    
                    i += 1
                    
                except KeyboardInterrupt:
                    print("\n🛑 Recording interrupted by user (Ctrl+C). Saving data...")
                    break
                except Exception as e:
                    print(f"⚠️ Error capturing frame {i}: {e}. Continuing...")
                    continue
            
            # Sleep briefly to prevent high CPU usage
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted by user (Ctrl+C). Saving data...")
    finally:
        # Cleanly stop listeners
        print("🛑 Stopping input listeners...")
        key_listener.stop()
        mouse_listener.stop()
        
        # Reset mouse wheel state to prevent corruption
        with data_lock:
            mouse_wheel['up'] = 0
            mouse_wheel['down'] = 0
            mouse_buttons['left'] = 0
            mouse_buttons['right'] = 0
        
        if actions:
            try:
                # Validate all actions before saving
                expected_length = len(COMMON_KEYS) + 2 + 2 + 2
                valid_actions = []
                for i, action in enumerate(actions):
                    if len(action) == expected_length:
                        valid_actions.append(action)
                    else:
                        print(f"⚠️ Skipping invalid action at index {i}: length {len(action)} (expected {expected_length})")
                
                if valid_actions:
                    actions_array = np.array(valid_actions, dtype=np.float32)
                    np.save(actions_path, actions_array)
                    print(f"\n✅ Saved {len(valid_actions)} valid actions and frames in {DATA_DIR}.")
                    if len(valid_actions) != len(actions):
                        print(f"⚠️ Skipped {len(actions) - len(valid_actions)} invalid actions.")
                else:
                    print("\n❌ No valid actions to save!")
            except Exception as e:
                print(f"\n❌ Error saving actions: {e}")
                print("   Data may be corrupted. Check the actions manually.")
        else:
            print("\nNo actions were recorded!")