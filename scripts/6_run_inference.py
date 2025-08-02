#!/usr/bin/env python3
"""
Inference script for AI Game Automation
Runs trained model to control games automatically
"""

import time
from collections import deque

import cv2
import mss
import numpy as np
import torch
import torch.nn as nn
from pynput import keyboard, mouse
from torchvision import transforms

from config import (
    CLICK_THRESHOLD,
    COMMON_KEYS,
    IMG_HEIGHT,
    IMG_WIDTH,
    INFERENCE_FPS,
    KEY_MAPPING,
    KEY_THRESHOLD,
    MODEL_FILE,
    MOUSE_SMOOTHING_ALPHA,
    SEQUENCE_LENGTH,
    SMOOTH_FACTOR,
)


class ImprovedBehaviorCloningCNNRNN(nn.Module):
    """Improved neural network for behavior cloning."""

    def __init__(self, output_dim):
        super().__init__()

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_output_size = self.cnn(dummy_input).shape[1]

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Output heads
        self.key_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, len(COMMON_KEYS)),
            nn.Sigmoid(),
        )

        self.mouse_pos_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid()
        )

        self.mouse_click_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(), nn.Linear(32, 2), nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape

        # Process through CNN
        cnn_out = self.cnn(x.view(batch_size * seq_len, channels, height, width))
        lstm_in = cnn_out.view(batch_size, seq_len, -1)

        # Process through LSTM
        lstm_out, _ = self.lstm(lstm_in)
        lstm_flat = lstm_out.reshape(batch_size * seq_len, -1)

        # Generate outputs
        key_out = self.key_head(lstm_flat)
        mouse_pos_out = self.mouse_pos_head(lstm_flat)
        mouse_click_out = self.mouse_click_head(lstm_flat)

        # Combine outputs
        combined = torch.cat([key_out, mouse_pos_out, mouse_click_out], dim=1)
        return combined.view(batch_size, seq_len, -1)


# Initialize model
output_dim = len(COMMON_KEYS) + 4
model = ImprovedBehaviorCloningCNNRNN(output_dim)

# Load trained model
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    print(f"✅ Model loaded from {MODEL_FILE}")
except FileNotFoundError:
    print(f"❌ Model file not found: {MODEL_FILE}")
    print("Please train a model first: python scripts/5_train_model.py")
    exit(1)

# Initialize controllers
keyboard_controller = keyboard.Controller()
mouse_controller = mouse.Controller()

# Screen capture setup
sct = mss.mss()
monitor = sct.monitors[1]  # Primary monitor
SCREEN_WIDTH = monitor["width"]
SCREEN_HEIGHT = monitor["height"]

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Global state
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_pressed_keys = set()
current_mouse_buttons = set()
prev_mouse_pos = None


def apply_output(output):
    """Apply model output to keyboard and mouse."""
    global prev_mouse_pos

    raw_output = output.detach().cpu().numpy()

    key_preds = raw_output[: len(COMMON_KEYS)]
    mouse_x = raw_output[len(COMMON_KEYS)] * SCREEN_WIDTH
    mouse_y = raw_output[len(COMMON_KEYS) + 1] * SCREEN_HEIGHT
    left_click = raw_output[len(COMMON_KEYS) + 2] > CLICK_THRESHOLD
    right_click = raw_output[len(COMMON_KEYS) + 3] > CLICK_THRESHOLD

    # Apply key presses
    for _, (key, pred) in enumerate(zip(COMMON_KEYS, key_preds)):
        if pred > KEY_THRESHOLD:
            if key not in current_pressed_keys:
                current_pressed_keys.add(key)
                if key in KEY_MAPPING:
                    keyboard_controller.press(KEY_MAPPING[key])
        else:
            if key in current_pressed_keys:
                current_pressed_keys.discard(key)
                if key in KEY_MAPPING:
                    keyboard_controller.release(KEY_MAPPING[key])

    # Apply mouse movement with smoothing
    if prev_mouse_pos is None:
        prev_mouse_pos = (mouse_x, mouse_y)

    smooth_x = prev_mouse_pos[0] * SMOOTH_FACTOR + mouse_x * (1 - SMOOTH_FACTOR)
    smooth_y = prev_mouse_pos[1] * SMOOTH_FACTOR + mouse_y * (1 - SMOOTH_FACTOR)

    mouse_controller.position = (int(smooth_x), int(smooth_y))
    prev_mouse_pos = (smooth_x, smooth_y)

    # Apply mouse clicks
    if left_click:
        mouse_controller.press(mouse.Button.left)
    else:
        mouse_controller.release(mouse.Button.left)

    if right_click:
        mouse_controller.press(mouse.Button.right)
    else:
        mouse_controller.release(mouse.Button.right)


def smooth_mouse_movement(target_mouse_pos):
    """Smooth mouse movement to target position."""
    current_pos = mouse_controller.position
    new_x = current_pos[0] * MOUSE_SMOOTHING_ALPHA + target_mouse_pos[0] * (
        1 - MOUSE_SMOOTHING_ALPHA
    )
    new_y = current_pos[1] * MOUSE_SMOOTHING_ALPHA + target_mouse_pos[1] * (
        1 - MOUSE_SMOOTHING_ALPHA
    )
    mouse_controller.position = (int(new_x), int(new_y))


def capture_and_process_frame():
    """Capture and preprocess a screen frame."""
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_tensor = transform(img)
    return img_tensor


def main():
    """Main inference function."""
    print("🤖 AI Game Automation - Inference")
    print("=" * 40)
    print("Press Ctrl+C to stop")

    # Initialize frame sequence
    for _ in range(SEQUENCE_LENGTH):
        frame_sequence.append(capture_and_process_frame())
        time.sleep(0.1)

    frame_interval = 1.0 / INFERENCE_FPS
    last_inference_time = time.time()

    try:
        while True:
            current_time = time.time()

            if current_time - last_inference_time >= frame_interval:
                # Capture new frame
                new_frame = capture_and_process_frame()
                frame_sequence.append(new_frame)

                # Prepare input for model
                input_sequence = torch.stack(list(frame_sequence)).unsqueeze(0)

                # Run inference
                with torch.no_grad():
                    output = model(input_sequence)
                    # Use the last prediction in the sequence
                    last_output = output[0, -1, :]

                # Apply output
                apply_output(last_output)

                last_inference_time = current_time

            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    except KeyboardInterrupt:
        print("\n⏹️  Stopping inference...")

        # Release all keys
        for key in current_pressed_keys:
            if key in KEY_MAPPING:
                keyboard_controller.release(KEY_MAPPING[key])

        # Release all mouse buttons
        mouse_controller.release(mouse.Button.left)
        mouse_controller.release(mouse.Button.right)

        print("✅ Cleanup complete")


if __name__ == "__main__":
    main()
