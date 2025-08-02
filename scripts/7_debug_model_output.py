#!/usr/bin/env python3
"""
Debug script for AI Game Automation
Analyzes model predictions for debugging
"""

from collections import deque

import cv2
import mss
import numpy as np
import torch
import torch.nn as nn

from config import (
    CLICK_THRESHOLD,
    COMMON_KEYS,
    IMG_HEIGHT,
    IMG_WIDTH,
    KEY_THRESHOLD,
    MODEL_FILE,
    SEQUENCE_LENGTH,
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
            cnn_output_size = self.cnn(torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)).shape[
                1
            ]

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


def capture_and_process_frame():
    """Capture and preprocess a screen frame."""
    sct = mss.mss()
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    # Resize and normalize
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0

    return torch.FloatTensor(img)


def debug_model_predictions():
    """Debug model predictions on current screen."""
    print("🔍 AI Game Automation - Model Debug")
    print("=" * 40)

    # Load model
    try:
        output_dim = len(COMMON_KEYS) + 4
        model = ImprovedBehaviorCloningCNNRNN(output_dim)
        model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
        model.eval()
        print(f"✅ Model loaded from {MODEL_FILE}")
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_FILE}")
        print("Please train a model first: python scripts/5_train_model.py")
        return

    # Capture current frame
    print("📸 Capturing current screen...")
    current_frame = capture_and_process_frame()

    # Create sequence (repeat current frame)
    frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        frame_sequence.append(current_frame)

    # Prepare input
    input_sequence = torch.stack(list(frame_sequence)).unsqueeze(0)

    # Run inference
    print("🧠 Running inference...")
    with torch.no_grad():
        output = model(input_sequence)
        last_output = output[0, -1, :].numpy()

    # Analyze predictions
    print("\n📊 Model Predictions:")
    print("-" * 30)

    # Key predictions
    key_preds = last_output[: len(COMMON_KEYS)]
    print("⌨️  Key Predictions:")
    for _, (key, pred) in enumerate(zip(COMMON_KEYS, key_preds)):
        if pred > KEY_THRESHOLD:
            print(f"  ✅ {key}: {pred:.4f} (would press)")
        else:
            print(f"  ❌ {key}: {pred:.4f} (below threshold)")

    # Mouse predictions
    mouse_x = last_output[len(COMMON_KEYS)]
    mouse_y = last_output[len(COMMON_KEYS) + 1]
    left_click = last_output[len(COMMON_KEYS) + 2]
    right_click = last_output[len(COMMON_KEYS) + 3]

    print("\n🖱️  Mouse Predictions:")
    print(f"  📍 Position: ({mouse_x:.3f}, {mouse_y:.3f})")

    if left_click > CLICK_THRESHOLD:
        print(f"  ✅ Left click would be triggered ({left_click:.4f})")
    else:
        print(f"  ❌ Left click below threshold ({left_click:.4f})")

    if right_click > CLICK_THRESHOLD:
        print(f"  ✅ Right click would be triggered ({right_click:.4f})")
    else:
        print(f"  ❌ Right click below threshold ({right_click:.4f})")

    # Summary
    print("\n📈 Summary:")
    active_keys = [
        key for key, pred in zip(COMMON_KEYS, key_preds) if pred > KEY_THRESHOLD
    ]
    print(f"  Keys that would be pressed: {active_keys if active_keys else 'None'}")
    print(
        f"  Mouse clicks: Left={left_click > CLICK_THRESHOLD}, Right={right_click > CLICK_THRESHOLD}"
    )


if __name__ == "__main__":
    debug_model_predictions()
