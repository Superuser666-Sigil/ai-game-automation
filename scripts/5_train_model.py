#!/usr/bin/env python3
"""
Training script for AI Game Automation
Trains neural network on recorded gameplay data
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, random_split

from config import (
    ACTIONS_FILE,
    BATCH_SIZE,
    COMMON_KEYS,
    EPOCHS,
    FRAME_DIR,
    LEARNING_RATE,
    MODEL_FILE,
    OVERSAMPLE_ACTION_FRAMES_MULTIPLIER,
    SEQUENCE_LENGTH,
    TRAIN_IMG_HEIGHT,
    TRAIN_IMG_WIDTH,
    VALIDATION_SPLIT,
)


class WoWSequenceDataset(Dataset):
    """Dataset for training with oversampling of action frames."""

    def __init__(self, frame_dir, actions_file, sequence_length=5):
        self.frame_dir = frame_dir
        self.actions = np.load(actions_file, allow_pickle=True)
        self.sequence_length = sequence_length

        # Find frames with actions for oversampling
        action_frames = set()
        for action in self.actions:
            if action["type"] in ["key_press", "mouse_click"]:
                action_frames.add(action["frame"])

        # Create oversampled dataset
        self.oversampled_indices = []
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

        for i in range(len(frame_files) - sequence_length + 1):
            # Check if this sequence contains action frames
            sequence_frames = set(range(i, i + sequence_length))
            has_actions = bool(sequence_frames & action_frames)

            if has_actions:
                # Oversample action sequences
                for _ in range(OVERSAMPLE_ACTION_FRAMES_MULTIPLIER):
                    self.oversampled_indices.append(i)
            else:
                # Normal sampling for non-action sequences
                self.oversampled_indices.append(i)

    def __len__(self):
        return len(self.oversampled_indices)

    def __getitem__(self, idx):
        start_idx = self.oversampled_indices[idx]

        # Load sequence of frames
        frames = []
        for i in range(self.sequence_length):
            frame_path = os.path.join(self.frame_dir, f"frame_{start_idx + i:06d}.png")
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (TRAIN_IMG_WIDTH, TRAIN_IMG_HEIGHT))
            frame = frame / 255.0  # Normalize
            frames.append(frame)

        frames = np.array(frames)

        # Create target vector
        target = np.zeros(len(COMMON_KEYS) + 4)  # keys + mouse_pos + mouse_click

        # Get actions for this sequence
        sequence_actions = [
            a
            for a in self.actions
            if start_idx <= a["frame"] < start_idx + self.sequence_length
        ]

        # Process key actions
        for action in sequence_actions:
            if action["type"] == "key_press" and action["key"] in COMMON_KEYS:
                key_idx = COMMON_KEYS.index(action["key"])
                target[key_idx] = 1.0

        # Process mouse actions (simplified)
        for action in sequence_actions:
            if action["type"] == "mouse_move":
                # Normalize mouse position (assuming 1920x1080)
                target[-4] = action["x"] / 1920
                target[-3] = action["y"] / 1080
            elif action["type"] == "mouse_click":
                if "left" in str(action["button"]):
                    target[-2] = 1.0
                if "right" in str(action["button"]):
                    target[-1] = 1.0

        return torch.FloatTensor(frames), torch.FloatTensor(target)


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
            cnn_output_size = self.cnn(
                torch.zeros(1, 3, TRAIN_IMG_HEIGHT, TRAIN_IMG_WIDTH)
            ).shape[1]

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


def validate(model, val_loader, device):
    """Validate model and return F1 score."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for frames, targets in val_loader:
            frames = frames.to(device)
            targets = targets.to(device)

            outputs = model(frames)
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

            # Convert to binary predictions
            preds = (outputs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate F1 score
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate F1 for key predictions only
    key_preds = all_preds[:, : len(COMMON_KEYS)]
    key_targets = all_targets[:, : len(COMMON_KEYS)]

    precision, recall, f1, _ = precision_recall_fscore_support(
        key_targets.flatten(), key_preds.flatten(), average="binary"
    )

    return f1


def main():
    """Main training function."""
    print("🧠 AI Game Automation - Model Training")
    print("=" * 50)

    # Check if data exists
    if not os.path.exists(FRAME_DIR):
        print(f"❌ Frames directory not found: {FRAME_DIR}")
        print("Please record some data first: python scripts/3_record_data.py")
        return

    if not os.path.exists(ACTIONS_FILE):
        print(f"❌ Actions file not found: {ACTIONS_FILE}")
        print("Please record some data first: python scripts/3_record_data.py")
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Create dataset
    print("📊 Loading dataset...")
    dataset = WoWSequenceDataset(FRAME_DIR, ACTIONS_FILE, SEQUENCE_LENGTH)

    # Split into train/validation
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    output_dim = len(COMMON_KEYS) + 4
    model = ImprovedBehaviorCloningCNNRNN(output_dim).to(device)

    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Output dimension: {output_dim}")
    print(f"📊 Keys to learn: {len(COMMON_KEYS)}")

    # Training loop
    print("\n🚀 Starting training...")
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_idx, (frames, targets) in enumerate(train_loader):
            frames = frames.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(frames)

            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1, targets.shape[-1])

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}"
                )

        # Validation
        val_f1 = validate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: Loss={avg_loss:.4f}, " f"Val F1={val_f1:.4f}"
        )

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"💾 Saved best model (F1: {val_f1:.4f})")

        # Learning rate scheduling
        scheduler.step(val_f1)

    print(f"\n✅ Training complete! Best F1: {best_f1:.4f}")
    print(f"💾 Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
