# 5_train_model.py - Improved Behavior Cloning with DirectML Support
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Import shared configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Validate configuration consistency
validate_config()

# Use training-specific dimensions from config
IMG_HEIGHT = TRAIN_IMG_HEIGHT
IMG_WIDTH = TRAIN_IMG_WIDTH
SEQUENCE_LENGTH = 3  # Reduced sequence length for memory efficiency

class WoWSequenceDataset(Dataset):
    def __init__(self, frame_dir, actions_file, sequence_length, transform=None):
        self.frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        self.actions = np.load(actions_file).astype(np.float32)
        
        # Ensure we have matching data
        min_len = min(len(self.frame_paths), len(self.actions))
        self.frame_paths = self.frame_paths[:min_len]
        self.actions = self.actions[:min_len]
        
        self.transform = transform
        self.sequence_length = sequence_length
        
        print(f"Dataset: {len(self.frame_paths)} frames, {len(self.actions)} actions")
        
        # Analyze the data
        key_actions = self.actions[:, :len(COMMON_KEYS)]
        key_totals = np.sum(key_actions, axis=0)
        active_keys = [(COMMON_KEYS[i], int(total)) for i, total in enumerate(key_totals) if total > 0]
        print(f"Active keys in data: {active_keys}")
        
        # Check for class imbalance
        total_key_presses = np.sum(key_actions)
        total_frames = len(self.actions)
        print(f"Key press rate: {total_key_presses / (total_frames * len(COMMON_KEYS)):.4f}")
        
        if total_key_presses / (total_frames * len(COMMON_KEYS)) < 0.01:
            print("⚠️  WARNING: Very low key press rate! This will cause training issues.")
    
    def __len__(self):
        return max(0, len(self.frame_paths) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        seq_frames = []
        
        for i in range(idx, end_idx):
            try:
                img = cv2.imread(self.frame_paths[i])
                if img is None:
                    raise ValueError(f"Could not load image: {self.frame_paths[i]}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.transform:
                    img = self.transform(img)
                seq_frames.append(img)
            except Exception as e:
                print(f"Error loading frame {i}: {e}")
                black_frame = torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)
                seq_frames.append(black_frame)
        
        seq_actions = self.actions[idx:end_idx]
        
        return torch.stack(seq_frames), torch.tensor(seq_actions, dtype=torch.float32)

class ImprovedBehaviorCloningCNNRNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        # Simplified CNN for better training
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        print(f"CNN output size: {cnn_output_size}")
        
        # DirectML-compatible temporal processing (replaces LSTM)
        # Simple but effective temporal layers
        self.temporal_layers = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256)  # Final temporal feature size
        )
        
        # Improved key head with better initialization
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
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent vanishing gradients"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        b, s, c, h, w = x.shape
        
        # Process through CNN
        cnn_out = self.cnn(x.view(b * s, c, h, w))
        temporal_in = cnn_out.view(b, s, -1)
        
        # Process through DirectML-compatible temporal layers
        # Apply temporal processing to each frame independently
        temporal_out = self.temporal_layers(temporal_in.view(b * s, -1))
        temporal_out = temporal_out.view(b, s, -1)
        
        # Simple temporal aggregation - use the last frame's features
        # (Alternative to LSTM's hidden state)
        final_features = temporal_out[:, -1, :]  # Take last timestep
        
        # Apply heads to final features
        key_out = self.key_head(final_features)
        mouse_pos_out = self.mouse_pos_head(final_features)
        mouse_click_out = self.mouse_click_head(final_features)
        
        # Combine outputs and expand back to sequence dimension for compatibility
        combined = torch.cat([key_out, mouse_pos_out, mouse_click_out], dim=1)
        
        return combined.unsqueeze(1).expand(b, s, -1)

def calculate_balanced_weights(actions, num_keys):
    """Calculate balanced weights that don't over-penalize key presses"""
    key_actions = actions[:, :num_keys]
    pos_counts = np.sum(key_actions, axis=0)
    neg_counts = len(actions) - pos_counts
    
    # Avoid division by zero and extreme weights
    pos_counts = np.maximum(pos_counts, 1)
    neg_counts = np.maximum(neg_counts, 1)
    
    # Use square root to reduce extreme weight differences
    weights = np.sqrt(neg_counts / pos_counts)
    weights = np.clip(weights, 1.0, 10.0)  # Much more reasonable limits
    
    return torch.tensor(weights, dtype=torch.float32)

def select_device():
    """Smart device selection: DirectML -> ROCm/CUDA -> CPU"""
    device = None
    device_name = "Unknown"
    
    # Try DirectML first (best for AMD GPUs on Windows)
    try:
        import torch_directml
        device = torch_directml.device()
        device_name = "DirectML (AMD GPU acceleration)"
        print("🚀 Using DirectML for GPU acceleration!")
    except ImportError:
        pass
    
    # Fallback to ROCm/CUDA
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = f"CUDA (GPU: {torch.cuda.get_device_name()})"
            print("⚡ Using CUDA for GPU acceleration!")
        else:
            # Check for ROCm (AMD's CUDA alternative)
            try:
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    device = torch.device("cuda")  # ROCm uses cuda device interface
                    device_name = "ROCm (AMD GPU acceleration)"
                    print("🔥 Using ROCm for AMD GPU acceleration!")
            except:
                pass
    
    # Final fallback to CPU
    if device is None:
        device = torch.device("cpu")
        device_name = "CPU (no GPU acceleration)"
        print("💻 Using CPU (consider installing DirectML for GPU acceleration)")
    
    return device, device_name

def improved_loss(outputs, targets, num_keys, device, key_weights):
    b, s, _ = outputs.shape
    
    # Reshape to treat sequence as batch (use reshape for non-contiguous tensors)
    outputs = outputs.reshape(b * s, -1)
    targets = targets.reshape(b * s, -1)
    
    # Split outputs and targets
    key_outputs = outputs[:, :num_keys]
    key_targets = targets[:, :num_keys]
    mouse_pos_outputs = outputs[:, num_keys:num_keys+2]
    mouse_pos_targets = targets[:, num_keys:num_keys+2]
    mouse_click_outputs = outputs[:, num_keys+2:]
    mouse_click_targets = targets[:, num_keys+2:]
    
    # Focal loss for keys to handle class imbalance better
    alpha = 0.25
    gamma = 2.0
    
    key_weights_expanded = key_weights.unsqueeze(0).expand_as(key_targets)
    
    # Focal loss implementation
    pt = key_targets * key_outputs + (1 - key_targets) * (1 - key_outputs)
    focal_weight = (1 - pt) ** gamma
    key_loss = -alpha * focal_weight * (key_targets * torch.log(key_outputs + 1e-7) + 
                                       (1 - key_targets) * torch.log(1 - key_outputs + 1e-7))
    key_loss = (key_loss * key_weights_expanded).mean()
    
    # MSE for mouse position (DirectML compatible)
    diff = mouse_pos_outputs - mouse_pos_targets
    pos_loss = torch.mean(diff * diff)
    
    # L1 regularization for mouse movement smoothness
    # Penalize large differences between consecutive mouse predictions
    mouse_smoothness_loss = 0.0
    if s > 1:  # Only if we have sequences
        # Reshape back to sequence format for temporal analysis
        mouse_seq_outputs = mouse_pos_outputs.reshape(b, s, 2)
        mouse_seq_targets = mouse_pos_targets.reshape(b, s, 2)
        
        # Calculate L1 loss on consecutive frame differences
        mouse_diff_pred = torch.abs(mouse_seq_outputs[:, 1:] - mouse_seq_outputs[:, :-1])
        mouse_diff_target = torch.abs(mouse_seq_targets[:, 1:] - mouse_seq_targets[:, :-1])
        
        # Encourage predicted movement to match target movement patterns (DirectML compatible)
        mouse_smoothness_loss = torch.mean(torch.abs(mouse_diff_pred - mouse_diff_target))
    
    # BCE for mouse clicks (DirectML compatible)
    # Custom BCE: -(target * log(input) + (1 - target) * log(1 - input))
    eps = 1e-7
    mouse_click_outputs_clamped = torch.clamp(mouse_click_outputs, eps, 1 - eps)
    click_loss = -torch.mean(mouse_click_targets * torch.log(mouse_click_outputs_clamped) + 
                            (1 - mouse_click_targets) * torch.log(1 - mouse_click_outputs_clamped))
    
    # Balanced loss weights (reduced key loss dominance for better learning)
    total_loss = key_loss * KEY_LOSS_WEIGHT + pos_loss * POS_LOSS_WEIGHT + click_loss * CLICK_LOSS_WEIGHT + mouse_smoothness_loss * SMOOTHNESS_LOSS_WEIGHT
    
    return total_loss, key_loss, pos_loss, click_loss, mouse_smoothness_loss

def train():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    all_datasets = []
    all_actions = []
    
    # Use centralized config paths
    frame_dir = FRAME_DIR
    actions_file = ACTIONS_FILE
        
        if os.path.exists(actions_file) and os.path.exists(frame_dir):
            print(f"Loading dataset from: {DATA_DIR}")
            dataset = WoWSequenceDataset(frame_dir, actions_file, SEQUENCE_LENGTH, transform)
            if len(dataset) > 0:
                all_datasets.append(dataset)
                all_actions.append(dataset.actions)
        else:
            print(f"Warning: Dataset not found at {DATA_DIR}")
    
    if not all_datasets:
        print("No valid datasets found!")
        return
    
    # Calculate balanced class weights
    combined_actions = np.vstack(all_actions)
    key_weights = calculate_balanced_weights(combined_actions, len(COMMON_KEYS))
    
    print(f"Key weights range: [{key_weights.min():.2f}, {key_weights.max():.2f}]")
    
    combined_dataset = ConcatDataset(all_datasets)
    dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    output_dim = len(COMMON_KEYS) + 4
    model = ImprovedBehaviorCloningCNNRNN(output_dim)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    
    # Smart device selection using utility function
    device, device_name = select_device()
    
    model.to(device)
    key_weights = key_weights.to(device)
    
    print(f"Training on {device_name} for {EPOCHS} epochs...")
    print(f"Dataset size: {len(combined_dataset)} sequences")
    
    # Training loop with detailed logging
    loss_history = []
    key_loss_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_key_loss = 0.0
        running_pos_loss = 0.0
        running_click_loss = 0.0
        running_smoothness_loss = 0.0
        
        for batch_idx, (sequences, actions) in enumerate(dataloader):
            sequences, actions = sequences.to(device), actions.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            total_loss, key_loss, pos_loss, click_loss, smoothness_loss = improved_loss(
                outputs, actions, len(COMMON_KEYS), device, key_weights
            )
            
            total_loss.backward()
            
            # Gradient clipping (increased for better learning)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            
            optimizer.step()
            
            running_loss += total_loss.item()
            running_key_loss += key_loss.item()
            running_pos_loss += pos_loss.item()
            running_click_loss += click_loss.item()
            running_smoothness_loss += smoothness_loss.item()
        
        scheduler.step(running_loss / len(dataloader))
        
        avg_loss = running_loss / len(dataloader)
        avg_key_loss = running_key_loss / len(dataloader)
        avg_pos_loss = running_pos_loss / len(dataloader)
        avg_click_loss = running_click_loss / len(dataloader)
        avg_smoothness_loss = running_smoothness_loss / len(dataloader)
        
        loss_history.append(avg_loss)
        key_loss_history.append(avg_key_loss)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"  Total Loss: {avg_loss:.5f}")
        print(f"  Key Loss: {avg_key_loss:.5f}")
        print(f"  Mouse Pos Loss: {avg_pos_loss:.5f}")
        print(f"  Mouse Click Loss: {avg_click_loss:.5f}")
        print(f"  Mouse Smoothness Loss: {avg_smoothness_loss:.5f}")
        print(f"  Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping if key loss is very low (model learned to ignore keys)
        if avg_key_loss < 0.001:
            print("⚠️  WARNING: Key loss is very low - model may be ignoring keys!")
    
    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
    
    # Plot training loss
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('Total Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(key_loss_history)
    plt.title('Key Loss (Should not be too low)')
    plt.xlabel('Epoch')
    plt.ylabel('Key Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('improved_training_loss.png')
    plt.show()

if __name__ == "__main__":
    train()