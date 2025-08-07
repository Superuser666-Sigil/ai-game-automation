import os
import re
import math
import subprocess
import threading
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from config import * # Import all settings

# === EARLY STOPPING ===
class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return
        if metric < self.best_score + self.min_delta:
            self.counter += 1
        else:
            self.best_score = metric
            self.counter = 0
        if self.counter >= self.patience:
            self.early_stop = True

# === DATASET ===
class WoWSequenceDataset(Dataset):
    """Custom dataset for loading sequences of frames and actions."""
    def __init__(self, frame_dir, actions_file, sequence_length, transform=None):
        self.transform = transform
        self.sequence_length = sequence_length

        frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        actions = np.load(actions_file).astype(np.float32)

        min_len = min(len(frame_paths), len(actions))
        self.frame_paths = frame_paths[:min_len]
        self.actions = actions[:min_len]

        self.indices = []
        num_keys = len(COMMON_KEYS)
        action_frames = 0

        for i in range(len(self.frame_paths) - self.sequence_length + 1):
            last_action = self.actions[i + self.sequence_length - 1]
            key_press = np.sum(last_action[:num_keys]) > 0
            mouse_click = np.sum(last_action[num_keys+2:]) > 0

            if key_press or mouse_click:
                self.indices.extend([i] * OVERSAMPLE_ACTION_FRAMES_MULTIPLIER)
                action_frames += 1
            else:
                self.indices.append(i)

        print(f"Loaded from {frame_dir}: {len(self.frame_paths)} frames, "
              f"{action_frames} action sequences -> {len(self.indices)} total sequences after oversampling.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_index = self.indices[idx]
        end_index = start_index + self.sequence_length

        imgs = []
        for i in range(start_index, end_index):
            img = cv2.imread(self.frame_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        seq_actions = self.actions[start_index:end_index]
        return torch.stack(imgs), torch.tensor(seq_actions, dtype=torch.float32)

# === NEW: POSITIONAL ENCODING ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# === UPDATED MODEL: TRANSFORMER ===
class BehaviorCloningTransformer(nn.Module):
    """CNN-Transformer model for behavior cloning."""
    def __init__(self, output_dim, d_model, nhead, nlayers, dropout):
        super().__init__()
        # 1. CNN Feature Extractor
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

        # 2. Projection layer to match transformer's d_model
        self.input_proj = nn.Linear(cnn_out_size, d_model)
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQUENCE_LENGTH)
        
        # 4. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # 5. Enhanced Output heads with better architecture
        # Separate heads for different action types with different complexities
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
        
        # Pass through CNN
        feat = self.cnn(x_reshaped)
        feat_reshaped = feat.view(b, s, -1)
        
        # Project and add positional encoding
        projected_feat = self.input_proj(feat_reshaped) * math.sqrt(self.d_model)
        pos_encoded_feat = self.pos_encoder(projected_feat)
        
        # Pass through Transformer
        transformer_out = self.transformer_encoder(pos_encoded_feat)
        
        # Pass through output heads
        key_out = self.key_head(transformer_out)
        pos_out = self.mouse_pos_head(transformer_out)
        click_out = self.mouse_click_head(transformer_out)
        wheel_out = self.mouse_wheel_head(transformer_out)
        
        # Concatenate for loss calculation
        return torch.cat([key_out, pos_out, click_out, wheel_out], dim=2)

# === IMPROVED LOSS FUNCTION WITH CLASS WEIGHTS ===
class FocalLoss(nn.Module):
    """Focal Loss to handle class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def weighted_bce_mse_loss(outputs, targets, writer=None, step=None):
    """Enhanced loss function with focal loss for imbalanced classes."""
    num_keys = len(COMMON_KEYS)
    
    # Updated slicing for new action vector: [keys, mouse_pos, clicks, wheel]
    key_out = outputs[..., :num_keys]
    pos_out = outputs[..., num_keys:num_keys+2]
    click_out = outputs[..., num_keys+2:num_keys+4]  # Left/right clicks
    wheel_out = outputs[..., num_keys+4:num_keys+6]  # Wheel up/down
    
    key_tgt = targets[..., :num_keys]
    pos_tgt = targets[..., num_keys:num_keys+2]
    click_tgt = targets[..., num_keys+2:num_keys+4]
    wheel_tgt = targets[..., num_keys+4:num_keys+6]
    
    # Use focal loss for keys, clicks, and wheel to handle class imbalance
    focal_loss = FocalLoss(alpha=1, gamma=2)
    loss_keys = focal_loss(key_out, key_tgt)
    loss_clicks = focal_loss(click_out, click_tgt)
    loss_wheel = focal_loss(wheel_out, wheel_tgt)
    
    # Use MSE for mouse position (regression)
    mse_loss = nn.MSELoss()
    loss_pos = mse_loss(pos_out, pos_tgt)
    
    # Weight the losses based on importance and class balance
    # Give more weight to clicks and wheel since they're rare
    total_loss = loss_keys + 3.0 * loss_clicks + 2.0 * loss_wheel + 0.5 * loss_pos
    
    # Log individual losses if writer is provided
    if writer and step is not None:
        writer.add_scalar("Loss/keys", loss_keys.item(), step)
        writer.add_scalar("Loss/clicks", loss_clicks.item(), step)
        writer.add_scalar("Loss/wheel", loss_wheel.item(), step)
        writer.add_scalar("Loss/position", loss_pos.item(), step)
        writer.add_scalar("Loss/total", total_loss.item(), step)
    
    return total_loss

# === VALIDATION ===
def validate(model, dataloader, device, writer, epoch):
    model.eval()
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for seqs, acts in dataloader:
            seqs, acts = seqs.to(device), acts.to(device)
            out = model(seqs)
            
            preds_keys_clicks = torch.sigmoid(torch.cat([out[..., :len(COMMON_KEYS)], out[..., len(COMMON_KEYS)+2:]], dim=-1))
            preds_pos = out[..., len(COMMON_KEYS):len(COMMON_KEYS)+2]
            preds = torch.cat([preds_keys_clicks, preds_pos], dim=-1)
            
            preds_agg = preds[:, -VALIDATION_WINDOW:, :].mean(dim=1)
            tgts_agg = acts[:, -VALIDATION_WINDOW:, :].max(dim=1)[0]

            all_preds.append(preds_agg.cpu().numpy())
            all_tgts.append(tgts_agg.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_tgts = np.vstack(all_tgts)

    num_keys = len(COMMON_KEYS)
    # Updated slicing for new action vector: [keys, mouse_pos, clicks, wheel]
    key_preds = all_preds[:, :num_keys]
    pos_preds = all_preds[:, num_keys:num_keys+2]
    click_preds = all_preds[:, num_keys+2:num_keys+4]
    wheel_preds = all_preds[:, num_keys+4:num_keys+6]
    
    key_tgts = all_tgts[:, :num_keys]
    pos_tgts = all_tgts[:, num_keys:num_keys+2]
    click_tgts = all_tgts[:, num_keys+2:num_keys+4]
    wheel_tgts = all_tgts[:, num_keys+4:num_keys+6]

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in THRESHOLD_SWEEP:
        binary_preds = (np.hstack([key_preds, click_preds, wheel_preds]) > thresh).astype(int)
        binary_tgts = np.hstack([key_tgts, click_tgts, wheel_tgts])
        _, _, f1, _ = precision_recall_fscore_support(binary_tgts, binary_preds, average="samples", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nValidation Best F1: {best_f1:.4f} at threshold {best_thresh:.2f}")

    # Enhanced logging for TensorBoard
    writer.add_scalar("Validation/Best_F1", best_f1, epoch)
    writer.add_scalar("Validation/Best_Threshold", best_thresh, epoch)
    
    # Log prediction statistics
    writer.add_scalar("Validation/Key_Pred_Mean", key_preds.mean(), epoch)
    writer.add_scalar("Validation/Key_Pred_Std", key_preds.std(), epoch)
    writer.add_scalar("Validation/Click_Pred_Mean", click_preds.mean(), epoch)
    writer.add_scalar("Validation/Click_Pred_Std", click_preds.std(), epoch)
    
    # Log action counts
    key_action_count = np.sum(np.any(key_tgts > 0, axis=1))
    click_action_count = np.sum(np.any(click_tgts > 0, axis=1))
    wheel_action_count = np.sum(np.any(wheel_tgts > 0, axis=1))
    writer.add_scalar("Validation/Key_Action_Count", key_action_count, epoch)
    writer.add_scalar("Validation/Click_Action_Count", click_action_count, epoch)
    writer.add_scalar("Validation/Wheel_Action_Count", wheel_action_count, epoch)
    
    # Create confusion matrix
    cm_preds = (np.hstack([key_preds, click_preds, wheel_preds]) > best_thresh).astype(int).flatten()
    cm_tgts = np.hstack([key_tgts, click_tgts, wheel_tgts]).flatten()
    cm = confusion_matrix(cm_tgts, cm_preds)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                xticklabels=['No Action', 'Action'], 
                yticklabels=['No Action', 'Action'])
    ax.set_title(f"Confusion Matrix (Thresh={best_thresh:.2f})")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    writer.add_figure("Validation/ConfusionMatrix", fig, epoch)
    plt.close(fig)
    
    # Plot prediction distributions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.hist(key_preds.flatten(), bins=50, alpha=0.7, label='Key predictions')
    ax1.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold {best_thresh:.2f}')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Key Prediction Distribution')
    ax1.legend()
    
    ax2.hist(click_preds.flatten(), bins=50, alpha=0.7, label='Click predictions')
    ax2.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold {best_thresh:.2f}')
    ax2.set_xlabel('Prediction Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Click Prediction Distribution')
    ax2.legend()
    
    ax3.hist(wheel_preds.flatten(), bins=50, alpha=0.7, label='Wheel predictions')
    ax3.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold {best_thresh:.2f}')
    ax3.set_xlabel('Prediction Probability')
    ax3.set_ylabel('Count')
    ax3.set_title('Wheel Prediction Distribution')
    ax3.legend()
    
    ax4.hist(pos_preds.flatten(), bins=50, alpha=0.7, label='Position predictions')
    ax4.set_xlabel('Prediction Value')
    ax4.set_ylabel('Count')
    ax4.set_title('Mouse Position Distribution')
    ax4.legend()
    
    writer.add_figure("Validation/Prediction_Distributions", fig, epoch)
    plt.close(fig)

    return best_f1, best_thresh

# === TENSORBOARD STARTUP ===
def start_tensorboard():
    """Start TensorBoard in a separate thread."""
    try:
        # Create the log directory if it doesn't exist
        os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
        
        # Start TensorBoard process
        cmd = ["tensorboard", "--logdir", TENSORBOARD_LOG_DIR, "--port", "6006", "--host", "0.0.0.0"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ TensorBoard started successfully!")
            print(f"   Open your browser and go to: http://localhost:6006")
            print(f"   Log directory: {TENSORBOARD_LOG_DIR}")
            return process
        else:
            print("❌ Failed to start TensorBoard")
            return None
    except Exception as e:
        print(f"❌ Error starting TensorBoard: {e}")
        return None

# === MAIN TRAINING LOOP ===
def train(start_tensorboard_auto=True):
    # Start TensorBoard if requested
    tb_process = None
    if start_tensorboard_auto:
        print("🚀 Starting TensorBoard...")
        tb_process = start_tensorboard()
    
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    early_stopper = EarlyStopping()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = [ds for d in DATA_DIRS if (fdir := os.path.join(d, "frames")) and (afile := os.path.join(d, ACTIONS_FILE)) and os.path.exists(fdir) and os.path.exists(afile) and len(ds := WoWSequenceDataset(fdir, afile, SEQUENCE_LENGTH, transform)) > 0]
    if not datasets:
        print("❌ No valid datasets found! Check DATA_DIRS in config.py.")
        return

    full_dataset = ConcatDataset(datasets)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"\nTotal sequences: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = len(COMMON_KEYS) + 4
    
    model = BehaviorCloningTransformer(output_dim, D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    start_epoch = 1
    best_f1 = 0.0
    if os.path.exists(MODEL_SAVE_DIR) and (checkpoint_files := list(Path(MODEL_SAVE_DIR).glob("model_epoch_*.pth"))):
        latest_checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        if match := re.search(r"model_epoch_(\d+).pth", latest_checkpoint_path.name):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_f1 = checkpoint.get('best_f1', 0.0)
                print(f"\n✅ Resuming training from {latest_checkpoint_path} at epoch {start_epoch}. Best F1: {best_f1:.4f}\n")
            except Exception as e:
                print(f"\n⚠️ Could not load checkpoint {latest_checkpoint_path}. Starting fresh. Error: {e}\n")

    print(f"\n🚀 Starting training on {device} for {EPOCHS} epochs…\n")

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for i, (seqs, acts) in enumerate(train_loader, 1):
                seqs, acts = seqs.to(device), acts.to(device)
                
                optimizer.zero_grad()
                outputs = model(seqs)
                
                # Calculate step for logging
                step = (epoch - 1) * len(train_loader) + i
                loss = weighted_bce_mse_loss(outputs, acts, writer, step)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                running_loss += loss.item()
                
                # Enhanced logging every 100 steps
                if i % 100 == 0:
                    # Log learning rate
                    writer.add_scalar("Training/Learning_Rate", 
                                    optimizer.param_groups[0]['lr'], step)
                    
                    # Log gradient norms
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    writer.add_scalar("Training/Gradient_Norm", total_norm, step)
                    
                    print(f"  Epoch {epoch}/{EPOCHS} | Step {i}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            avg_loss = running_loss / len(train_loader)
            writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
            print(f"Epoch {epoch} Summary | Avg Loss: {avg_loss:.4f}")

            val_f1, best_thresh = validate(model, val_loader, device, writer, epoch)
            writer.add_scalar("F1/validation", val_f1, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
            scheduler.step(val_f1)

            early_stopper(val_f1)
            if early_stopper.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}.")
                break

            ckpt_path = MODEL_SAVE_PATH_TEMPLATE.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, ckpt_path)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                }, MODEL_FILE)
                print(f"⭐ New best model saved to {MODEL_FILE} (F1={best_f1:.4f})")

                best_threshold_path = os.path.join(os.path.dirname(MODEL_FILE) or ".", "best_threshold.txt")
                try:
                    with open(best_threshold_path, "w") as f: f.write(str(best_thresh))
                    print(f"   Saved best threshold ({best_thresh:.2f}) to {best_threshold_path}")
                except Exception as e:
                    print(f"   Could not save best threshold: {e}")

    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user. Saving final state...")
    finally:
        writer.close()
        print("\n✅ Training complete. TensorBoard logs saved.")
        
        # Clean up TensorBoard process if it was started
        if tb_process and tb_process.poll() is None:
            print("🛑 Stopping TensorBoard...")
            tb_process.terminate()
            try:
                tb_process.wait(timeout=5)
                print("✅ TensorBoard stopped successfully.")
            except subprocess.TimeoutExpired:
                print("⚠️ Force killing TensorBoard...")
                tb_process.kill()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train behavior cloning model')
    parser.add_argument('--no-tensorboard', action='store_true', 
                       help='Disable automatic TensorBoard startup')
    args = parser.parse_args()
    
    train(start_tensorboard_auto=not args.no_tensorboard)