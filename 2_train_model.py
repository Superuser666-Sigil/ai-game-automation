import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, utils as tv_utils
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import subprocess
import socket
import random
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import *
from config import MODEL_SAVE_PATH_TEMPLATE

# === EARLY STOPPING ===
class EarlyStopping:
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

# === DATASET WITH OVERSAMPLING ===
class WoWSequenceDataset(Dataset):
    def __init__(self, frame_dir, actions_file, sequence_length, transform=None):
        self.transform = transform
        self.sequence_length = sequence_length

        frame_paths = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir) if f.endswith(".jpg")
        ])
        actions = np.load(actions_file).astype(np.float32)

        # align lengths
        min_len = min(len(frame_paths), len(actions))
        self.frame_paths = frame_paths[:min_len]
        self.actions     = actions[:min_len]

        # oversample sequences with any key-press or click
        self.indices = []
        num_keys = len(COMMON_KEYS)
        action_frames = 0

        for i in range(len(self.frame_paths) - self.sequence_length + 1):
            last = self.actions[i + self.sequence_length - 1]
            key_press   = np.sum(last[:num_keys]) > 0
            mouse_click = np.sum(last[num_keys+2:]) > 0

            if key_press or mouse_click:
                self.indices.extend([i] * OVERSAMPLE_ACTION_FRAMES_MULTIPLIER)
                action_frames += 1
            else:
                self.indices.append(i)

        print(f"Loaded {len(self.frame_paths)} frames, "
              f"{action_frames} action sequences, "
              f"{len(self.indices)} total sequences.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.sequence_length

        imgs = []
        for fi in range(start, end):
            img = cv2.imread(self.frame_paths[fi])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        seq_actions = self.actions[start:end]
        return torch.stack(imgs), torch.tensor(seq_actions, dtype=torch.float32)

# === MODEL & HELPERS ===
class ImprovedBehaviorCloningCNNRNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_out_size = self.cnn(dummy).shape[1]

        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.key_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, len(COMMON_KEYS)), nn.Sigmoid()
        )
        self.mouse_pos_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid()
        )
        self.mouse_click_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Sigmoid()
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feat = self.cnn(x)
        feat = feat.view(b, s, -1)
        out, _ = self.lstm(feat)
        flat = out.reshape(b * s, -1)

        k = self.key_head(flat)
        p = self.mouse_pos_head(flat)
        c = self.mouse_click_head(flat)
        concat = torch.cat([k, p, c], dim=1)
        return concat.view(b, s, -1)

def bce_loss(outputs, targets):
    return nn.functional.binary_cross_entropy(outputs, targets)

# === VALIDATION WITH CONFUSION MATRIX LOGGING ===
def validate(model, dataloader, device, writer, epoch):
    model.eval()
    all_key_out, all_click_out = [], []
    all_key_tgt, all_click_tgt = [], []

    with torch.no_grad():
        for seqs, acts in dataloader:
            seqs, acts = seqs.to(device), acts.to(device)
            out = model(seqs)  # [B, S, D]

            # last VALIDATION_WINDOW frames
            k_out = out[:, -VALIDATION_WINDOW:, :len(COMMON_KEYS)].mean(dim=1)
            k_tgt = acts[:, -VALIDATION_WINDOW:, :len(COMMON_KEYS)].max(dim=1)[0]

            start = len(COMMON_KEYS) + 2
            c_out = out[:, -VALIDATION_WINDOW:, start:start+2].mean(dim=1)
            c_tgt = acts[:, -VALIDATION_WINDOW:, start:start+2].max(dim=1)[0]

            all_key_out.append(k_out.cpu().numpy())
            all_key_tgt.append(k_tgt.cpu().numpy())
            all_click_out.append(c_out.cpu().numpy())
            all_click_tgt.append(c_tgt.cpu().numpy())

    all_key_out   = np.vstack(all_key_out)
    all_key_tgt   = np.vstack(all_key_tgt)
    all_click_out = np.vstack(all_click_out)
    all_click_tgt = np.vstack(all_click_tgt)

    best = {"threshold": None, "f1": 0.0}
    for thresh in THRESHOLD_SWEEP:
        kp = (all_key_out > thresh).astype(int)
        cp = (all_click_out > thresh).astype(int)
        preds = np.hstack([kp, cp])
        tgts  = np.hstack([all_key_tgt, all_click_tgt])

        pr, rc, f1, _ = precision_recall_fscore_support(
            tgts, preds, average="samples", zero_division=0
        )
        print(f" thresh={thresh:.2f}  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")
        if f1 > best["f1"]:
            best["threshold"], best["f1"] = thresh, f1

    print(f"\nBest @ thresh={best['threshold']}: F1={best['f1']:.3f}\n")

    # Log confusion matrix figure
    cm = confusion_matrix(
        np.hstack([all_key_tgt, all_click_tgt]).flatten(),
        np.hstack([(all_key_out>best["threshold"]).astype(int),
                   (all_click_out>best["threshold"]).astype(int)]).flatten()
    )
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    writer.add_figure("ConfusionMatrix", fig, epoch)
    plt.close(fig)

    return best["f1"]

# === TENSORBOARD PORT UTILITIES ===
def find_free_port(start=6006, end=6099):
    ports = list(range(start, end+1))
    random.shuffle(ports)
    for port in ports:
        with socket.socket() as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None

def kill_tensorboard_on_port(port):
    for proc in psutil.process_iter(['pid','name','cmdline']):
        if 'tensorboard' in proc.info['name'] and str(port) in ' '.join(proc.info['cmdline']):
            proc.kill()
            print(f"Killed TensorBoard on port {port}")

# === TRAINING LOOP ===
def train():
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    early_stopper = EarlyStopping()
    
    # Method 3: Stop-file sentinel for remote control
    stop_file = Path("STOP_TRAINING")

    # launch TensorBoard
    port = find_free_port()
    if port:
        kill_tensorboard_on_port(port)
        subprocess.Popen([
            "tensorboard",
            "--logdir", TENSORBOARD_LOG_DIR,
            "--port", str(port)
        ])
        print(f"TensorBoard on http://localhost:{port}")
    else:
        print("No free port for TensorBoard.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # load datasets
    datasets = []
    for d in DATA_DIRS:
        fdir = os.path.join(d, "frames")
        af   = os.path.join(d, ACTIONS_FILE)
        if os.path.exists(fdir) and os.path.exists(af):
            ds = WoWSequenceDataset(fdir, af, SEQUENCE_LENGTH, transform)
            if len(ds):
                datasets.append(ds)
    if not datasets:
        print("No valid datasets found!")
        return

    full = ConcatDataset(datasets)
    vsize = int(len(full) * VALIDATION_SPLIT)
    tsize = len(full) - vsize
    train_ds, val_ds = random_split(full, [tsize, vsize])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ImprovedBehaviorCloningCNNRNN(len(COMMON_KEYS)+4).to(device)
    opt    = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    sched  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                 factor=0.5, patience=2)

    best_f1 = 0.0
    print(f"Starting training on {device} for {EPOCHS} epochs…")

    # Method 2: KeyboardInterrupt guard
    try:
        for epoch in range(1, EPOCHS+1):
            # Method 3: Check for stop file at start of each epoch
            if stop_file.exists():
                print("🛑 Found STOP_TRAINING file – finishing current epoch then terminating...")
                break
                
            model.train()
            running_loss = 0.0

        for i, (seqs, acts) in enumerate(train_loader, 1):
            seqs, acts = seqs.to(device), acts.to(device)
            opt.zero_grad()
            out  = model(seqs)
            loss = bce_loss(out, acts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss.item()

            if i % 100 == 0:
                step = (epoch-1)*len(train_loader) + i
                writer.add_scalar("Loss/train_batch", loss.item(), step)
                print(f" Epoch {epoch}/{EPOCHS} — Step {i}/{len(train_loader)} — Loss {loss:.4f}")

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
        print(f"Epoch {epoch} done — Avg Loss {avg_loss:.4f}")

        # Log weight & gradient histograms
        for name, param in model.named_parameters():
            writer.add_histogram(f"weights/{name}", param, epoch)
            if param.grad is not None:
                writer.add_histogram(f"grads/{name}", param.grad, epoch)

        # Log one input sequence as an image grid
        grid = tv_utils.make_grid(seqs[0], nrow=seqs.size(1), normalize=True)
        writer.add_image("input_sequence", grid, epoch)

        # validate & log confusion matrix
        f1 = validate(model, val_loader, device, writer, epoch)
        writer.add_scalar("F1/validation", f1, epoch)
        writer.add_scalar("LR", opt.param_groups[0]["lr"], epoch)
        sched.step(f1)

        early_stopper(f1)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        # checkpoints
        ckpt = MODEL_SAVE_PATH_TEMPLATE.format(epoch)
        torch.save(model.state_dict(), ckpt)
        print(f"Saved checkpoint: {ckpt}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"New best model: {MODEL_FILE} (F1={f1:.3f})")

    # Method 2: Handle KeyboardInterrupt gracefully
    except KeyboardInterrupt:
        print("⏹ Caught CTRL-C – finishing current epoch then exiting...")
        print("✅ Training stopped gracefully. Checkpoints saved.")
    
    finally:
        writer.close()
        print("📊 TensorBoard logs saved. Training complete!")

if __name__ == "__main__":
    train()
