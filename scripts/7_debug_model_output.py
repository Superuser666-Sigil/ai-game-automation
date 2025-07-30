# 7_debug_model_output.py - Model Output Analysis
import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import time
from collections import deque
from torchvision import transforms

# === CONFIG ===
with mss.mss() as sct: 
    monitor = sct.monitors[1]
SCREEN_WIDTH, SCREEN_HEIGHT = monitor["width"], monitor["height"]
MODEL_PATH = "model_rnn.pt"
IMG_WIDTH, IMG_HEIGHT = 960, 540
SEQUENCE_LENGTH = 5

COMMON_KEYS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
    'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '0', 'space', 'shift', 'ctrl', 'alt', 'tab', 'enter', 'backspace',
    'up', 'down', 'left', 'right', 'f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
    'f10', 'f11', 'f12', '-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/'
]

class BehaviorCloningCNNRNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        # Match training architecture exactly
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size, 
            hidden_size=512, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        
        # Separate heads for different outputs
        self.key_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(COMMON_KEYS)),
            nn.Sigmoid()
        )
        
        self.mouse_pos_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        
        self.mouse_click_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
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

# === Load Model ===
output_dim = len(COMMON_KEYS) + 4
model = BehaviorCloningCNNRNN(output_dim)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Debug Analysis ===
print("\n🔍 Starting Model Output Analysis...")
print("=" * 50)

# Initialize frame sequence
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
print("📸 Capturing frames for analysis...")
for i in range(SEQUENCE_LENGTH):
    frame = capture_and_process_frame()
    frame_sequence.append(frame)
    print(f"  Frame {i+1}/{SEQUENCE_LENGTH} captured")

# Run inference and analyze outputs
print("\n🤖 Running inference...")
input_tensor = torch.stack(list(frame_sequence)).unsqueeze(0)

with torch.no_grad():
    output_sequence = model(input_tensor)
    last_output = output_sequence[:, -1, :].squeeze()

raw_output = last_output.detach().cpu().numpy()

# Extract predictions
key_preds = raw_output[:len(COMMON_KEYS)]
mouse_x = raw_output[len(COMMON_KEYS)]
mouse_y = raw_output[len(COMMON_KEYS) + 1]
left_click = raw_output[len(COMMON_KEYS) + 2]
right_click = raw_output[len(COMMON_KEYS) + 3]

print(f"\n📊 MODEL OUTPUT ANALYSIS:")
print(f"Mouse Position: X={mouse_x:.4f}, Y={mouse_y:.4f}")
print(f"Mouse Clicks: Left={left_click:.4f}, Right={right_click:.4f}")

print(f"\n🔑 KEY PRESS ANALYSIS:")
print(f"Total keys: {len(COMMON_KEYS)}")
print(f"Key predictions range: [{key_preds.min():.6f}, {key_preds.max():.6f}]")
print(f"Key predictions mean: {key_preds.mean():.6f}")
print(f"Key predictions std: {key_preds.std():.6f}")

# Show top key predictions
key_confidences = [(COMMON_KEYS[i], key_preds[i]) for i in range(len(COMMON_KEYS))]
key_confidences.sort(key=lambda x: x[1], reverse=True)

print(f"\n🏆 TOP 10 KEY PREDICTIONS:")
for i, (key, conf) in enumerate(key_confidences[:10]):
    print(f"  {i+1:2d}. {key:8s}: {conf:.6f}")

print(f"\n🔍 THRESHOLD ANALYSIS:")
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
for threshold in thresholds:
    active_keys = [key for key, conf in key_confidences if conf > threshold]
    print(f"  Threshold {threshold:.1f}: {len(active_keys)} keys active")

# Check for model bias
print(f"\n⚠️  MODEL BIAS ANALYSIS:")
if key_preds.mean() < 0.01:
    print("  ❌ CRITICAL: Model is heavily biased against key presses!")
    print("  💡 This suggests training issues with class imbalance or loss function")
elif key_preds.mean() < 0.1:
    print("  ⚠️  WARNING: Model shows bias against key presses")
    print("  💡 Consider lowering key threshold or retraining with balanced data")
else:
    print("  ✅ Model shows reasonable key press distribution")

if key_preds.max() < 0.3:
    print("  ❌ CRITICAL: No key reaches 0.3 threshold!")
    print("  💡 Lower the KEY_THRESHOLD in inference script")

print(f"\n🎯 RECOMMENDATIONS:")
print("1. If key predictions are all < 0.3, lower KEY_THRESHOLD to 0.1-0.2")
print("2. If model is biased against keys, retrain with balanced loss function")
print("3. Check training data for proper key press distribution")
print("4. Consider using a simpler model architecture for initial testing")

print("\n" + "=" * 50)