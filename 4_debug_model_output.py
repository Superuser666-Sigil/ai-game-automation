# 4_debug_model_output.py - UPDATED with correct model architecture
import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
from collections import deque
from torchvision import transforms
from config import (
    MODEL_FILE, COMMON_KEYS, KEY_THRESHOLD,
    IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH
)  # Import specific settings from the config file

# === CONFIG (Loaded from config.py) ===
# Ensure this path points to the model you want to debug
MODEL_PATH = MODEL_FILE  # Final trained model is in root directory

# --- Use the CORRECT model class ---
class ImprovedBehaviorCloningCNNRNN(nn.Module):
    # This class definition now MATCHES the training script
    def __init__(self, output_dim):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,32,5,stride=2,padding=2),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,stride=2,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,128,3,stride=2,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d((6,6)),nn.Flatten())
        with torch.no_grad(): cnn_output_size=self.cnn(torch.zeros(1,3,IMG_HEIGHT,IMG_WIDTH)).shape[1]
        self.lstm=nn.LSTM(input_size=cnn_output_size,hidden_size=256,num_layers=2,batch_first=True,dropout=0.1)
        self.key_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,len(COMMON_KEYS)),nn.Sigmoid())
        self.mouse_pos_head=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,2),nn.Sigmoid())
        self.mouse_click_head=nn.Sequential(nn.Linear(256,32),nn.ReLU(),nn.Linear(32,2),nn.Sigmoid())
    def forward(self,x):
        b,s,c,h,w=x.shape; cnn_out=self.cnn(x.view(b*s,c,h,w)); lstm_in=cnn_out.view(b,s,-1); lstm_out,_=self.lstm(lstm_in); lstm_flat=lstm_out.reshape(b*s,-1); key_out=self.key_head(lstm_flat); mouse_pos_out=self.mouse_pos_head(lstm_flat); mouse_click_out=self.mouse_click_head(lstm_flat); return torch.cat([key_out,mouse_pos_out,mouse_click_out],dim=1).view(b,s,-1)

# --- The rest of the debug script is unchanged ---
# ... (all other functions and the main execution block remain the same) ...
def capture_and_process_frame():
    with mss.mss() as sct:
        monitor=sct.monitors[1]
        img=cv2.cvtColor(np.array(sct.grab(monitor)),cv2.COLOR_BGRA2RGB)
        return transform(img)
output_dim=len(COMMON_KEYS)+4
model=ImprovedBehaviorCloningCNNRNN(output_dim)
try:
    model.load_state_dict(torch.load(MODEL_PATH,map_location="cpu"))
    print(f"✅ Model '{MODEL_PATH}' loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}");exit(1)
model.eval()
transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
print("\n🔍 Starting Model Output Analysis...")
frame_sequence=deque(maxlen=SEQUENCE_LENGTH)
for i in range(SEQUENCE_LENGTH):
    frame_sequence.append(capture_and_process_frame())
    print(f"  Frame {i+1}/{SEQUENCE_LENGTH} captured")
input_tensor=torch.stack(list(frame_sequence)).unsqueeze(0)
with torch.no_grad():
    last_output=model(input_tensor)[:,-1,:].squeeze()
raw_output=last_output.detach().cpu().numpy()
key_preds=raw_output[:len(COMMON_KEYS)]
mouse_x,mouse_y=raw_output[len(COMMON_KEYS)],raw_output[len(COMMON_KEYS)+1]
left_click,right_click=raw_output[len(COMMON_KEYS)+2],raw_output[len(COMMON_KEYS)+3]
print(f"\n📊 MODEL OUTPUT ANALYSIS:")
print(f"Mouse Position: X={mouse_x:.4f}, Y={mouse_y:.4f}")
print(f"Mouse Clicks: Left={left_click:.4f}, Right={right_click:.4f}")
print(f"\n🔑 KEY PRESS ANALYSIS (Threshold: {KEY_THRESHOLD}):")
key_confidences=sorted([(COMMON_KEYS[i],key_preds[i])for i in range(len(COMMON_KEYS))],key=lambda x:x[1],reverse=True)
print(f"🏆 TOP 10 KEY PREDICTIONS:")
for i,(key,conf)in enumerate(key_confidences[:10]):
    active_marker="✅" if conf > KEY_THRESHOLD else "❌"
    print(f"  {i+1:2d}. {key:8s}: {conf:.4f} {active_marker}")