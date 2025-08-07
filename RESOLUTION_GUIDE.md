# Resolution Configuration Guide for FPS Gaming

## Current Settings (Optimized for FPS)
- **Resolution**: 480x320 (upgraded from 360x240)
- **FPS**: 60 (upgraded from 30)
- **Batch Size**: 16 (reduced from 32 due to higher resolution)

## Resolution Options

### 🟢 **480x320 (RECOMMENDED - Current)**
- **Downscale**: 4.0x3.4 from 1920x1080
- **Memory**: 0.4MB/frame, 6.6MB/sequence
- **Pros**: Good detail for FPS, reasonable training speed
- **Cons**: Slightly slower training than 360x240
- **Best for**: Most FPS games, good balance

### 🟡 **360x240 (FAST)**
- **Downscale**: 5.3x4.5 from 1920x1080
- **Memory**: 0.2MB/frame, 3.7MB/sequence
- **Pros**: Fastest training, lowest memory usage
- **Cons**: Limited detail, poor for precise aiming
- **Best for**: Quick prototyping, simple games

### 🔴 **640x480 (DETAILED)**
- **Downscale**: 3.0x2.2 from 1920x1080
- **Memory**: 0.9MB/frame, 13.2MB/sequence
- **Pros**: Maximum detail, excellent for precise aiming
- **Cons**: Slow training, high memory usage
- **Best for**: Competitive FPS, maximum accuracy

### 🔵 **720x480 (HD)**
- **Downscale**: 2.7x2.2 from 1920x1080
- **Memory**: 1.0MB/frame, 14.8MB/sequence
- **Pros**: HD quality, best visual detail
- **Cons**: Very slow training, highest memory usage
- **Best for**: Research, maximum quality

## How to Change Resolution

### Option 1: Edit config.py
```python
# In config.py, change these lines:
IMG_WIDTH = 480   # Change to desired width
IMG_HEIGHT = 320  # Change to desired height
```

### Option 2: Quick Resolution Switcher
Create different config files:
- `config_fast.py` (360x240)
- `config_balanced.py` (480x320) - current
- `config_detailed.py` (640x480)
- `config_hd.py` (720x480)

## FPS Gaming Considerations

### **Aiming Precision**
- **360x240**: Poor - targets become pixelated
- **480x320**: Good - adequate for most FPS games
- **640x480**: Excellent - precise aiming possible
- **720x480**: Best - maximum precision

### **Training Speed**
- **360x240**: Fastest (baseline)
- **480x320**: ~2x slower (current)
- **640x480**: ~4x slower
- **720x480**: ~5x slower

### **Memory Usage**
- **360x240**: 3.7MB/sequence
- **480x320**: 6.6MB/sequence (current)
- **640x480**: 13.2MB/sequence
- **720x480**: 14.8MB/sequence

## Recommendations by Game Type

### **Fast-Paced FPS (Call of Duty, Battlefield)**
- **Resolution**: 480x320 (current)
- **FPS**: 60
- **Reason**: Good balance of speed and detail

### **Tactical FPS (Counter-Strike, Valorant)**
- **Resolution**: 640x480
- **FPS**: 60
- **Reason**: Precision aiming is critical

### **Simple Games (Minecraft, Roblox)**
- **Resolution**: 360x240
- **FPS**: 30
- **Reason**: Less visual complexity

### **Research/Development**
- **Resolution**: 720x480
- **FPS**: 30
- **Reason**: Maximum quality for analysis

## Performance Impact

### **Training Time Estimates**
- **360x240**: 1 hour (baseline)
- **480x320**: 2 hours (current)
- **640x480**: 4 hours
- **720x480**: 5 hours

### **GPU Memory Requirements**
- **360x240**: 4GB VRAM minimum
- **480x320**: 6GB VRAM minimum (current)
- **640x480**: 8GB VRAM minimum
- **720x480**: 10GB VRAM minimum

## Quick Test Commands

Test different resolutions quickly:
```bash
# Test current resolution
python check_resolution.py

# Compare performance
python -c "import time; import cv2; import numpy as np; 
start=time.time(); 
img=np.random.randint(0,255,(320,480,3),dtype=np.uint8); 
cv2.resize(img,(360,240)); 
print(f'360x240: {time.time()-start:.3f}s'); 
start=time.time(); 
cv2.resize(img,(480,320)); 
print(f'480x320: {time.time()-start:.3f}s')"
``` 