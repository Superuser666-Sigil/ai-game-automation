# 4_analyze_data_quality.py - Analyze Data Collection Quality
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_data_quality():
    """Analyze the quality of collected data"""
    
    data_dir = "data_human"
    actions_file = os.path.join(data_dir, "actions.npy")
    frames_dir = os.path.join(data_dir, "frames")
    
    if not os.path.exists(actions_file):
        print("❌ No actions.npy file found!")
        return False
    
    if not os.path.exists(frames_dir):
        print("❌ No frames directory found!")
        return False
    
    # Load actions
    actions = np.load(actions_file)
    print(f"✅ Loaded {len(actions)} action records")
    
    # Count frames
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    print(f"✅ Found {len(frame_files)} frame files")
    
    # Check data consistency
    if len(actions) != len(frame_files):
        print(f"⚠️  WARNING: Mismatch between actions ({len(actions)}) and frames ({len(frame_files)})")
        print(f"💡 This suggests multiple recording sessions. Using all available data.")
        
        # Use the smaller of the two to avoid index errors
        min_len = min(len(actions), len(frame_files))
        actions = actions[:min_len]
        print(f"📊 Using {min_len} frames for analysis")
    else:
        print("✅ Actions and frames count match")
    
    # Analyze key press data
    COMMON_KEYS: list[str] = [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6',
        '7', '8', '9', '0', 'space', 'shift', 'ctrl', 'alt', 'tab', 'enter', 'backspace',
        'up', 'down', 'left', 'right', 'f1', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9',
        'f10', 'f11', 'f12', '-', '=', '[', ']', '\\', ';', '\'', ',', '.', '/'
    ]
    
    key_actions = actions[:, :len(COMMON_KEYS)]
    mouse_actions = actions[:, len(COMMON_KEYS):len(COMMON_KEYS)+2]
    click_actions = actions[:, len(COMMON_KEYS)+2:]
    
    print(f"\n📊 DATA ANALYSIS:")
    print(f"Total frames: {len(actions)}")
    print(f"Recording duration: {len(actions) / 10:.1f} seconds ({len(actions) / 10 / 60:.1f} minutes)")
    
    # Key press analysis
    total_key_presses = np.sum(key_actions)
    frames_with_keys = np.sum(np.sum(key_actions, axis=1) > 0)  # Frames where any key was pressed
    key_press_rate = total_key_presses / (len(actions) * len(COMMON_KEYS))
    frame_key_rate = frames_with_keys / len(actions)  # Percentage of frames with any key press
    
    print(f"\n🔑 KEY PRESS ANALYSIS:")
    print(f"Total key presses: {total_key_presses}")
    print(f"Frames with key presses: {frames_with_keys} / {len(actions)} ({frame_key_rate*100:.1f}%)")
    print(f"Key press rate (all keys): {key_press_rate:.4f} ({key_press_rate*100:.2f}%)")
    
    if frame_key_rate < 0.05:  # Less than 5% of frames have key presses
        print("❌ CRITICAL: Very low key press rate! This will cause training issues.")
        print("💡 Make sure you're actually pressing keys during recording!")
        return False
    elif frame_key_rate < 0.1:  # Less than 10% of frames have key presses
        print("⚠️  WARNING: Low key press rate. Consider recording more active gameplay.")
    else:
        print("✅ Good key press rate detected")
    
    # Show most active keys
    key_totals = np.sum(key_actions, axis=0)
    active_keys: list[tuple[str, int]] = [(COMMON_KEYS[i], int(total)) for i, total in enumerate(key_totals) if total > 0]
    active_keys.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 MOST ACTIVE KEYS:")
    for i, (key, count) in enumerate(active_keys[:10]):
        print(f"  {i+1:2d}. {key:8s}: {count:4d} presses")
    
    # Mouse analysis
    print(f"\n🖱️ MOUSE ANALYSIS:")
    mouse_x_range = (mouse_actions[:, 0].min(), mouse_actions[:, 0].max())
    mouse_y_range = (mouse_actions[:, 1].min(), mouse_actions[:, 1].max())
    print(f"Mouse X range: [{mouse_x_range[0]:.3f}, {mouse_x_range[1]:.3f}]")
    print(f"Mouse Y range: [{mouse_y_range[0]:.3f}, {mouse_y_range[1]:.3f}]")
    
    # Click analysis
    left_clicks = np.sum(click_actions[:, 0])
    right_clicks = np.sum(click_actions[:, 1])
    print(f"Left clicks: {int(left_clicks)}")
    print(f"Right clicks: {int(right_clicks)}")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Key press timeline
    plt.subplot(2, 2, 1)
    key_activity = np.sum(key_actions, axis=1)
    plt.plot(key_activity)
    plt.title('Key Press Activity Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Keys Pressed')
    plt.grid(True)
    
    # Mouse movement
    plt.subplot(2, 2, 2)
    plt.scatter(mouse_actions[:, 0], mouse_actions[:, 1], alpha=0.5, s=1)
    plt.title('Mouse Movement Pattern')
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.grid(True)
    
    # Key press distribution
    plt.subplot(2, 2, 3)
    top_keys = active_keys[:15]
    keys, counts = zip(*top_keys)
    plt.bar(range(len(keys)), counts)
    plt.title('Top 15 Most Pressed Keys')
    plt.xlabel('Key')
    plt.ylabel('Press Count')
    plt.xticks(range(len(keys)), keys, rotation=45)
    
    # Click timeline
    plt.subplot(2, 2, 4)
    plt.plot(click_actions[:, 0], label='Left Click', alpha=0.7)
    plt.plot(click_actions[:, 1], label='Right Click', alpha=0.7)
    plt.title('Click Activity Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Click State')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('data_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Visualization saved as 'data_quality_analysis.png'")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if frame_key_rate < 0.1:
        print("1. Record more active gameplay with frequent key presses")
        print("2. Try recording shorter sessions (30-60 seconds) with intense gameplay")
        print("3. Make sure you're pressing different keys, not just one key")
        print("4. Consider recording a game that requires both keyboard and mouse")
    else:
        print("1. Data quality looks good for training!")
        print("2. Consider recording multiple sessions for better variety")
        print("3. Make sure to include mouse movements and clicks")
    
    return True

if __name__ == "__main__":
    print("🔍 Testing Data Collection Quality...")
    print("=" * 50)
    
    success = analyze_data_quality()
    
    if success:
        print("\n✅ Data collection test completed successfully!")
        print("💡 You can now proceed with training using 5_train_model.py")
    else:
        print("\n❌ Data collection issues detected!")
        print("💡 Please fix the issues and record new data before training")