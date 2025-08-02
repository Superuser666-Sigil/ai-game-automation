#!/usr/bin/env python3
"""
Data quality analysis script for AI Game Automation
Analyzes recorded data for training suitability
"""

import os

import numpy as np

from config import ACTIONS_FILE, FRAME_DIR


def load_data():
    """Load recorded frames and actions."""
    print("📊 Loading recorded data...")

    # Check if data exists
    if not os.path.exists(ACTIONS_FILE):
        print(f"❌ Actions file not found: {ACTIONS_FILE}")
        return None, None

    if not os.path.exists(FRAME_DIR):
        print(f"❌ Frames directory not found: {FRAME_DIR}")
        return None, None

    # Load actions
    actions = np.load(ACTIONS_FILE, allow_pickle=True)

    # Count frames
    frame_files = [f for f in os.listdir(FRAME_DIR) if f.endswith(".png")]
    frame_count = len(frame_files)

    print(f"✅ Loaded {len(actions)} actions and {frame_count} frames")
    return actions, frame_count


def analyze_key_presses(actions):
    """Analyze key press patterns."""
    print("\n⌨️  Key Press Analysis:")

    # Count key presses
    key_presses = [a for a in actions if a["type"] == "key_press"]
    key_releases = [a for a in actions if a["type"] == "key_release"]

    print(f"📊 Total key presses: {len(key_presses)}")
    print(f"📊 Total key releases: {len(key_releases)}")

    # Key frequency
    key_counts = {}
    for action in key_presses:
        key = action["key"]
        key_counts[key] = key_counts.get(key, 0) + 1

    print(f"📊 Unique keys pressed: {len(key_counts)}")

    # Show most common keys
    if key_counts:
        sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
        print("🔝 Most common keys:")
        for key, count in sorted_keys[:5]:
            print(f"   {key}: {count} times")

    # Key press rate
    if len(actions) > 0:
        key_press_rate = len(key_presses) / len(actions) * 100
        print(f"📊 Key press rate: {key_press_rate:.1f}%")

        if key_press_rate < 5:
            print("⚠️  Low key press rate - consider recording more active " "gameplay")
        elif key_press_rate > 50:
            print(
                "⚠️  Very high key press rate - may have too many accidental " "presses"
            )
        else:
            print("✅ Good key press rate")


def analyze_mouse_actions(actions):
    """Analyze mouse action patterns."""
    print("\n🖱️  Mouse Action Analysis:")

    # Count mouse actions
    mouse_clicks = [a for a in actions if a["type"] == "mouse_click"]
    mouse_moves = [a for a in actions if a["type"] == "mouse_move"]

    print(f"📊 Total mouse clicks: {len(mouse_clicks)}")
    print(f"📊 Total mouse moves: {len(mouse_moves)}")

    # Click analysis
    if mouse_clicks:
        left_clicks = [c for c in mouse_clicks if "left" in str(c["button"])]
        right_clicks = [c for c in mouse_clicks if "right" in str(c["button"])]

        print(f"📊 Left clicks: {len(left_clicks)}")
        print(f"📊 Right clicks: {len(right_clicks)}")

    # Mouse movement coverage
    if mouse_moves:
        x_coords = [m["x"] for m in mouse_moves]
        y_coords = [m["y"] for m in mouse_moves]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        print(f"📊 Mouse X range: {x_range:.0f} pixels")
        print(f"📊 Mouse Y range: {y_range:.0f} pixels")


def analyze_data_quality(actions, frame_count):
    """Overall data quality assessment."""
    print("\n📈 Data Quality Assessment:")

    # Action density
    if frame_count > 0:
        actions_per_frame = len(actions) / frame_count
        print(f"📊 Actions per frame: {actions_per_frame:.2f}")

        if actions_per_frame < 0.1:
            print("⚠️  Very low action density - consider more active gameplay")
        elif actions_per_frame > 10:
            print("⚠️  Very high action density - may be too noisy")
        else:
            print("✅ Good action density")

    # Data balance
    key_actions = [a for a in actions if "key" in a["type"]]
    mouse_actions = [a for a in actions if "mouse" in a["type"]]

    total_actions = len(actions)
    if total_actions > 0:
        key_ratio = len(key_actions) / total_actions
        mouse_ratio = len(mouse_actions) / total_actions

        print(f"📊 Key actions: {key_ratio:.1%}")
        print(f"📊 Mouse actions: {mouse_ratio:.1%}")

        if key_ratio < 0.1:
            print("⚠️  Very few key actions - may not learn keyboard controls well")
        elif mouse_ratio < 0.1:
            print("⚠️  Very few mouse actions - may not learn mouse controls well")
        else:
            print("✅ Good balance of key and mouse actions")

    # Overall assessment
    print("\n🎯 Overall Assessment:")
    if len(actions) < 100:
        print("❌ Very little data - record at least 2-3 minutes of gameplay")
    elif len(actions) < 500:
        print("⚠️  Limited data - consider recording more gameplay")
    else:
        print("✅ Sufficient data for training")

    if frame_count < 100:
        print("❌ Very few frames - record longer gameplay sessions")
    elif frame_count < 500:
        print("⚠️  Limited frames - consider longer recording sessions")
    else:
        print("✅ Sufficient frames for training")


def main():
    """Main analysis function."""
    print("📊 AI Game Automation - Data Quality Analysis")
    print("=" * 50)

    # Load data
    actions, frame_count = load_data()
    if actions is None:
        print("\n❌ No data found. Please record some gameplay first:")
        print("   python scripts/3_record_data.py")
        return

    # Analyze data
    analyze_key_presses(actions)
    analyze_mouse_actions(actions)
    analyze_data_quality(actions, frame_count)

    print("\n✅ Analysis complete!")
    print("🎉 If data looks good, proceed to training:")
    print("   python scripts/5_train_model.py")


if __name__ == "__main__":
    main()
