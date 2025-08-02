#!/usr/bin/env python3
"""
Data recording script for AI Game Automation
Records screen frames and user input for training
"""

import os
import time

import cv2
import mss
import numpy as np
from pynput import keyboard, mouse

from config import (
    ACTIONS_FILE,
    COMMON_KEYS,
    DATA_DIR,
    FRAME_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    RECORDING_FPS,
)


class DataRecorder:
    """Records screen frames and user input."""

    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        self.frames = []
        self.actions = []
        self.recording = False
        self.frame_count = 0

        # Initialize input listeners
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press, on_release=self.on_key_release
        )
        self.mouse_listener = mouse.Listener(
            on_click=self.on_mouse_click, on_move=self.on_mouse_move
        )

    def on_key_press(self, key):
        """Handle key press events."""
        try:
            key_char = key.char
            if key_char in COMMON_KEYS:
                self.actions.append(
                    {
                        "type": "key_press",
                        "key": key_char,
                        "frame": self.frame_count,
                    }
                )
        except AttributeError:
            pass

    def on_key_release(self, key):
        """Handle key release events."""
        try:
            key_char = key.char
            if key_char in COMMON_KEYS:
                self.actions.append(
                    {
                        "type": "key_release",
                        "key": key_char,
                        "frame": self.frame_count,
                    }
                )
        except AttributeError:
            pass

    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click events."""
        self.actions.append(
            {
                "type": "mouse_click",
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed,
                "frame": self.frame_count,
            }
        )

    def on_mouse_move(self, x, y):
        """Handle mouse movement events."""
        self.actions.append(
            {"type": "mouse_move", "x": x, "y": y, "frame": self.frame_count}
        )

    def capture_frame(self):
        """Capture a single screen frame."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img

    def start_recording(self):
        """Start recording screen and input."""
        print("🎬 Starting recording...")
        print("Press F2 to stop recording")

        # Start input listeners
        self.keyboard_listener.start()
        self.mouse_listener.start()

        self.recording = True
        frame_interval = 1.0 / RECORDING_FPS
        last_frame_time = time.time()

        while self.recording:
            current_time = time.time()

            if current_time - last_frame_time >= frame_interval:
                # Capture frame
                frame = self.capture_frame()
                self.frames.append(frame)
                self.frame_count += 1
                last_frame_time = current_time

                # Check for stop key
                if keyboard.Key.f2 in keyboard._listener._pressed:
                    self.stop_recording()
                    break

    def stop_recording(self):
        """Stop recording and save data."""
        print("\n⏹️  Stopping recording...")
        self.recording = False

        # Stop listeners
        self.keyboard_listener.stop()
        self.mouse_listener.stop()

        # Save frames
        print(f"💾 Saving {len(self.frames)} frames...")
        for i, frame in enumerate(self.frames):
            frame_path = os.path.join(FRAME_DIR, f"frame_{i:06d}.png")
            cv2.imwrite(frame_path, frame)

        # Save actions
        print(f"💾 Saving {len(self.actions)} actions...")
        np.save(ACTIONS_FILE, self.actions)

        print(f"✅ Recording saved to {DATA_DIR}/")
        print(f"📊 Frames: {len(self.frames)}")
        print(f"📊 Actions: {len(self.actions)}")


def main():
    """Main recording function."""
    print("🎬 AI Game Automation - Data Recording")
    print("=" * 40)
    print("This will record your screen and input for training.")
    print("Press F2 to stop recording when done.")
    print()

    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FRAME_DIR, exist_ok=True)

    # Start recording
    recorder = DataRecorder()
    recorder.start_recording()


if __name__ == "__main__":
    main()
