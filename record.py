from pynput import mouse, keyboard
import cv2
import numpy as np
import pyautogui
import time
import pickle
import os

def record_task():
    """Record gameplay actions for an OSRS task and save them to a file."""
    task_name = input("Enter the task (e.g., 'cut yew trees and bank' or 'reset'): ").strip()
    dataset = []
    recording = False
    task_id_map = {}
    
    if os.path.exists("task_ids.pkl"):
        try:
            with open("task_ids.pkl", "rb") as f:
                task_id_map = pickle.load(f)
        except Exception as e:
            print(f"Failed to load task_ids.pkl: {e}")
    
    if task_name not in task_id_map and task_name != "reset":
        task_id_map[task_name] = len(task_id_map)

    def on_click(x, y, button, pressed):
        nonlocal recording
        if recording and pressed:
            try:
                screen = np.array(pyautogui.screenshot())
                if screen.shape[-1] == 4:
                    screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
                dataset.append({"state": screen, "action": {"x": x, "y": y, "type": "click"}, "task": task_name})
                print(f"Captured click at ({x}, {y})")
            except Exception as e:
                print(f"Capture failed: {e}")

    def on_press(key):
        nonlocal recording
        if recording:
            try:
                screen = np.array(pyautogui.screenshot())
                if screen.shape[-1] == 4:
                    screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
                dataset.append({"state": screen, "action": {"key": str(key), "type": "press"}, "task": task_name})
                print(f"Captured key press: {key}")
            except Exception as e:
                print(f"Capture failed: {e}")

    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener.start()
    keyboard_listener.start()

    print(f"Press Enter to start recording '{task_name}' (full screen). Press Ctrl+C to stop.")
    input()
    recording = True
    print(f"Recording '{task_name}'... Play OSRS now!")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        recording = False
        mouse_listener.stop()
        keyboard_listener.stop()
        filename = f"data/{task_name.replace(' ', '_')}.pkl"
        os.makedirs("data", exist_ok=True)
        try:
            with open(filename, "wb") as f:
                pickle.dump(dataset, f)
            if task_name != "reset":
                with open("task_ids.pkl", "wb") as f:
                    pickle.dump(task_id_map, f)
            print(f"Saved {len(dataset)} actions to {filename}")
        except Exception as e:
            print(f"Failed to save data: {e}")

if __name__ == "__main__":
    record_task()