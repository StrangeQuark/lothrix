import torch
import numpy as np
import pyautogui
import time
import os
import cv2
import pickle

class TaskConditionedModel(torch.nn.Module):
    def __init__(self, num_tasks, input_height, input_width):
        super(TaskConditionedModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_height, input_width)
            conv_output = self.conv(dummy_input)
            self.flat_size = conv_output.view(1, -1).size(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.flat_size + num_tasks, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x, task_id):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, task_id], dim=1)
        return self.fc(x)

def detect_state(screen):
    # Placeholder: Adjust these for your OSRS layout
    last_slot = screen[550:560, 700:710]
    if np.mean(last_slot) > 100:
        return "full_inventory"
    tree_region = screen[100:300, 200:400]
    brown_pixels = cv2.inRange(tree_region, (50, 30, 0), (100, 60, 20)).sum()
    if brown_pixels < 5000:
        return "no_trees"
    return "normal"

def run_reset():
    reset_file = "data/reset.pkl"
    if not os.path.exists(reset_file):
        print("Reset task not recorded! Please record a reset task first.")
        return
    
    try:
        with open(reset_file, "rb") as f:
            reset_data = pickle.load(f)
        print("Executing reset sequence...")
        for step in reset_data:
            action = step["action"]
            if action["type"] == "click":
                pyautogui.click(action["x"], action["y"])
            elif action["type"] == "press":
                pyautogui.press(action["key"])
            time.sleep(1)
        print("Reset complete.")
    except Exception as e:
        print(f"Failed to execute reset: {e}")

def run_task(task_name):
    if not os.path.exists("task_ids.pkl") or not os.path.exists("models/task_conditioned_model.pth"):
        print("Model or task IDs not found!")
        return
    
    with open("task_ids.pkl", "rb") as f:
        task_id_map = pickle.load(f)
    
    if task_name not in task_id_map:
        print(f"Task '{task_name}' not trained!")
        return

    target_size = (800, 600)
    screen_width, screen_height = pyautogui.size()  # Get screen dimensions
    
    model = TaskConditionedModel(num_tasks=len(task_id_map), input_height=target_size[1], input_width=target_size[0])
    model.load_state_dict(torch.load("models/task_conditioned_model.pth"))
    model.eval()

    task_id = torch.zeros(len(task_id_map), dtype=torch.float32)
    task_id[task_id_map[task_name]] = 1.0
    task_id = task_id.unsqueeze(0)

    stuck_counter = 0
    print(f"Running '{task_name}' (full screen). Press Ctrl+C to stop.")
    try:
        while True:
            screen = np.array(pyautogui.screenshot())
            if screen.shape[-1] == 4:
                screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
            state = detect_state(screen)

            if state == "full_inventory" and "bank" in task_name.lower():
                print("Full inventory detected, continuing to bank...")
            elif state == "no_trees":
                print("No trees detected!")
                stuck_counter += 1
                if stuck_counter >= 5:
                    run_reset()
                    stuck_counter = 0
            else:
                stuck_counter = max(0, stuck_counter - 1)

            input_state = cv2.resize(screen, target_size, interpolation=cv2.INTER_AREA)
            input_state = torch.tensor(input_state.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(input_state, task_id).numpy()[0]
            x, y = int(action[0] * screen_width), int(action[1] * screen_height)
            pyautogui.click(x, y)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")

if __name__ == "__main__":
    task = input("Enter task to run (e.g., 'cut yew trees and bank'): ").strip()
    run_task(task)