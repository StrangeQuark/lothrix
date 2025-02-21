import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import cv2

class GameplayDataset(Dataset):
    def __init__(self, data_files, task_id_map, target_size=(800, 600)):
        """Initialize dataset with downsampled images."""
        self.data = []
        self.task_id_map = task_id_map
        self.num_tasks = len(task_id_map)
        self.target_size = target_size
        
        for data_file in data_files:
            try:
                with open(data_file, "rb") as f:
                    self.data.extend(pickle.load(f))
            except Exception as e:
                print(f"Failed to load {data_file}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx]["state"]
        # Downsample image to target size
        state = cv2.resize(state, self.target_size, interpolation=cv2.INTER_AREA)
        state = state.transpose(2, 0, 1) / 255.0  # Normalize to [0, 1]
        action = self.data[idx]["action"]
        task = self.data[idx]["task"]
        
        if action["type"] == "click":
            # Scale coordinates to match downsampled size
            orig_height, orig_width = self.data[idx]["state"].shape[:2]
            label = torch.tensor([
                action["x"] * self.target_size[0] / orig_width,
                action["y"] * self.target_size[1] / orig_height
            ], dtype=torch.float32) / self.target_size[0]  # Normalize to [0, 1]
        else:
            label = torch.tensor([0.5, 0.5], dtype=torch.float32)
        
        task_id = torch.zeros(self.num_tasks, dtype=torch.float32)
        if task in self.task_id_map:
            task_id[self.task_id_map[task]] = 1.0
        return torch.tensor(state, dtype=torch.float32), task_id, label

class TaskConditionedModel(nn.Module):
    def __init__(self, num_tasks, input_height, input_width):
        super(TaskConditionedModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_height, input_width)
            conv_output = self.conv(dummy_input)
            self.flat_size = conv_output.view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size + num_tasks, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, task_id):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, task_id], dim=1)
        return self.fc(x)

def train_model():
    """Train the task-conditioned model on recorded gameplay data."""
    if not os.path.exists("task_ids.pkl"):
        print("No tasks recorded yet!")
        return
    
    with open("task_ids.pkl", "rb") as f:
        task_id_map = pickle.load(f)

    data_files = [f"data/{f}" for f in os.listdir("data") if f.endswith(".pkl") and f != "reset.pkl"]
    if not data_files:
        print("No training data found!")
        return

    dataset = GameplayDataset(data_files, task_id_map)
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TaskConditionedModel(num_tasks=len(task_id_map), input_height=dataset.target_size[1], input_width=dataset.target_size[0])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        for states, task_ids, actions in loader:
            optimizer.zero_grad()
            outputs = model(states, task_ids)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/task_conditioned_model.pth")
    print("Trained and saved single model.")

if __name__ == "__main__":
    train_model()