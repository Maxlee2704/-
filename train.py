import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


data = np.load("D:/AI Challenge/Emotion-cls/x_train.npy")
data_tensor = torch.tensor(data, dtype=torch.long)

label = np.load("D:/AI Challenge/Emotion-cls/y_train.npy")
labels_tensor = torch.tensor(label, dtype=torch.long)
print(label.shape)

dataset = CustomDataset(data_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for inputs, labels in dataloader:
    print(inputs.shape, labels.shape)
    break