import torch.nn as nn
import torch
from dcn import DeformableConv2d

class EmotionModel(nn.Module):
    def __init__(self):
        super.__init__(EmotionModel)
        self.dcn1 = DeformableConv2d(3,32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dcn2 = DeformableConv2d(32,64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Softmax(128, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x