import torch
import torch.nn as nn

class CrowdCNN(nn.Module):
    def __init__(self):
        super(CrowdCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=9, padding=4),   # features.0
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=7, padding=3),  # features.3
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=7, padding=3),  # features.6
            nn.ReLU(),

            nn.Conv2d(16, 8, kernel_size=7, padding=3),   # features.8
            nn.ReLU(),

            nn.Conv2d(8, 1, kernel_size=1)                # features.10
        )

    def forward(self, x):
        return self.features(x)