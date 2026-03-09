import torch
import torch.nn as nn


class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, 7, padding=3),
            nn.ReLU(),

            nn.Conv2d(16, 8, 7, padding=3),
            nn.ReLU(),

            nn.Conv2d(8, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        return x
