import torch
import torch.nn as nn


class MarioNet(nn.Module):
    def __init__(self):
        super().__init__()

    def _build_cnn_layers(self, input_dim, output_dim):
        """Construct the convolutional layers"""
        self.conv1 = nn.Conv2d(
            in_channels=input_dim, out_channels=32, kernel_size=8, stride=4
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        return nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.conv3,
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
