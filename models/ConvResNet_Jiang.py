import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, out_channels=None):
        if out_channels is None:
            out_channels = hidden_channels

        self.conv1 = nn.Sequential(
            nn.conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.conv2d(hidden_channels, hidden_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)
        x = self.conv3(x) + x1  # residual connection
        return x


class ConvResNet(nn.Module):
    def __init__(self, num_attr: int):
        self.conv1 = nn.Sequential(
            nn.conv2d(1, 64, 3, padding="same"), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.conv2d(1, 64, 3, padding="same"), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.mp1 = nn.MaxPool2d(2, 2)

        self.resblocks1 = nn.Sequential(
            ResBlock(64, 128, 3),
            ResBlock(128, 128, 3),
            ResBlock(128, 128, 3),
        )

        self.mp2 = nn.MaxPool2d(2, 2)

        self.resblocks2 = nn.Sequential(
            ResBlock(128, 256, 3),
            ResBlock(256, 256, 3),
            ResBlock(256, 256, 3),
        )

        self.mlp = nn.Sequential(
            nn.Linear(256 + num_attr, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, x_attrs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mp1(x)
        x = self.resblocks1(x)
        x = self.mp2(x)
        x = self.resblocks2(x)  # B C H W
        x = torch.mean(x, dim=(2, 3))  # B C, Global Average pooling the H and W

        x = torch.stack(
            [x, x_attrs], dim=-1
        )  # batch_size x 256+6, Stack along the Channel dim
        x = self.mlp(x)

        return x
