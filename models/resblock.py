import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size,  out_channels=None):
        super(ResBlock, self).__init__()
        if out_channels is None:
            out_channels = hidden_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(),
        # )

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1) + x1
        # x = self.conv3(x) + x1  # residual connection
        return x