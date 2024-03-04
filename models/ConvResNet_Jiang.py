import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x1)
        x = self.conv3(x) + x1  # residual connection
        return x


class ConvResNet(nn.Module):
    def __init__(self, num_attr: int = 5, input_channels=1, output_channels=1):
        super(ConvResNet, self).__init__()

        self.input_channels = input_channels
        self.conv1 = nn.Sequential( 
            nn.Conv2d(self.input_channels, 64, 3, padding="same"), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding="same"), nn.BatchNorm2d(64), nn.ReLU()
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
            nn.Linear(64, output_channels),
        )

    def forward(self, x, x_attrs):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mp1(x)
        x = self.resblocks1(x)
        x = self.mp2(x)
        x = self.resblocks2(x)  # B C H W
        x = torch.mean(x, dim=(2, 3))  # B C, Global Average pooling the H and W

        x = torch.hstack(
            [x, x_attrs]
        )  # batch_size x 256+6, Stack along the Channel dim
        x = self.mlp(x)

        return x

class ConvResNet_dropout(ConvResNet):
    def __init__(self, num_attr: int = 5, input_channels=1, output_channels=1):
        super(ConvResNet_dropout, self).__init__(num_attr, input_channels, output_channels)

        self.mlp = nn.Sequential(
            nn.Linear(256 + num_attr, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, output_channels),
        )

class ConvResNet_inputCdropout(ConvResNet):
    def __init__(self, num_attr: int = 5, input_channels=1, output_channels=1):
        super().__init__(num_attr, input_channels, output_channels)

        self.conv1 = nn.Sequential( 
            nn.Dropout2d(p=.3),  ## Drop outs the input_channels with 0.3 probability
            nn.Conv2d(self.input_channels, 64, 3, padding="same"), 
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )