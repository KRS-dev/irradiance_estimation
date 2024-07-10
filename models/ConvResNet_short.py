import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.resblock import ResBlock

class ConvResNet_short(nn.Module):
    def __init__(self, num_attr: int = 5, input_channels=1, output_channels=1):
        super(ConvResNet_short, self).__init__()

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
        )

        self.mp2 = nn.MaxPool2d(2, 2)

        self.resblocks2 = nn.Sequential(
            ResBlock(128, 256, 3),
        )

        self.mlp = nn.Sequential(
            nn.Linear(256 + num_attr, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_channels),
        )

        # self.relu_rect = nn.ReLU()


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
        
        # x = self.relu_rect(x + 1) - 1
        return x


class ConvResNet_short_relu(nn.Module):
    def __init__(self, num_attr: int = 5, input_channels=1, output_channels=1):
        super(ConvResNet_short_relu, self).__init__()

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
        )

        self.mp2 = nn.MaxPool2d(2, 2)

        self.resblocks2 = nn.Sequential(
            ResBlock(128, 256, 3),
        )

        self.mlp = nn.Sequential(
            nn.Linear(256 + num_attr, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_channels),
        )

        self.relu_rect = nn.ReLU()


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
        
        x = self.relu_rect(x + 1) - 1
        return x

