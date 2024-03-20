import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class residual_block(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(residual_block, self).__init__()
        self.fcn1 = nn.Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.relu1 = nn.ReLU()

        self.fcn2 = nn.Linear(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x1 = self.fcn1(x)
        x1 = self.relu1(self.bn1(x1))
        x = x1 + x
        x = self.fcn2(x)
        x = self.relu2(self.bn2(x))
        return x



class FCN(nn.Module):
    def __init__(self, patch_size, input_channels, input_features, channel_size, output_channels) -> None:
        super(FCN).__init__()

        input_size = patch_size[0]*patch_size[1]*input_channels + input_features
        self.mlp = nn.Sequential(
            nn.Linear(input_size, channel_size),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(channel_size, channel_size),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(channel_size, channel_size),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Linear(channel_size, output_channels)
        )
    def forward(self, X, x_attrs):
        x = X.flatten()
        x = torch.cat([x, x_attrs])
        x = self.mlp(x)
        return x

class residual_FCN(nn.Module):
    def __init__(self, patch_size, input_channels, input_features, channel_size, output_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        input_size = patch_size[0]*patch_size[1]*input_channels + input_features

        self.mlp = nn.Sequential(
            nn.Linear(input_size, channel_size),
            nn.BatchNorm1d(channel_size),
            nn.ReLU(),
            )
        self.res1 = residual_block(channel_size, channel_size)
        self.res2 = residual_block(channel_size, channel_size)

        self.linear = nn.Linear(channel_size, output_channels)

    def forward(self, X, x_attrs):
        x = X.reshape(X.shape[0], -1)
        x = torch.cat([x, x_attrs], dim=1)
        x = self.mlp(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':

    model = residual_FCN( (3, 3), 3, 5, 256, 1)

    X = torch.randn(10, 3, 3, 3)
    x_attrs = torch.randn(10, 5)


    y = model(X, x_attrs)
    print(y.shape)