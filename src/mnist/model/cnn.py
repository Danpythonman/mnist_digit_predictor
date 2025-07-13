'''
Multi-layer perceptron.
'''

import typing

import torch.nn as nn
from torch import Tensor


class CNN(nn.Module):
    '''
    Convolutional neural network.

    The architecture we will be using is

        (Conv -> ReLU -> BatchNorm -> Conv -> ReLU -> MaxPool) * N -> Flatten -> (Linear -> ReLU) * M -> Linear
    '''

    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding='same')
        self.conv_1_relu_1 = nn.ReLU()
        self.conv_1_norm = nn.BatchNorm2d(num_features=16)
        self.conv_1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding='same')
        self.conv_1_relu_2 = nn.ReLU()
        self.conv_1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(in_features=32 * 28//2 * 28//2, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        # `x` is a tensor of shape `(B, C, H, W)` where `C = 1`
        B, C, H, W = x.shape

        x = self.conv_1_1(x)      # (B, 1, H, W) -> (B, 16, H, W)
        x = self.conv_1_relu_1(x) # Same shape
        x = self.conv_1_norm(x)   # Same shape
        x = self.conv_1_2(x)      # (B, 16, H, W) -> (B, 32, H, W)
        x = self.conv_1_relu_2(x) # Same shape
        x = self.conv_1_pool(x)   # (B, 32, H, W) -> (B, 32, H/2, W/2)
        x = self.flatten(x)       # (B, 32, H/2, H/2) -> (B, 32 * H/2 * W/2)
        x = self.linear(x)
        return x


class CNN2(nn.Module):
    '''
    Convolutional neural network.

    The architecture we will be using is

        (Conv -> ReLU -> BatchNorm -> Conv -> ReLU -> MaxPool) * N -> Flatten -> (Linear -> ReLU) * M -> Linear
    '''

    def __init__(self):
        super().__init__()
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2, padding='same')
        self.conv_1_relu_1 = nn.ReLU()
        self.conv_1_norm = nn.BatchNorm2d(num_features=8)
        self.conv_1_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, padding='same')
        self.conv_1_relu_2 = nn.ReLU()
        self.conv_1_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding='same')
        self.conv_2_relu_1 = nn.ReLU()
        self.conv_2_norm = nn.BatchNorm2d(num_features=32)
        self.conv_2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding='same')
        self.conv_2_relu_2 = nn.ReLU()
        self.conv_2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten(start_dim=1)
        self.linear_1 = nn.Linear(in_features=64 * 28//4 * 28//4, out_features=1024)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(in_features=1024, out_features=256)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        # `x` is a tensor of shape `(B, C, H, W)` where `C = 1`
        B, C, H, W = x.shape

        x = self.conv_1_1(x)      # (B, 1, H, W) -> (B, 8, H, W)
        x = self.conv_1_relu_1(x) # Same shape
        x = self.conv_1_norm(x)   # Same shape
        x = self.conv_1_2(x)      # (B, 8, H, W) -> (B, 16, H, W)
        x = self.conv_1_relu_2(x) # Same shape
        x = self.conv_1_pool(x)   # (B, 16, H, W) -> (B, 16, H/2, W/2)

        x = self.conv_2_1(x)      # (B, 16, H/2, W/2) -> (B, 32, H/2, W/2)
        x = self.conv_2_relu_1(x) # Same shape
        x = self.conv_2_norm(x)   # Same shape
        x = self.conv_2_2(x)      # (B, 32, H/2, W/2) -> (B, 64, H/2, W/2)
        x = self.conv_2_relu_2(x) # Same shape
        x = self.conv_2_pool(x)   # (B, 64, H/2, W/2) -> (B, 64, H/4, W/4)

        x = self.flatten(x)       # (B, 64, H/4, H/4) -> (B, 64 * H/4 * W/4)

        x = self.linear_1(x)      # (B, 64 * H/4 * W/4) -> (B, 1024)
        x = self.relu_1(x)        # Same shape
        x = self.linear_2(x)      # (B, 1024) -> (B, 256)
        x = self.relu_2(x)        # Same shape

        x = self.linear_3(x)      # (B, 256) -> (B, 10)

        return x
