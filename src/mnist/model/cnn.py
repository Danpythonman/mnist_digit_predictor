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
        self.conv_1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same')
        self.conv_1_relu_1 = nn.ReLU()
        self.conv_1_norm = nn.BatchNorm2d(num_features=16)
        self.conv_1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
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
        x = self.linear(x)        # (B, 32 * H/2 * W/2) -> (B, 10)
        return x


class CNN2(nn.Module):
    '''
    Convolutional neural network.

    The architecture we will be using is

        (Conv -> ReLU -> BatchNorm -> Conv -> ReLU -> MaxPool -> Dropout) * N
            -> Flatten -> Dropout -> Linear
            -> (Linear -> Dropout -> ReLU) * M
            -> Linear
    '''

    def __init__(self, W: int, H: int):
        super().__init__()

        self.W = W
        self.H = H

        self.conv_1_1       = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same')
        self.conv_1_relu_1  = nn.ReLU()
        self.conv_1_norm    = nn.BatchNorm2d(num_features=16)
        self.conv_1_2       = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
        self.conv_1_relu_2  = nn.ReLU()
        self.conv_1_pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1_dropout = nn.Dropout(0.25)

        self.conv_2_1       = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same')
        self.conv_2_relu_1  = nn.ReLU()
        self.conv_2_norm    = nn.BatchNorm2d(num_features=32)
        self.conv_2_2       = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv_2_relu_2  = nn.ReLU()
        self.conv_2_pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2_dropout = nn.Dropout(0.25)

        self.conv_3_1       = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.conv_3_relu_1  = nn.ReLU()
        self.conv_3_norm    = nn.BatchNorm2d(num_features=64)
        self.conv_3_2       = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.conv_3_relu_2  = nn.ReLU()
        self.conv_3_pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_3_dropout = nn.Dropout(0.25)

        self.flatten         = nn.Flatten(start_dim=1)
        self.flatten_dropout = nn.Dropout(0.5)

        self.linear_1         = nn.Linear(in_features=64 * self.W//8 * self.H//8, out_features=2048)
        self.linear_1_relu    = nn.ReLU()
        self.linear_1_dropout = nn.Dropout(0.5)

        self.linear_2         = nn.Linear(in_features=2048, out_features=512)
        self.linear_2_relu    = nn.ReLU()
        self.linear_2_dropout = nn.Dropout(0.5)

        self.linear_3         = nn.Linear(in_features=512, out_features=128)
        self.linear_3_relu    = nn.ReLU()
        self.linear_3_dropout = nn.Dropout(0.5)

        self.projection = nn.Linear(in_features=128, out_features=10)

    def forward(self, x: Tensor) -> Tensor:
        # `x` is a tensor of shape `(B, C, H, W)` where `C = 1`
        B, C, H, W = x.shape

        x = self.conv_1_1(x)       # (B, 1, H, W) -> (B, 16, H, W)
        x = self.conv_1_relu_1(x)  # Same shape
        x = self.conv_1_norm(x)    # Same shape
        x = self.conv_1_2(x)       # (B, 16, H, W) -> (B, 16, H, W)
        x = self.conv_1_relu_2(x)  # Same shape
        x = self.conv_1_pool(x)    # (B, 16, H, W) -> (B, 16, H/2, W/2)
        x = self.conv_1_dropout(x) # Same shape

        x = self.conv_2_1(x)       # (B, 16, H/2, W/2) -> (B, 32, H/2, W/2)
        x = self.conv_2_relu_1(x)  # Same shape
        x = self.conv_2_norm(x)    # Same shape
        x = self.conv_2_2(x)       # (B, 32, H/2, W/2) -> (B, 32, H/2, W/2)
        x = self.conv_2_relu_2(x)  # Same shape
        x = self.conv_2_pool(x)    # (B, 32, H/2, W/2) -> (B, 32, H/4, W/4)
        x = self.conv_2_dropout(x) # Same shape

        x = self.conv_3_1(x)       # (B, 32, H/4, W/4) -> (B, 64, H/4, W/4)
        x = self.conv_3_relu_1(x)  # Same shape
        x = self.conv_3_norm(x)    # Same shape
        x = self.conv_3_2(x)       # (B, 64, H/4, W/4) -> (B, 64, H/4, W/4)
        x = self.conv_3_relu_2(x)  # Same shape
        x = self.conv_3_pool(x)    # (B, 64, H/4, W/4) -> (B, 64, H/8, W/8)
        x = self.conv_3_dropout(x) # Same shape

        x = self.flatten(x)         # (B, 64, H/8, W/8) -> (B, 64 * H/8 * W/8)
        x = self.flatten_dropout(x) # Same shape

        x = self.linear_1(x)         # (B, 256 * H/8 * W/8) -> (B, 2048)
        x = self.linear_1_relu(x)    # Same shape
        x = self.linear_1_dropout(x) # Same shape

        x = self.linear_2(x)         # (B, 2048) -> (B, 512)
        x = self.linear_2_relu(x)    # Same shape
        x = self.linear_2_dropout(x) # Same shape

        x = self.linear_3(x)         # (B, 512) -> (B, 128)
        x = self.linear_3_relu(x)    # Same shape
        x = self.linear_3_dropout(x) # Same shape

        x = self.projection(x)      # (B, 128) -> (B, 10)

        return x
