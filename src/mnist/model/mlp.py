'''
Multi-layer perceptron.
'''

import typing

import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):

    layers: nn.ModuleList

    def __init__(self, n_embds: typing.List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for n_embd1, n_embd2 in zip(n_embds, n_embds[1:]):
            self.layers.append(nn.Linear(n_embd1, n_embd2))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
