"""Independent implementations of SIREN and Fourier-feature ReLU baselines."""
import math
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(self, input_dim, width, first=False, omega=30.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, width)
        self.omega = omega
        bound = 1 / input_dim if first else math.sqrt(6 / input_dim) / omega
        nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class Field(nn.Module):
    def __init__(self, kind="siren", width=64, depth=3, frequencies=16, scale=3.0):
        super().__init__()
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")
        self.kind = kind
        if kind == "siren":
            layers = [SineLayer(3, width, first=True)]
            layers += [SineLayer(width, width) for _ in range(depth - 1)]
            output = nn.Linear(width, 1)
            nn.init.uniform_(output.weight, -math.sqrt(6 / width) / 30, math.sqrt(6 / width) / 30)
            self.net = nn.Sequential(*layers, output)
        elif kind == "fourier":
            self.register_buffer("basis", torch.randn(3, frequencies) * scale)
            layers = [nn.Linear(2 * frequencies, width), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(width, width), nn.ReLU()]
            self.net = nn.Sequential(*layers, nn.Linear(width, 1))
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "fourier":
            phase = 2 * torch.pi * (x @ self.basis)
            x = torch.cat((phase.sin(), phase.cos()), -1)
        return self.net(x)
