"""
tiny convolutional wake-word head.

input:  (B, 1, T, n_mels)  log-mel filterbank, T ~= 98 frames
output: (B, n_classes)     class logits

roughly follows the arik / berg-kirkpatrick style small keyword-spotting
conv net used in a bunch of on-device speech papers. under ~120K params so
it fits comfortably in the ANE and NNAPI budgets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WakeWordCNN(nn.Module):
    def __init__(
        self,
        n_mels: int = 40,
        n_classes: int = 12,
        channels=(16, 32, 32),
        fc: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        c1, c2, c3 = channels

        # depthwise + pointwise blocks keep params low and stay ANE friendly
        self.conv1 = nn.Conv2d(1, c1, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # global-avg-pool at the end so time dim doesn't have to match at export
        self.fc1 = nn.Linear(c3, fc)
        self.fc2 = nn.Linear(fc, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, n_mels)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))

        # global-avg over (T, mel)
        x = x.mean(dim=(2, 3))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
