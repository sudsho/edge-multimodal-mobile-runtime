"""
Distilled ECAPA-TDNN speaker embedding, kept small enough to sit on a phone.

Full ECAPA-TDNN from Desplanques et al. (2020) is ~6M params and 1024-dim
channels. This shrinks channels to ~96, keeps the res-2 SE blocks, attentive
statistics pooling, and drops down to a 128-dim embedding. Roughly 800K
params, which quantizes cleanly to int8 for on-device use.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // r, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // r, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1, keepdim=True)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class Res2Block(nn.Module):
    """Res2Net-style block with 4 scales, dilated convs, and an SE gate."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, scale: int = 4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        pad = (kernel_size // 2) * dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, kernel_size, padding=pad, dilation=dilation)
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spx = torch.split(x, self.width, dim=1)
        out = [spx[0]]
        y = spx[1]
        for i, conv in enumerate(self.convs):
            if i > 0:
                y = y + out[-1]
            y = conv(y)
            y = F.relu(self.bns[i](y))
            out.append(y)
        out = torch.cat(out, dim=1)
        return self.se(out)


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_channels: int, attn_channels: int = 32):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels * 3, attn_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attn_channels),
            nn.Tanh(),
            nn.Conv1d(attn_channels, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global context features
        t = x.size(-1)
        mu = x.mean(dim=-1, keepdim=True).expand(-1, -1, t)
        sig = x.std(dim=-1, keepdim=True).expand(-1, -1, t)
        h = torch.cat([x, mu, sig], dim=1)
        w = torch.softmax(self.attn(h), dim=-1)
        mean = (x * w).sum(dim=-1)
        var = ((x - mean.unsqueeze(-1)) ** 2 * w).sum(dim=-1).clamp(min=1e-6)
        return torch.cat([mean, var.sqrt()], dim=1)


class SpeakerECAPATiny(nn.Module):
    def __init__(
        self,
        n_mels: int = 40,
        channels: int = 96,
        kernel_sizes=(5, 3, 3, 3),
        dilations=(1, 2, 3, 4),
        emb_dim: int = 128,
        attn_channels: int = 32,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                Res2Block(channels, k, d)
                for k, d in zip(kernel_sizes, dilations)
            ]
        )
        self.mix = nn.Conv1d(channels * len(self.blocks), channels, kernel_size=1)
        self.pool = AttentiveStatsPool(channels, attn_channels)
        self.head = nn.Sequential(
            nn.Linear(channels * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, 1, T, n_mels) coming from MelFrontend, reshape to (B, n_mels, T)
        if mel.dim() == 4:
            mel = mel.squeeze(1).transpose(1, 2)
        x = self.stem(mel)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = F.relu(self.mix(x))
        x = self.pool(x)
        e = self.head(x)
        return F.normalize(e, p=2, dim=-1)
