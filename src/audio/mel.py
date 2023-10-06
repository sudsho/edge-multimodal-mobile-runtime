"""
log-mel filterbank front-end. exportable via torchaudio ops.

kept as a nn.Module so it can be traced and folded into the CoreML / ONNX
graph if we want, or bypassed and computed on the phone side.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class MelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_ms: int = 25,
        hop_ms: int = 10,
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 7600.0,
    ):
        super().__init__()
        win_length = int(sample_rate * win_ms / 1000)
        hop_length = int(sample_rate * hop_ms / 1000)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            center=True,
        )
        self.eps = 1e-6

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) or (B, 1, T)
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        m = self.mel(wav)                           # (B, n_mels, frames)
        m = torch.log(m + self.eps)
        # to (B, 1, frames, n_mels) which matches WakeWordCNN input
        return m.transpose(1, 2).unsqueeze(1)
