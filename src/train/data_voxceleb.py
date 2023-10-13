"""
minimal VoxCeleb1 subset loader for speaker training.

expects the standard `wav/<spk_id>/<video_id>/<utt>.wav` layout.
we don't ship the data. this loader just walks the tree and yields
(waveform, speaker_index) pairs, with per-utterance random cropping so
each batch has fixed-length clips.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


class VoxCeleb1Subset(Dataset):
    def __init__(
        self,
        root: str,
        speakers: List[str] = None,
        sample_rate: int = 16000,
        clip_ms: int = 2000,
    ):
        self.root = Path(root)
        self.sr = sample_rate
        self.n_samples = int(sample_rate * clip_ms / 1000)

        all_speakers = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if speakers is None:
            speakers = all_speakers
        self.speakers = speakers
        self.spk_to_idx = {s: i for i, s in enumerate(self.speakers)}

        self.items: List[Tuple[Path, int]] = []
        for spk in self.speakers:
            for wav in (self.root / spk).rglob("*.wav"):
                self.items.append((wav, self.spk_to_idx[spk]))

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(dim=0)  # mono
        n = wav.shape[-1]
        if n >= self.n_samples:
            start = random.randint(0, n - self.n_samples)
            return wav[start:start + self.n_samples]
        return torch.nn.functional.pad(wav, (0, self.n_samples - n))

    def __getitem__(self, idx: int):
        path, spk = self.items[idx]
        return self._load(path), spk

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)
