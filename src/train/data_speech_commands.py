"""
Speech Commands v0.02 dataset loader for the wake-word head.

we treat 10 common target words as positives, plus an "unknown" bucket for
the rest, plus a synthesized "silence" bucket sampled from the _background_
folder. this is the standard v2 setup used in the tf keyword-spotting tutorials.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


TARGETS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]
UNKNOWN = "_unknown_"
SILENCE = "_silence_"
LABELS = TARGETS + [UNKNOWN, SILENCE]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}


def _list_split(root: Path, split: str) -> List[Tuple[Path, str]]:
    """
    Speech Commands ships validation_list.txt and testing_list.txt.
    Everything else is training.
    """
    val = set((root / "validation_list.txt").read_text().splitlines())
    tst = set((root / "testing_list.txt").read_text().splitlines())

    items: List[Tuple[Path, str]] = []
    for word_dir in sorted(root.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("_"):
            continue
        word = word_dir.name
        for wav in word_dir.glob("*.wav"):
            rel = f"{word}/{wav.name}"
            if split == "val" and rel in val:
                items.append((wav, word))
            elif split == "test" and rel in tst:
                items.append((wav, word))
            elif split == "train" and rel not in val and rel not in tst:
                items.append((wav, word))
    return items


class SpeechCommandsWW(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 16000,
        clip_ms: int = 1000,
        silence_prob: float = 0.10,
        unknown_prob: float = 0.10,
    ):
        self.root = Path(root)
        self.split = split
        self.sr = sample_rate
        self.n_samples = int(sample_rate * clip_ms / 1000)
        self.silence_prob = silence_prob
        self.unknown_prob = unknown_prob

        self.items = _list_split(self.root, split)
        self.bg_files = list((self.root / "_background_noise_").glob("*.wav"))
        if not self.bg_files:
            raise RuntimeError("no _background_noise_ files found")

        # cache background noise, they are long
        self.bg_cache = []
        for bf in self.bg_files:
            wav, sr = torchaudio.load(bf)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            self.bg_cache.append(wav.squeeze(0))

    def __len__(self) -> int:
        return len(self.items)

    def _pad_or_crop(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n >= self.n_samples:
            start = random.randint(0, n - self.n_samples)
            return wav[..., start:start + self.n_samples]
        pad = self.n_samples - n
        left = random.randint(0, pad)
        return torch.nn.functional.pad(wav, (left, pad - left))

    def _sample_silence(self) -> torch.Tensor:
        bg = random.choice(self.bg_cache)
        # guard against short bg files (silence.wav in v0.02 is only ~1s)
        max_start = max(0, bg.shape[-1] - self.n_samples)
        start = random.randint(0, max_start) if max_start > 0 else 0
        clip = bg[start:start + self.n_samples].clone()
        if clip.shape[-1] < self.n_samples:
            clip = torch.nn.functional.pad(clip, (0, self.n_samples - clip.shape[-1]))
        clip = clip * random.uniform(0.0, 0.1)
        return clip.unsqueeze(0)

    def __getitem__(self, idx: int):
        r = random.random()
        if self.split == "train" and r < self.silence_prob:
            wav = self._sample_silence()
            label = SILENCE
        else:
            path, word = self.items[idx]
            wav, sr = torchaudio.load(path)
            if sr != self.sr:
                wav = torchaudio.functional.resample(wav, sr, self.sr)
            wav = self._pad_or_crop(wav)
            label = word if word in TARGETS else UNKNOWN

        return wav, LABEL_TO_IDX[label]
