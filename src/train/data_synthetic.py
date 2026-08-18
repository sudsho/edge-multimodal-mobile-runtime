"""
synthetic mel-spectrogram dataset for the wake-word head.

this exists so the training / export / parity loop can run offline on a CPU
in seconds, with no Speech Commands download and no torchaudio decode. it is
NOT a substitute for the real dataset for any real result, it just gives the
tiny CNN a learnable signal so we can prove the train -> export -> onnxruntime
path end to end.

each of the n_classes gets a fixed spectral "signature" (a couple of localized
time-frequency bands, drawn once from a seeded generator). a sample is that
signature plus gaussian noise. the bands are local so the conv filters have
something real to latch onto, and the noise is strong enough that the task is
learnable rather than trivial. accuracy climbs well above 1/n_classes chance
within a few epochs.

samples come out already in the WakeWordCNN input layout (1, T, n_mels), so no
mel front-end is needed at train time on this path.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


def make_class_templates(
    n_classes: int,
    n_frames: int,
    n_mels: int,
    bands_per_class: int = 2,
    seed: int = 123,
) -> torch.Tensor:
    """returns (n_classes, n_frames, n_mels) fixed per-class signatures."""
    g = torch.Generator().manual_seed(seed)
    templates = torch.zeros(n_classes, n_frames, n_mels)
    for c in range(n_classes):
        for _ in range(bands_per_class):
            f_center = int(torch.randint(0, n_mels, (1,), generator=g))
            t_center = int(torch.randint(0, n_frames, (1,), generator=g))
            f_half = int(torch.randint(3, 8, (1,), generator=g))
            t_half = int(torch.randint(10, 30, (1,), generator=g))
            amp = 1.5 + float(torch.rand(1, generator=g))
            f0, f1 = max(0, f_center - f_half), min(n_mels, f_center + f_half)
            t0, t1 = max(0, t_center - t_half), min(n_frames, t_center + t_half)
            templates[c, t0:t1, f0:f1] += amp
    return templates


class SyntheticMelWW(Dataset):
    """
    yields (mel, label) where mel is (1, n_frames, n_mels), matching the
    WakeWordCNN input, and label is an int class id.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        n_classes: int = 12,
        n_frames: int = 98,
        n_mels: int = 40,
        noise_std: float = 1.0,
        seed: int = 0,
        template_seed: int = 123,
    ):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.noise_std = noise_std
        self.templates = make_class_templates(
            n_classes, n_frames, n_mels, seed=template_seed
        )
        # precompute a fixed label per index so length and labels are stable
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, n_classes, (n_samples,), generator=g)
        self.base_seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        label = int(self.labels[idx])
        # per-sample deterministic noise so the dataset is reproducible
        g = torch.Generator().manual_seed(self.base_seed * 1_000_003 + idx)
        noise = torch.randn(self.n_frames, self.n_mels, generator=g) * self.noise_std
        mel = self.templates[label] + noise
        return mel.unsqueeze(0), label
