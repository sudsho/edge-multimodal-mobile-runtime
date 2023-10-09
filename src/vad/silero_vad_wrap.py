"""
thin wrapper around silero-vad v4.

we don't retrain VAD. we ship the released torchscript model and just gate
the wake-word / speaker path on speech frames. wraps the load + inference so
train and bench code doesn't have to care about the torch.hub weirdness.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch


class SileroVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 100,
        min_silence_ms: int = 200,
        model=None,
        utils=None,
    ):
        if model is None or utils is None:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                onnx=False,
                trust_repo=True,
            )
        self.model = model
        (self.get_speech_ts, _, _, _, _) = utils
        self.sr = sample_rate
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms

    def speech_segments(self, wav: torch.Tensor) -> List[Tuple[int, int]]:
        """returns [(start_sample, end_sample), ...] for voiced regions."""
        if wav.dim() > 1:
            wav = wav.squeeze()
        ts = self.get_speech_ts(
            wav,
            self.model,
            sampling_rate=self.sr,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
        )
        return [(x["start"], x["end"]) for x in ts]

    def is_speech(self, wav: torch.Tensor) -> bool:
        return len(self.speech_segments(wav)) > 0

    def crop_first_speech(
        self, wav: torch.Tensor, max_samples: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        segs = self.speech_segments(wav)
        if not segs:
            return None
        s, e = segs[0]
        if max_samples is not None:
            e = min(e, s + max_samples)
        return wav[..., s:e]
