"""
vad wrapper unit test with a fake silero model.

we don't hit torch.hub in unit tests. we build a tiny stand-in that returns
canned speech segments so the wrapper's public surface is exercised.
"""

import torch

from src.vad.silero_vad_wrap import SileroVAD


class _FakeModel:
    pass


def _fake_utils():
    def get_speech_ts(wav, model, sampling_rate, threshold, min_speech_duration_ms, min_silence_duration_ms):
        # pretend there's speech in the middle half of the clip
        n = wav.shape[-1]
        return [{"start": n // 4, "end": 3 * n // 4}]
    return (get_speech_ts, None, None, None, None)


def test_speech_segments_uses_utils():
    vad = SileroVAD(model=_FakeModel(), utils=_fake_utils())
    wav = torch.zeros(16000)
    segs = vad.speech_segments(wav)
    assert segs == [(4000, 12000)]
    assert vad.is_speech(wav)


def test_crop_first_speech_clips_to_max_samples():
    vad = SileroVAD(model=_FakeModel(), utils=_fake_utils())
    wav = torch.arange(16000, dtype=torch.float32)
    cropped = vad.crop_first_speech(wav, max_samples=2000)
    assert cropped.shape[-1] == 2000
    # starts at n/4 = 4000
    assert cropped[0].item() == 4000.0


def test_no_speech_returns_none_for_crop():
    def empty_utils():
        return (lambda *a, **k: [], None, None, None, None)
    vad = SileroVAD(model=_FakeModel(), utils=empty_utils())
    assert vad.crop_first_speech(torch.zeros(16000)) is None
