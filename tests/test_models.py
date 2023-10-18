import torch

from src.models.wake_word_cnn import WakeWordCNN
from src.models.speaker_ecapa_tiny import SpeakerECAPATiny


def test_wakeword_shapes():
    net = WakeWordCNN(n_mels=40, n_classes=12)
    x = torch.randn(2, 1, 98, 40)
    y = net(x)
    assert y.shape == (2, 12)


def test_wakeword_param_budget():
    net = WakeWordCNN(n_mels=40, n_classes=12)
    assert net.n_params() < 130_000, f"too big: {net.n_params()}"


def test_speaker_shapes():
    net = SpeakerECAPATiny(n_mels=40, channels=96, emb_dim=128)
    x = torch.randn(2, 1, 200, 40)
    e = net(x)
    assert e.shape == (2, 128)
    # embeddings are L2 normalized
    norms = e.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_speaker_variable_time():
    net = SpeakerECAPATiny(n_mels=40, channels=96, emb_dim=128)
    for T in (150, 200, 300):
        x = torch.randn(1, 1, T, 40)
        e = net(x)
        assert e.shape == (1, 128)
