"""
offline smoke tests: synthetic training learns, and the trained net exports
to ONNX with onnxruntime output matching torch.

these run on CPU in a few seconds with no dataset and no network.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.wake_word_cnn import WakeWordCNN
from src.train.data_synthetic import SyntheticMelWW


def _train_a_few_epochs(epochs=5):
    torch.manual_seed(0)
    train_ds = SyntheticMelWW(768, n_classes=12, n_frames=98, n_mels=40, seed=1)
    val_ds = SyntheticMelWW(256, n_classes=12, n_frames=98, n_mels=40, seed=7)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64)
    net = WakeWordCNN(n_mels=40, n_classes=12)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(epochs):
        net.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                correct += (net(x).argmax(-1) == y).sum().item()
                total += y.numel()
        best = max(best, correct / total)
    return net, best


def test_synthetic_training_beats_chance():
    _, best = _train_a_few_epochs()
    # 12 classes -> chance is 1/12 = 0.083; require clearly above it
    assert best > 2 * (1.0 / 12), f"best val acc {best:.3f} did not clear 2x chance"


def test_trained_wakeword_onnx_parity():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    net, _ = _train_a_few_epochs(epochs=2)
    net.eval()
    example = torch.randn(1, 1, 98, 40)
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "wakeword.onnx")
        torch.onnx.export(
            net, example, out,
            input_names=["mel"], output_names=["logits"],
            opset_version=15, do_constant_folding=True,
        )
        onnx.checker.check_model(onnx.load(out))
        with torch.no_grad():
            y_torch = net(example).numpy()
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        y_onnx = sess.run(None, {"mel": example.numpy()})[0]
        assert y_torch.shape == y_onnx.shape
        assert np.abs(y_torch - y_onnx).max() < 1e-4
