"""
offline end-to-end smoke test.

runs the whole clone-and-run path on a CPU with no downloads and no keys:

  1. train the tiny WakeWordCNN a few epochs on seeded synthetic mels
     (accuracy has to climb above 1/n_classes chance)
  2. export the trained net to ONNX (opset 15)
  3. run ONNX Runtime inference and confirm outputs match torch within tol
  4. print a CPU latency number for both torch and onnxruntime
  5. report CoreML / TFLite export status (skipped off-mac / when TF missing)

exits 0 on success, non-zero on any failed assertion. this is the verified
path; the real Speech Commands result and the on-device latency numbers still
need the dataset and a phone respectively.

usage:
    python smoke.py
    python smoke.py --epochs 5 --train-samples 1024
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.wake_word_cnn import WakeWordCNN
from src.train.data_synthetic import SyntheticMelWW
from src.train.data_speech_commands import LABELS

N_CLASSES = len(LABELS)
N_FRAMES = 98
N_MELS = 40


def train(epochs: int, train_samples: int, seed: int = 0):
    torch.manual_seed(seed)
    train_ds = SyntheticMelWW(train_samples, N_CLASSES, N_FRAMES, N_MELS, seed=1)
    val_ds = SyntheticMelWW(256, N_CLASSES, N_FRAMES, N_MELS, seed=7)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    net = WakeWordCNN(n_mels=N_MELS, n_classes=N_CLASSES)
    print(f"wake-word CNN params: {net.n_params() / 1e3:.1f}K")
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()

    chance = 1.0 / N_CLASSES
    best_acc = 0.0
    for e in range(epochs):
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
        val_acc = correct / total
        best_acc = max(best_acc, val_acc)
        print(f"  epoch {e}  val_acc {val_acc:.3f}  (chance {chance:.3f})")
    return net, best_acc


def export_onnx(net: nn.Module, path: str, opset: int = 15):
    net.eval()
    example = torch.randn(1, 1, N_FRAMES, N_MELS)
    torch.onnx.export(
        net, example, path,
        input_names=["mel"], output_names=["logits"],
        opset_version=opset, do_constant_folding=True,
    )
    return example


def check_parity(net: nn.Module, path: str, example: torch.Tensor, tol: float):
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(path))
    with torch.no_grad():
        y_torch = net(example).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"mel": example.numpy()})[0]
    assert y_torch.shape == y_onnx.shape, (y_torch.shape, y_onnx.shape)
    max_abs = float(np.abs(y_torch - y_onnx).max())
    return sess, max_abs


def latency(fn, x, n_warmup=20, n_runs=200):
    for _ in range(n_warmup):
        fn(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], times[int(0.9 * (len(times) - 1))]


def report_optional_exports():
    is_mac = platform.system() == "Darwin"
    if is_mac:
        try:
            import coremltools  # noqa: F401
            print("CoreML: coremltools available on macOS (run `make export-coreml`)")
        except ImportError:
            print("CoreML: on macOS but coremltools not installed (pip install coremltools)")
    else:
        print(f"CoreML: skipped, export needs macOS (this host is {platform.system()})")
    try:
        import tensorflow  # noqa: F401
        import onnx_tf  # noqa: F401
        print("TFLite: tensorflow + onnx_tf available (run `make export-tflite`)")
    except ImportError:
        print("TFLite: skipped, optional path needs tensorflow + onnx_tf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train-samples", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    torch.set_num_threads(1)

    print("== 1. train tiny wake-word on synthetic mels ==")
    net, best_acc = train(args.epochs, args.train_samples)
    chance = 1.0 / N_CLASSES
    assert best_acc > 2 * chance, f"best acc {best_acc:.3f} did not clear 2x chance {2*chance:.3f}"
    print(f"  -> trained, best val_acc {best_acc:.3f} clears 2x chance {2*chance:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = str(Path(tmp) / "wakeword.onnx")

        print("== 2. export to ONNX ==")
        example = export_onnx(net, onnx_path)
        size_kb = Path(onnx_path).stat().st_size / 1024
        print(f"  -> {onnx_path}  ({size_kb:.1f} KB)")

        print("== 3. onnxruntime vs torch parity ==")
        sess, max_abs = check_parity(net, onnx_path, example, args.tol)
        assert max_abs < args.tol, f"parity {max_abs:.2e} exceeds tol {args.tol:.0e}"
        print(f"  -> max abs diff {max_abs:.2e} < tol {args.tol:.0e}  OK")

        print("== 4. CPU latency (1 thread, 200 runs) ==")
        net.eval()

        def torch_fn(x):
            with torch.no_grad():
                net(torch.from_numpy(x))

        input_name = sess.get_inputs()[0].name

        def onnx_fn(x):
            sess.run(None, {input_name: x})

        x = example.numpy()
        t_p50, t_p90 = latency(torch_fn, x)
        o_p50, o_p90 = latency(onnx_fn, x)
        print(f"  torch  p50 {t_p50:.3f} ms  p90 {t_p90:.3f} ms")
        print(f"  onnx   p50 {o_p50:.3f} ms  p90 {o_p90:.3f} ms")

    print("== 5. optional exports ==")
    report_optional_exports()

    print("\nSMOKE OK: train -> onnx -> onnxruntime parity -> latency all passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
