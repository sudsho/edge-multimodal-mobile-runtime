"""
per-runtime latency probe.

runs N inference passes on random mel input and reports p50/p90/p99 in
milliseconds. designed to run inside the CI shell (torch/onnxruntime path),
and can be pointed at a coreml or tflite file when a mac / android device
is attached.

on-device numbers in benchmarks/results.md come from the Xcode Instruments
profiler and Android Studio Profiler, this script produces the desktop /
CPU baseline that we compare against.
"""

from __future__ import annotations

import argparse
import time
from statistics import median
from typing import Callable, List

import numpy as np


def percentile(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[k]


def bench(fn: Callable[[np.ndarray], None], x: np.ndarray, n_warmup: int, n_runs: int):
    for _ in range(n_warmup):
        fn(x)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return {
        "p50_ms": median(times),
        "p90_ms": percentile(times, 90),
        "p99_ms": percentile(times, 99),
        "mean_ms": sum(times) / len(times),
        "n": len(times),
    }


# runtime backends ---------------------------------------------------------

def make_torch(ckpt: str, which: str, shape):
    import torch, yaml
    from src.models.wake_word_cnn import WakeWordCNN
    from src.models.speaker_ecapa_tiny import SpeakerECAPATiny
    cfg = yaml.safe_load(open("configs/default.yaml"))
    if which == "wakeword":
        net = WakeWordCNN(
            n_mels=cfg["audio"]["n_mels"], n_classes=12,
            channels=cfg["wake_word"]["channels"], fc=cfg["wake_word"]["fc"],
        )
    else:
        net = SpeakerECAPATiny(
            n_mels=cfg["audio"]["n_mels"],
            channels=cfg["speaker"]["channels"][-1],
            emb_dim=cfg["speaker"]["emb_dim"],
        )
    if ckpt:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    torch.set_num_threads(1)

    def fn(x):
        with torch.no_grad():
            _ = net(torch.from_numpy(x))
    return fn


def make_onnx(model_path: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def fn(x):
        sess.run(None, {input_name: x})
    return fn


def make_coreml(model_path: str):
    import coremltools as ct
    m = ct.models.MLModel(model_path)
    input_name = list(m.get_spec().description.input)[0].name

    def fn(x):
        m.predict({input_name: x})
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["torch", "onnx", "coreml"], required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--which", choices=["wakeword", "speaker"], default="wakeword")
    ap.add_argument("--shape", type=int, nargs="+", default=[1, 1, 98, 40])
    ap.add_argument("--n-warmup", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=200)
    args = ap.parse_args()

    x = np.random.randn(*args.shape).astype(np.float32)

    if args.runtime == "torch":
        fn = make_torch(args.model, args.which, args.shape)
    elif args.runtime == "onnx":
        fn = make_onnx(args.model)
    else:
        fn = make_coreml(args.model)

    stats = bench(fn, x, args.n_warmup, args.n_runs)
    print(f"runtime={args.runtime} which={args.which} shape={args.shape}")
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
