"""
Export to TFLite via the onnx -> tf -> tflite path.

TFLite is the third target, mostly for older Android and cross-vendor DSPs
that don't have NNAPI acceleration. We use int8 dynamic-range as the default
because the wake-word model is tiny and it lets us use the GPU delegate.

requires onnx-tf and tensorflow (both are not pinned in requirements.txt
because this path is optional; install on demand).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _require(mod: str):
    try:
        __import__(mod)
    except ImportError:
        raise SystemExit(f"missing optional dep: {mod}. install to use tflite export.")


def onnx_to_tf(onnx_path: str, saved_model_dir: str):
    _require("onnx")
    _require("onnx_tf")
    import onnx
    from onnx_tf.backend import prepare
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    tf_rep.export_graph(saved_model_dir)


def tf_to_tflite(saved_model_dir: str, out: str, dynamic_int8: bool = True):
    _require("tensorflow")
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if dynamic_int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = conv.convert()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(tflite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true", help="skip int8 dynamic-range")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "saved_model"
        onnx_to_tf(args.onnx, str(saved))
        tf_to_tflite(str(saved), args.out, dynamic_int8=(not args.fp32))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
