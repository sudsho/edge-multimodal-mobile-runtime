"""
int8 post-training quantization for ONNX and CoreML.

- ONNX Runtime: static PTQ with a small calibration set (100 mel clips).
- CoreML: coremltools 7 `linear_quantize_weights` for int8 weight-only, and
  `linear_quantize_activations` if we have enough calibration samples.

TFLite int8 is handled inside to_tflite.py by passing a representative dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


# ONNX side ----------------------------------------------------------------

def quantize_onnx_static(
    src: str,
    dst: str,
    calibration_data: Iterable[np.ndarray],
    input_name: str = "mel",
):
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantType,
    )

    class Reader(CalibrationDataReader):
        def __init__(self, data):
            self.data = iter(data)

        def get_next(self):
            try:
                arr = next(self.data)
                return {input_name: arr.astype(np.float32)}
            except StopIteration:
                return None

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        src, dst, Reader(calibration_data),
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=True,
    )


def quantize_onnx_dynamic(src: str, dst: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)


# CoreML side --------------------------------------------------------------

def quantize_coreml_weights(src: str, dst: str, bits: int = 8):
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    mlmodel = ct.models.MLModel(src)
    op_cfg = OpLinearQuantizerConfig(mode="linear_symmetric", dtype=f"int{bits}")
    cfg = OptimizationConfig(global_config=op_cfg)
    q = linear_quantize_weights(mlmodel, cfg)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    q.save(dst)


# CLI ----------------------------------------------------------------------

def _fake_calib(n: int, shape):
    for _ in range(n):
        yield np.random.randn(*shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["onnx-dynamic", "onnx-static", "coreml"], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--calib-n", type=int, default=100)
    ap.add_argument("--shape", type=int, nargs="+", default=[1, 1, 98, 40])
    args = ap.parse_args()

    if args.which == "onnx-dynamic":
        quantize_onnx_dynamic(args.src, args.dst)
    elif args.which == "onnx-static":
        quantize_onnx_static(args.src, args.dst, _fake_calib(args.calib_n, args.shape))
    else:
        quantize_coreml_weights(args.src, args.dst)
    print(f"quantized -> {args.dst}")


if __name__ == "__main__":
    main()
