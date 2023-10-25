# ONNX Runtime Mobile notes

Toolchain: `onnxruntime==1.16.0` on the desktop side, ONNX Runtime Mobile
1.16 on Android (AAR from Maven).

## producing the `.ort` file

ONNX Runtime Mobile ships a minimal binary that only knows a fixed set of
kernels. The tooling converts our full `.onnx` to a `.ort` that inlines
the required kernels and strips everything else.

```
python -m onnxruntime.tools.convert_onnx_models_to_ort \
    models/wakeword.onnx \
    --optimization_style Fixed \
    --target_platform arm64
```

The resulting `.ort` is roughly 30% smaller than the equivalent `.onnx`
and loads faster on cold start.

## execution providers

Order we ask for on Android:

1. `NnapiExecutionProvider` (skipped on API < 27, falls through)
2. `XnnpackExecutionProvider` (int8 kernels on CPU)
3. `CPUExecutionProvider` (fallback)

```kotlin
val opts = OrtSession.SessionOptions().apply {
    addNnapi(EnumSet.of(NnapiFlags.CPU_DISABLED))
    addXnnpack(mapOf("intra_op_num_threads" to "2"))
}
```

Disabling the CPU fallback inside NNAPI is important. If we let NNAPI
punt to CPU on unsupported subgraphs the whole thing runs slower than the
XNNPACK path because of cross-provider copies.

## kernel coverage

The WakeWordCNN graph is Conv2d + BN + ReLU + MaxPool + Linear. All of
that runs natively on NNAPI's Qualcomm Hexagon / Samsung NPU / Tensor NPU
backends and on XNNPACK.

The SpeakerECAPATiny graph uses Conv1d + BN + ReLU + Softmax + a
`concat + reduce_mean + reduce_std` block for attentive stats pooling.
NNAPI accelerates Conv1d, softmax, and reductions on Pixel 7 and S22.
On older devices (Pixel 4a, first-gen Tensor) some of these subgraphs
fall to XNNPACK, hence the wider p50/p90 spread in the results table.

## thread pinning

`intra_op_num_threads=2` is the sweet spot on Pixel 7. Going to 4 doesn't
help because the ONNX Runtime Mobile scheduler over-partitions the tiny
conv graphs and pays more in synchronization than it recovers in compute.

## known warts

- `.ort` files are ABI-locked to the ORT version they were produced from.
  Do not mix a 1.16-produced `.ort` with a 1.15 runtime.
- `dynamic_axes` for the speaker model's time dim works, but NNAPI on
  Tensor G2 refuses variable shapes; we pad every inference to 200 frames
  (2 s) on that path.
