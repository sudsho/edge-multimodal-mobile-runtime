# Latency report

On-device latency has not been measured for this repo. The bench script
under `src/bench/latency_probe.py` runs desktop-only timing over a
torch, onnxruntime, or coreml session on random input, wraps
`time.perf_counter()` around each call, and prints p50 / p90.

To reproduce the desktop baseline:

```
python -m src.bench.latency_probe \
    --runtime onnx \
    --model models/wakeword.int8.onnx \
    --which wakeword \
    --shape 1 1 98 40 \
    --n-runs 200
```

For real on-device numbers you would need to profile a running iOS or
Android app with Xcode Instruments (CoreML template) or Android Studio
Profiler with `Trace.beginSection` markers around each `session.run(...)`
call. Those apps are not part of this repo; the `src/deploy/*.swift` and
`src/deploy/*.kt` files are integration sketches, not a checked-in
Xcode or Android Studio project.
