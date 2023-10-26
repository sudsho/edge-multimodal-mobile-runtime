# Latency report

Full numbers in [benchmarks/results.md](../benchmarks/results.md). This doc
covers how they were measured and what we compare against.

## method

- 200 inference runs after 20 warmup runs, per (device, runtime, model, precision).
- Input: real 1 s (wake) / 2 s (speaker) audio clips converted to log-mel
  offline, not synthesized noise.
- iOS: Xcode 15 Instruments -> CoreML template, timing pulled from the
  `Compute Time` column per invocation.
- Android: Android Studio 2023.1 Profiler CPU + custom `Trace.beginSection`
  markers around each `session.run(...)` call, timings pulled from the
  systrace export.
- Desktop: `src/bench/latency_probe.py` with `time.perf_counter()` and single
  intra-op thread to remove noise from the numbers.

## comparisons

The literature and other on-device speech libraries we cross-checked:

- Pixel 7 NNAPI wake-word 4.2 ms p50: aligns with the numbers Google
  published for their own KWS "Hey Google" small models on the same SoC.
- iPhone 14 Pro ANE 1.8 ms p50 wake-word: consistent with the coremltools 7
  benchmark blog post on Conv2d fp16 workloads under 200K params.
- Speaker verify 11 ms p50 on ANE: about half the number the WeSpeaker
  authors report for a full ECAPA-TDNN on the same class of chip. That
  matches roughly because our net has ~1/6 the params.

## power budget

We do not report power numbers directly because they need controlled
lab conditions. As a proxy we report duty cycle: at 100 ms wake-word
tick with VAD gating, and 300 ms max window for speaker after a wake
event, the wake pipeline is active roughly 6% of the time in a normal
listening session (based on 5 test users, 10 minutes each). That
matches the "always on, minimal battery" claims for AirPods-class
wake-word implementations.

## thermal behavior

- Sustained wake path is fine, ANE stays cold.
- Sustained speaker-verify is fine at 1 verify / 4 s or slower.
- If the app calls verify > 5 times / second (which the pipeline never
  should, but we tested), the S22 SD8G1 unit throttles NNAPI to CPU
  around the 45 s mark and p50 spikes from 24 to 41 ms. Pixel 7 does
  not throttle at that load. iPhone 14 Pro does not throttle at any
  load we could sustain.

## repro

To reproduce the desktop baseline:

```
python -m src.bench.latency_probe \
    --runtime onnx \
    --model models/wakeword.int8.onnx \
    --which wakeword \
    --shape 1 1 98 40 \
    --n-runs 200
```

On-device numbers need the sample iOS / Android projects in `src/deploy/`
plus the profiler dance described above. Screenshots of representative
Instruments and systrace captures live in `benchmarks/` next to the raw
CSVs from the runs.
