# Latency results

Measured on-device via the CoreML profiler (Xcode 15 Instruments) and
Android Studio 2023.1 profiler. Desktop numbers via `src/bench/latency_probe.py`.
Single-threaded where possible for a fair comparison; on-device numbers are
whatever the runtime picks by default.

Input clip: 1.0 s of 16 kHz audio, 40 mel bins, 98 frames.

## Wake-word (WakeWordCNN, ~118K params)

| Device | Runtime | Precision | p50 (ms) | p90 (ms) |
|---|---|---|---:|---:|
| iPhone 14 Pro, A16 Neural Engine | CoreML fp16 | fp16 | 1.8 | 2.4 |
| iPhone 14 Pro, A16 CPU only | CoreML fp16 | fp16 | 3.4 | 4.1 |
| iPhone 11, A13 Neural Engine | CoreML fp16 | fp16 | 3.1 | 3.9 |
| Pixel 7, Tensor G2 NNAPI | ONNX Runtime Mobile | fp16 | 4.2 | 5.8 |
| Pixel 7, Tensor G2 CPU | ONNX Runtime Mobile | int8 dyn | 5.9 | 7.6 |
| Samsung S22, SD8G1 NNAPI | ONNX Runtime Mobile | fp16 | 4.7 | 6.3 |
| Pixel 4a, SD730 CPU | TFLite XNNPACK | int8 | 8.2 | 10.4 |
| M1 MacBook Air, CoreML | CoreML fp16 | fp16 | 1.1 | 1.5 |
| Ryzen 5600 desktop, ORT CPU | ONNX Runtime | fp32 | 3.6 | 4.9 |

## Speaker verify (SpeakerECAPATiny, ~810K params, 2 s clip)

| Device | Runtime | Precision | p50 (ms) | p90 (ms) |
|---|---|---|---:|---:|
| iPhone 14 Pro, A16 Neural Engine | CoreML fp16 | fp16 | 11.4 | 13.0 |
| iPhone 11, A13 Neural Engine | CoreML fp16 | fp16 | 18.7 | 22.2 |
| Pixel 7, Tensor G2 NNAPI | ONNX Runtime Mobile | fp16 | 22.8 | 27.9 |
| Samsung S22, SD8G1 NNAPI | ONNX Runtime Mobile | fp16 | 24.6 | 30.1 |
| M1 MacBook Air, CoreML | CoreML fp16 | fp16 | 6.9 | 8.4 |

## End-to-end wake + verify (single pass)

| Device | Total p50 (ms) |
|---|---:|
| iPhone 14 Pro (ANE) | 13.2 |
| iPhone 11 (ANE) | 21.8 |
| Pixel 7 (NNAPI) | 27.0 |
| Samsung S22 (NNAPI) | 29.3 |

Sub-30 ms target hits on Pixel 7 and everything Apple. Samsung is on the
boundary depending on thermal state, older SD8G1 units under load can spill
to ~34 ms and the wake-word gate stops the speaker path from running unless
we saw voiced speech in the last 300 ms window.

Method notes are in [docs/latency_report.md](../docs/latency_report.md).
