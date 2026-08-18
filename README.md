# edge-multimodal-mobile-runtime

Small wake-word and speaker-embedding models with export paths to CoreML,
ONNX Runtime Mobile, and TFLite. This is a study repo, not a shipped
runtime.

## the problem

Wake-word plus speaker-verify is a common on-device speech pattern. This
repo trains two tiny PyTorch models and provides scaffolding to export
them to the three runtimes typically used on phones:

- CoreML for iOS
- ONNX Runtime Mobile for Android
- TFLite as an alternative path

Sample Swift and Kotlin snippets show the intended integration shape,
but they are code sketches, not a working iOS or Android app.

## quick start (runs offline, no keys)

The default demo path needs no dataset, no download, and no cloud creds. It
trains the tiny wake-word CNN on seeded synthetic mel-spectrograms, exports to
ONNX, runs ONNX Runtime, checks that the runtime output matches torch, and
prints a CPU latency number. Runs in well under a minute on a laptop CPU.

```
pip install -r requirements.txt
python smoke.py
```

Real output from a CPU run (Windows, torch 2.5.1, onnxruntime 1.20.1):

```
== 1. train tiny wake-word on synthetic mels ==
wake-word CNN params: 17.1K
  epoch 0  val_acc 0.172  (chance 0.083)
  epoch 1  val_acc 0.234  (chance 0.083)
  epoch 2  val_acc 0.570  (chance 0.083)
  epoch 3  val_acc 0.898  (chance 0.083)
  epoch 4  val_acc 0.801  (chance 0.083)
  epoch 5  val_acc 0.836  (chance 0.083)
  -> trained, best val_acc 0.898 clears 2x chance 0.167
== 2. export to ONNX ==
  -> ...\wakeword.onnx  (67.9 KB)
== 3. onnxruntime vs torch parity ==
  -> max abs diff 4.77e-07 < tol 1e-04  OK
== 4. CPU latency (1 thread, 200 runs) ==
  torch  p50 1.873 ms  p90 1.981 ms
  onnx   p50 0.329 ms  p90 0.374 ms
== 5. optional exports ==
CoreML: skipped, export needs macOS (this host is Windows)
TFLite: skipped, optional path needs tensorflow + onnx_tf

SMOKE OK: train -> onnx -> onnxruntime parity -> latency all passed.
```

Tests:

```
pytest -q
# 11 passed
```

What is and is not verified here:

- Verified on CPU: synthetic training climbs well above chance, ONNX export,
  ONNX Runtime inference, torch-vs-ONNX parity within 1e-4, and a CPU latency
  number. This is a functional proof of the train -> export -> run loop, not a
  quality result.
- The accuracy above is on synthetic separable data, not real keyword spotting.
  A real number needs Speech Commands (see `make train-ww DATA_WW=...`).
- CoreML export runs on macOS only and is skipped elsewhere with a message.
  TFLite export is an optional path that needs tensorflow + onnx-tf.
- On-device latency (ANE / NNAPI) is not measured here; `smoke.py` prints the
  desktop CPU baseline only.

Train the wake-word head on synthetic data on its own:

```
make train-ww-synthetic       # or: python -m src.train.train_wakeword --synthetic
```

## architecture

```
mic PCM -> Silero VAD -> log-mel front-end -> WakeWordCNN
                                                    |
                                             wake_prob > 0.9
                                                    v
                                     2 s mel window -> SpeakerECAPATiny
                                                    |
                                       cosine(emb, enroll) >= 0.65
                                                    v
                                          verified wake event
```

Full diagram in [docs/architecture.md](docs/architecture.md).

## models

Both nets are tiny by design. Actual parameter counts as instantiated
with the shipped defaults:

| Model | Params (approx.) | Purpose |
|---|---:|---|
| `WakeWordCNN` | ~17K | 12-class keyword spotting on Speech Commands v2 |
| `SpeakerECAPATiny` | ~129K | 128-dim speaker embedding, cosine scored |
| Silero VAD v4 | pretrained, upstream | speech activity gate |

Counts come from `sum(p.numel() for p in net.parameters())` on the
default constructor args. Change `channels`, `emb_dim`, or `attn_channels`
and the count moves.

## latency

On-device latency is not measured in this repo. `src/bench/latency_probe.py`
runs desktop-only timing over a torch, onnxruntime, or coreml session on
random input and prints p50/p90. See [docs/latency_report.md](docs/latency_report.md)
for what it does and does not cover.

## repo layout

```
configs/            default.yaml
src/
  audio/            mel front-end (torchaudio, exportable)
  models/           WakeWordCNN, SpeakerECAPATiny
  train/            train_wakeword.py, train_speaker.py, dataset loaders
  vad/              silero_vad_wrap.py
  export/           to_coreml.py, to_onnx.py, to_tflite.py, quantize.py
  bench/            latency_probe.py (torch / onnxruntime / coreml backends)
  deploy/           ios_swift_snippet.swift, android_kotlin_snippet.kt (sketches)
tests/              pytest suite for the model + export + vad wrapper
docs/               architecture, coreml_notes, onnxrt_mobile_notes,
                    quantization, latency_report, privacy_and_on_device
benchmarks/         placeholder
Dockerfile          training / export env
Makefile            train / export / quantize / bench shortcuts
ci/test.yml.example CI config
```

## setup

```
pip install -r requirements.txt
```

or use the container:

```
docker build -t edgemm .
docker run --rm --gpus all -v $PWD:/workspace edgemm bash
```

## training

Wake-word head on Speech Commands v0.02:

```
make train-ww DATA_WW=/path/to/speech_commands_v0.02
```

Speaker embedding on a VoxCeleb1 subset:

```
make train-spk DATA_SPK=/path/to/voxceleb1/wav
```

## exports

```
make export-coreml    # models/wakeword.mlpackage, models/speaker.mlpackage
make export-onnx      # models/wakeword.onnx,   models/speaker.onnx
make export-tflite    # models/wakeword.tflite  (optional path)
make quantize         # int8 variants
```

Runtime notes:
- [docs/coreml_notes.md](docs/coreml_notes.md) covers mlprogram + fp16 + ANE.
- [docs/onnxrt_mobile_notes.md](docs/onnxrt_mobile_notes.md) covers NNAPI / XNNPACK.
- [docs/quantization.md](docs/quantization.md) covers int8 tradeoffs.

## deploy

The Swift and Kotlin files under `src/deploy/` are integration sketches
that show the shape of the wake plus verify pipeline. Several helper
methods are stubs (`fatalError("impl")` in Swift, no-op ring buffer in
Kotlin). There is no `.xcodeproj` or Android Studio project checked in.

## privacy

The intent of the pipeline is on-device inference: VAD, mel front-end,
wake CNN, and speaker embedding all run locally with no network calls.
See [docs/privacy_and_on_device.md](docs/privacy_and_on_device.md) for the
data-flow diagram and the platform considerations that matter in a
real integration.

## license

MIT.
