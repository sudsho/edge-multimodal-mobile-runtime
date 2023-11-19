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
