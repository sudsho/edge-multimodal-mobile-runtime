# edge-multimodal-mobile-runtime

Wake-word + speaker verification on phones. Sub-30 ms end-to-end on iPhone
Neural Engine via CoreML, ONNX Runtime Mobile for Android (NNAPI), TFLite as
a third path.

## the problem

Phones and earbuds want to wake up on a keyword, then confirm it is
actually the owner speaking, without a round-trip to a server. This repo
trains two tiny models and exports them to the three runtimes people
actually ship:

- CoreML for iOS / Apple Neural Engine
- ONNX Runtime Mobile for Android (via NNAPI where available)
- TFLite as a fallback for older hardware and cross-vendor DSPs

Target budget: wake-word under 15 ms, speaker verify under 30 ms end to end,
running on a modern phone SoC.

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

| Model | Params | fp16 size | Purpose |
|---|---:|---:|---|
| `WakeWordCNN` | 118K | 236 KB | 12-class keyword spotting on Speech Commands v2 |
| `SpeakerECAPATiny` | 810K | 1.62 MB | 128-dim speaker embedding, cosine scored |
| Silero VAD v4 | 1.8M | 2.1 MB | speech activity gate (pretrained, not fine-tuned) |

## latency table

| Device | Wake p50 | Speaker p50 | End-to-end |
|---|---:|---:|---:|
| iPhone 14 Pro (A16 ANE) | 1.8 ms | 11.4 ms | 13.2 ms |
| iPhone 11 (A13 ANE) | 3.1 ms | 18.7 ms | 21.8 ms |
| Pixel 7 (Tensor G2 NNAPI) | 4.2 ms | 22.8 ms | 27.0 ms |
| Samsung S22 (SD8G1 NNAPI) | 4.7 ms | 24.6 ms | 29.3 ms |

Full table with p90, precisions, and older SoCs in
[benchmarks/results.md](benchmarks/results.md).

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
  deploy/           ios_swift_snippet.swift, android_kotlin_snippet.kt
tests/              pytest suite, no network / device required
docs/               architecture, coreml_notes, onnxrt_mobile_notes,
                    quantization, latency_report, privacy_and_on_device
benchmarks/         on-device numbers + method
notebooks/          analyze_latency.ipynb
Dockerfile          training / export env, deploy is on-device
Makefile            train / export / quantize / bench shortcuts
ci/test.yml.example CI config, out of .github/workflows for now
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

- iOS: drop `models/wakeword.mlpackage` and `models/speaker.mlpackage` into
  an Xcode 15 project. Use [src/deploy/ios_swift_snippet.swift](src/deploy/ios_swift_snippet.swift)
  as a starting point.
- Android: convert `.onnx` -> `.ort` with the ORT Mobile tool, drop into
  `app/src/main/assets/`. Use [src/deploy/android_kotlin_snippet.kt](src/deploy/android_kotlin_snippet.kt).

## privacy

Audio never leaves the device. VAD, mel front-end, wake CNN, and speaker
embedding all run in the on-device runtime. Enrollment embeddings live in
the platform keystore (iOS Keychain / Android EncryptedSharedPreferences).
See [docs/privacy_and_on_device.md](docs/privacy_and_on_device.md).

## license

MIT.
