# Architecture

```
+------------------+     +-------------+     +-----------------+     +------------------+
|  16 kHz mic PCM  | --> |  Silero VAD | --> |  log-mel front  | --> |  WakeWordCNN     |
|  (device audio   |     |  gate       |     |  (40 mel, 10ms) |     |  (~17K params)   |
|   thread)        |     +-------------+     +-----------------+     +--------+---------+
+------------------+                                                          |
                                                                              | wake_prob > 0.9
                                                                              v
                                                     +---------------+  +------------+
                                                     |  2 s mel win  |  |  Speaker   |
                                                     |  (200 x 40)   |->|  ECAPA     |
                                                     +---------------+  |  Tiny      |
                                                                        +-----+------+
                                                                              |
                                                                              v
                                                                   cosine(emb, enroll) >= 0.65
                                                                              |
                                                                              v
                                                                   verified wake event
```

## why the pipeline is shaped this way

- **VAD in front, always.** Silero VAD runs cheaply on every 30 ms window
  and gates both downstream models, so the wake path only runs on frames
  the VAD marks as speech. The repo does not measure the resulting
  duty-cycle reduction.
- **wake-word every 100 ms.** The CNN takes 1 s of mel context. Sliding
  the window in 100 ms hops means a spoken keyword crosses several
  evaluation windows and we take the max prob.
- **speaker verify only after a wake event.** Speaker verify is the more
  expensive model. Gating it on a wake means it runs once per wake
  attempt, not every 100 ms.
- **enrollment is on-device.** The intended flow is that enrollment
  audio, once turned into a 128-dim embedding, is what gets stored on
  the device. The Swift and Kotlin snippets expose a `setEnrollment(emb)`
  setter only; the capture + averaging step and the platform keystore
  write are not implemented in this repo.

## model sizes

Instantiated with the default constructor arguments:

| Model | Params (approx.) | Purpose |
|---|---:|---|
| WakeWordCNN | ~17K | 12-class keyword spotting on Speech Commands v2 |
| SpeakerECAPATiny | ~129K | 128-dim speaker embedding, cosine-scored |
| Silero VAD | pretrained, upstream | speech activity gate |

fp16 or int8 on-disk sizes depend on which export path is used. See
[quantization.md](quantization.md) for the paths available; this repo
does not ship measured file sizes for the two nets.

## runtime story

| OS | Runtime | Delegate |
|---|---|---|
| iOS 16+ | CoreML (ML Program) | Neural Engine + CPU fallback |
| Android 12+ | ONNX Runtime Mobile 1.16 | NNAPI where available, XNNPACK CPU otherwise |
| Older Android / cross-vendor DSP | TFLite 2.14 | GPU delegate for wake, CPU for speaker |

See [coreml_notes.md](coreml_notes.md), [onnxrt_mobile_notes.md](onnxrt_mobile_notes.md),
[quantization.md](quantization.md), and [latency_report.md](latency_report.md)
for the details of each path.
