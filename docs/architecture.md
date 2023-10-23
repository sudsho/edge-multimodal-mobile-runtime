# Architecture

```
+------------------+     +-------------+     +-----------------+     +------------------+
|  16 kHz mic PCM  | --> |  Silero VAD | --> |  log-mel front  | --> |  WakeWordCNN     |
|  (device audio   |     |  gate       |     |  (40 mel, 10ms) |     |  ~118K params    |
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

- **VAD in front, always.** Silero VAD runs cheaply on every 30 ms window and
  gates both models. It removes ~85% of silence and non-speech from the
  wake-word workload, which is what makes the average power budget viable.
- **wake-word every 100 ms.** The CNN takes 1 s of mel context. We slide the
  window in 100 ms hops so a spoken keyword crosses ~10 evaluation windows
  and we take the max prob. This dominates the accuracy vs latency trade.
- **speaker verify only after a wake event.** Speaker verify is the more
  expensive model (~10 ms on ANE, ~25 ms on NNAPI). Gating it on wake means
  we run it once per wake attempt, not every 100 ms.
- **enrollment is on-device.** The user enrolls their voice by saying the
  keyword three times. We average the three L2-normalized embeddings and
  store the result in the app keychain. No audio leaves the device.

## model sizes

| Model | Params | fp16 size | int8 size | Purpose |
|---|---:|---:|---:|---|
| WakeWordCNN | 118K | 236 KB | 118 KB | binary keyword spotting on 12-class Speech Commands v2 |
| SpeakerECAPATiny | 810K | 1.62 MB | 810 KB | 128-dim speaker embedding, cosine-scored |
| Silero VAD | 1.8M | 2.1 MB (torchscript) | n/a | speech activity gate |

Total on-disk footprint at int8: ~3 MB. Fits inside an app bundle without
needing on-demand resources.

## runtime story

| OS | Runtime | Delegate |
|---|---|---|
| iOS 16+ | CoreML (ML Program) | Neural Engine + CPU fallback |
| Android 12+ | ONNX Runtime Mobile 1.16 | NNAPI where available, XNNPACK CPU otherwise |
| Older Android / cross-vendor DSP | TFLite 2.14 | GPU delegate for wake, CPU for speaker |

See [coreml_notes.md](coreml_notes.md), [onnxrt_mobile_notes.md](onnxrt_mobile_notes.md),
[quantization.md](quantization.md), and [latency_report.md](latency_report.md)
for the details of each path.
