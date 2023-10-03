# edge-multimodal-mobile-runtime

wake-word + speaker verification on phones. sub-30 ms on iPhone Neural Engine
via CoreML, ONNX Runtime Mobile as the Android path, TFLite as a third.

## the problem

phones and earbuds want to wake up on a keyword, then confirm it's actually
the owner speaking, without a round-trip to a server. this repo trains two
tiny models and exports them to the three runtimes people actually ship:

- CoreML for iOS / Apple Neural Engine
- ONNX Runtime Mobile for Android (via NNAPI where available)
- TFLite as a fallback for older hardware and cross-vendor DSPs

target budget: wake-word inference under 15 ms, speaker verify under 30 ms,
both on a single 30 ms mel frame window, running on a modern phone SoC.

## approach

1. mel filterbank front-end, 25 ms window, 10 ms hop, 40 mel bins
2. small conv wake-word head, ~120K params
3. tiny distilled ECAPA-TDNN speaker embedding, ~800K params
4. silero VAD in front so the models only fire on voiced frames

training on Speech Commands v2 for the wake-word, on a VoxCeleb1 subset for
the speaker net. see `src/train/`.

## deployment

exports live in `src/export/`. one script per runtime plus int8 quantization.
`src/bench/latency_probe.py` measures per-device latency on real audio.

## status

WIP. see `docs/` for design notes and `benchmarks/results.md` for numbers.
