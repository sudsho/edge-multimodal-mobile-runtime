# CoreML export notes

Toolchain: `coremltools==7.0`, Xcode 15, iOS 16 minimum deployment target.

## conversion command

```
python -m src.export.to_coreml \
    --which wakeword \
    --ckpt runs/wakeword/best.pt \
    --out models/wakeword.mlpackage
```

## what actually works on the Neural Engine

Not every op will fall onto the Neural Engine even when the model looks
tiny. The rules that matter for our two nets:

- **`convert_to="mlprogram"`** is mandatory. The old NeuralNetwork format
  does not reliably use the ANE on iOS 16.
- **fp16 compute precision** is what triggers ANE dispatch on A14 and up.
  fp32 stays on CPU/GPU. That is why every export uses
  `compute_precision=ct.precision.FLOAT16`.
- **BatchNorm folding**: coremltools folds BN into conv weights during
  conversion as long as the model is in `eval()` mode when we trace. If
  you skip `net.eval()` you get a scattered Add/Mul that the ANE refuses.
- **Attentive statistics pooling in ECAPA**: the `softmax(attn(h), dim=-1)`
  along the time axis compiles cleanly. But the concatenation of
  `[x, mu, sig]` needs the same dtype on all three tensors, otherwise
  coremltools inserts an explicit Cast and the ANE bails out mid-graph.
  We handle this by keeping the mean/std computed in fp16 as well.

## Xcode integration

Drop the `.mlpackage` into the project navigator, check "Add to target",
and Xcode 15 generates a Swift class named after the file. For
`wakeword.mlpackage` you get `WakeWordModel` with a `prediction(mel:)` method.

## profiling on device

Open Instruments -> CoreML template -> select the app -> record for 10 s
while triggering the wake path. Inspect the "compute unit" column: rows
should say `ANE` for the WakeWordCNN forward, occasionally `CPU` for the
first cold call, then ANE steady-state. If you see `GPU` on the wake path
you probably lost the fp16 precision hint somewhere in the export.

## known warts

- iOS 15 does not have ML Program. If you must support iOS 15, add a
  `minimum_deployment_target=ct.target.iOS15` branch and expect ~1.6x more
  latency because the neural-network format falls back to GPU on the wake path.
- ANE has a hard limit around 16 MB of activations per op. Neither of our
  models comes close, but if you crank the speaker embedding dim above 256
  you can trip it.
