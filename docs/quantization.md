# Quantization

We ship fp16 weights by default. int8 is available as an optional path for
devices that need the extra 2x speedup or the smaller download.

## impact summary

| Model | Precision | Params size | Wake acc / EER | Pixel 7 NNAPI p50 |
|---|---|---:|---|---:|
| WakeWord | fp16 | 236 KB | 94.7% acc | 4.2 ms |
| WakeWord | int8 dyn | 118 KB | 94.5% acc | 3.1 ms |
| WakeWord | int8 static | 118 KB | 94.3% acc | 2.9 ms |
| Speaker | fp16 | 1.62 MB | 3.1% EER | 22.8 ms |
| Speaker | int8 weights | 810 KB | 3.4% EER | 19.4 ms |
| Speaker | int8 full | 810 KB | 4.6% EER | 17.1 ms |

fp16 is a free win on ANE and NNAPI. int8-weights is a small accuracy hit
worth taking on Android to fit the download budget. Full int8 on the speaker
net loses more EER than we're comfortable with because of the softmax /
statistics-pool block, so we ship int8-weights-only there.

## ONNX static PTQ

- Calibration set: 500 spoken utterances (mixed target words + unknown +
  silence) drawn from Speech Commands validation.
- Per-channel weight quantization, per-tensor activation quantization.
- `activation_type=QUInt8`, `weight_type=QInt8`.

```
python -m src.export.quantize \
    --which onnx-static \
    --src models/wakeword.onnx \
    --dst models/wakeword.int8.onnx \
    --calib-n 500 --shape 1 1 98 40
```

## CoreML weight-only quantization

`coremltools.optimize.coreml.linear_quantize_weights` with
`mode="linear_symmetric"`. This keeps activations in fp16 so the ANE stays
happy and only shrinks the on-disk weights.

## TFLite

`tf.lite.Optimize.DEFAULT` gives dynamic-range int8 by default (weights
int8, activations kept in fp32 at runtime). For full int8 pass a
`representative_dataset` yielding ~100 mel clips.

## calibration data

Whatever you use for calibration should match the deploy distribution.
We use held-out Speech Commands v2 for the wake-word head, and held-out
VoxCeleb1 for the speaker net. Using random noise (like the CI smoke test
does) will give visibly worse accuracy after static quantization.

## rules of thumb we ended up with

- Always fp16 on the ANE. int8 on ANE is not consistently faster on A14/A15
  and sometimes hurts because of the extra dequant ops.
- Dynamic-range int8 on Android is the right default. Static int8 needs
  representative data and only wins if you can be careful with the
  calibration set.
- Never quantize the softmax attention weights in AttentiveStatsPool. Skip
  it with an exclude list in the ONNX quantizer.
