# Quantization

fp16 is the default weight precision on the export paths. int8 is
available for the extra size and (on some devices) extra speed.

The repo does not include an evaluation harness for wake-word accuracy
or speaker EER, and does not ship any measured before/after accuracy
numbers for the fp16 vs int8 paths.

## ONNX static PTQ

- Per-channel weight quantization, per-tensor activation quantization.
- `activation_type=QUInt8`, `weight_type=QInt8`.
- Calibration data should come from a real held-out set. The CLI
  fallback in `src/export/quantize.py` uses random normals via
  `_fake_calib`, which is only useful as a smoke test; a real
  calibration set will change quantized accuracy meaningfully.

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
`representative_dataset` yielding real mel clips.

## rules of thumb

- On the ANE, fp16 is the safe default. int8 on ANE is not consistently
  faster on A14 / A15 in the general case, and can pay for extra dequant
  ops.
- Dynamic-range int8 on Android is a reasonable default. Static int8
  needs representative data and only helps when the calibration set
  matches the deploy distribution.
- The softmax attention weights inside `AttentiveStatsPool` are worth
  excluding from an int8 pass. Skip them with an exclude list in the
  ONNX quantizer.
