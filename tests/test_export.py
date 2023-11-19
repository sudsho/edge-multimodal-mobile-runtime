"""
smoke tests for the ONNX export path.

CoreML and TFLite tests need mac and TF respectively so we skip if missing.
"""

import os
import tempfile

import pytest
import torch

from src.models.wake_word_cnn import WakeWordCNN


def test_wakeword_onnx_export_smoke():
    net = WakeWordCNN(n_mels=40, n_classes=12)
    net.eval()
    example = torch.randn(1, 1, 98, 40)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "wakeword.onnx")
        torch.onnx.export(
            net, example, out,
            input_names=["mel"], output_names=["logits"],
            opset_version=15, do_constant_folding=True,
        )
        assert os.path.getsize(out) > 0
        try:
            import onnx
            model = onnx.load(out)
            onnx.checker.check_model(model)
        except ImportError:
            pytest.skip("onnx not installed in this env")


def test_wakeword_onnx_runtime_matches_torch():
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnx / onnxruntime not installed")

    net = WakeWordCNN(n_mels=40, n_classes=12)
    net.eval()
    example = torch.randn(1, 1, 98, 40)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "wakeword.onnx")
        torch.onnx.export(net, example, out, input_names=["mel"],
                          output_names=["logits"], opset_version=15)
        with torch.no_grad():
            y_torch = net(example).numpy()
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        y_onnx = sess.run(None, {"mel": example.numpy()})[0]
        assert y_torch.shape == y_onnx.shape
        assert (abs(y_torch - y_onnx).max()) < 1e-4
