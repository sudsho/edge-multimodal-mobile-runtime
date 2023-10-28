.PHONY: install test lint train-ww train-spk export-coreml export-onnx export-tflite quantize bench clean

PY ?= python
DATA_WW ?= data/speech_commands_v0.02
DATA_SPK ?= data/voxceleb1/wav

install:
	pip install -r requirements.txt

test:
	pytest -q

lint:
	python -m compileall src tests

train-ww:
	$(PY) -m src.train.train_wakeword --data $(DATA_WW) --out runs/wakeword

train-spk:
	$(PY) -m src.train.train_speaker --data $(DATA_SPK) --out runs/speaker

export-coreml:
	$(PY) -m src.export.to_coreml --which wakeword --ckpt runs/wakeword/best.pt --out models/wakeword.mlpackage
	$(PY) -m src.export.to_coreml --which speaker --ckpt runs/speaker/speaker_last.pt --out models/speaker.mlpackage

export-onnx:
	$(PY) -m src.export.to_onnx --which wakeword --ckpt runs/wakeword/best.pt --out models/wakeword.onnx
	$(PY) -m src.export.to_onnx --which speaker --ckpt runs/speaker/speaker_last.pt --out models/speaker.onnx

export-tflite:
	$(PY) -m src.export.to_tflite --onnx models/wakeword.onnx --out models/wakeword.tflite

quantize:
	$(PY) -m src.export.quantize --which onnx-dynamic --src models/wakeword.onnx --dst models/wakeword.int8.onnx
	$(PY) -m src.export.quantize --which coreml --src models/wakeword.mlpackage --dst models/wakeword.int8.mlpackage

bench:
	$(PY) -m src.bench.latency_probe --runtime onnx --model models/wakeword.onnx --which wakeword

clean:
	rm -rf runs models build dist __pycache__
