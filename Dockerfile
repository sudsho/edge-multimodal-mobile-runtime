# training / export container. deploy is on-device, this image is not
# meant to run in production.

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# coremltools needs a mac to run predictions but converts fine on linux.
# tflite export path (onnx_tf + tensorflow) is optional and installed on demand.

COPY . .

ENV PYTHONPATH=/workspace

CMD ["python", "-m", "src.train.train_wakeword", "--help"]
