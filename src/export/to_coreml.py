"""
export a trained wake-word or speaker model to CoreML via coremltools 7.

we target iOS 16 (Neural Engine + ML Program) with fp16 weights. this is the
recommended path in the coremltools 7 docs and gets us onto the ANE cleanly.

usage:
    python -m src.export.to_coreml \
        --which wakeword \
        --ckpt runs/wakeword/best.pt \
        --out models/wakeword.mlpackage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

import coremltools as ct
from coremltools import ImageType, TensorType

from src.audio.mel import MelFrontend
from src.models.wake_word_cnn import WakeWordCNN
from src.models.speaker_ecapa_tiny import SpeakerECAPATiny


def build_wakeword(cfg):
    from src.train.data_speech_commands import LABELS
    net = WakeWordCNN(
        n_mels=cfg["audio"]["n_mels"],
        n_classes=len(LABELS),
        channels=cfg["wake_word"]["channels"],
        fc=cfg["wake_word"]["fc"],
    )
    return net, (1, 1, cfg["wake_word"]["n_frames"], cfg["audio"]["n_mels"])


def build_speaker(cfg):
    net = SpeakerECAPATiny(
        n_mels=cfg["audio"]["n_mels"],
        channels=cfg["speaker"]["channels"][-1],
        emb_dim=cfg["speaker"]["emb_dim"],
        attn_channels=cfg["speaker"]["attention_channels"],
    )
    # speaker takes ~2s -> 200 frames at 10ms hop
    return net, (1, 1, 200, cfg["audio"]["n_mels"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--which", choices=["wakeword", "speaker"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    if args.which == "wakeword":
        net, shape = build_wakeword(cfg)
    else:
        net, shape = build_speaker(cfg)

    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd)
    net.eval()

    example = torch.randn(*shape)
    traced = torch.jit.trace(net, example)

    inputs = [TensorType(name="mel", shape=shape)]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )

    mlmodel.author = "sudsho"
    mlmodel.short_description = f"{args.which} model, edge-multimodal-mobile-runtime"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
