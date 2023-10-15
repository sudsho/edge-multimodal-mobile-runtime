"""
Export to ONNX for ONNX Runtime Mobile on Android.

we ship opset 15 (works with ORT Mobile 1.16 and NNAPI). the exported graph
takes the mel spectrogram as input, front-end lives on the app side so it
can run against the audio thread directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from src.models.wake_word_cnn import WakeWordCNN
from src.models.speaker_ecapa_tiny import SpeakerECAPATiny


def build(which: str, cfg):
    if which == "wakeword":
        from src.train.data_speech_commands import LABELS
        net = WakeWordCNN(
            n_mels=cfg["audio"]["n_mels"],
            n_classes=len(LABELS),
            channels=cfg["wake_word"]["channels"],
            fc=cfg["wake_word"]["fc"],
        )
        shape = (1, 1, cfg["wake_word"]["n_frames"], cfg["audio"]["n_mels"])
    else:
        net = SpeakerECAPATiny(
            n_mels=cfg["audio"]["n_mels"],
            channels=cfg["speaker"]["channels"][-1],
            emb_dim=cfg["speaker"]["emb_dim"],
            attn_channels=cfg["speaker"]["attention_channels"],
        )
        shape = (1, 1, 200, cfg["audio"]["n_mels"])
    return net, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--which", choices=["wakeword", "speaker"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=15)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    net, shape = build(args.which, cfg)
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    net.eval()

    example = torch.randn(*shape)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.which == "speaker":
        # allow variable T for speaker embeddings
        dynamic_axes = {"mel": {2: "T"}, "emb": {0: "B"}}

    output_name = "logits" if args.which == "wakeword" else "emb"

    torch.onnx.export(
        net,
        example,
        args.out,
        input_names=["mel"],
        output_names=[output_name],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
