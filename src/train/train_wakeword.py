"""
train the wake-word conv net.

two data paths:
  - real:      Speech Commands v0.02 on disk (raw wav -> mel front-end)
  - synthetic: seeded synthetic mel-spectrograms, no download, CPU seconds

the synthetic path (--synthetic or WW_SYNTHETIC=1) is the default offline demo
so the whole train -> export -> onnxruntime loop runs with no dataset. the real
path stays selectable by passing --data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.wake_word_cnn import WakeWordCNN
from src.train.data_synthetic import SyntheticMelWW

# these labels are the real Speech Commands class set. they double as the class
# count for the synthetic path so exports match either way.
from src.train.data_speech_commands import LABELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default=None, help="path to speech_commands_v0.02/")
    p.add_argument(
        "--synthetic",
        action="store_true",
        default=os.environ.get("WW_SYNTHETIC", "") == "1",
        help="train on seeded synthetic mels (offline, no download)",
    )
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--out", default="runs/wakeword")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not args.synthetic and args.data is None:
        p.error("pass --data <speech_commands dir> or --synthetic")
    return args


def run_epoch(model, mel, loader, optim, loss_fn, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    it = tqdm(loader, desc="train" if train else "val", leave=False)
    for wav, y in it:
        wav = wav.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            # synthetic path already yields mels; real path yields raw wav
            m = mel(wav) if mel is not None else wav
            logits = model(m)
            loss = loss_fn(logits, y)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        it.set_postfix(loss=f"{loss.item():.3f}", acc=f"{correct/total:.3f}")
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    torch.manual_seed(cfg["train"]["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_frames = cfg["wake_word"]["n_frames"]
    n_mels = cfg["audio"]["n_mels"]

    if args.synthetic:
        print("data: synthetic mels (offline, no download)")
        train_ds = SyntheticMelWW(
            n_samples=1024, n_classes=len(LABELS),
            n_frames=n_frames, n_mels=n_mels, seed=1,
        )
        val_ds = SyntheticMelWW(
            n_samples=256, n_classes=len(LABELS),
            n_frames=n_frames, n_mels=n_mels, seed=7,
        )
        # synthetic samples are already mels, so no front-end and no workers
        mel = None
        num_workers = 0
    else:
        print(f"data: Speech Commands at {args.data}")
        from src.audio.mel import MelFrontend
        from src.train.data_speech_commands import SpeechCommandsWW

        train_ds = SpeechCommandsWW(args.data, split="train")
        val_ds = SpeechCommandsWW(args.data, split="val")
        mel = MelFrontend(
            sample_rate=cfg["audio"]["sample_rate"],
            n_fft=cfg["audio"]["n_fft"],
            win_ms=cfg["audio"]["window_ms"],
            hop_ms=cfg["audio"]["hop_ms"],
            n_mels=n_mels,
        ).to(args.device)
        num_workers = cfg["train"]["num_workers"]

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )

    model = WakeWordCNN(
        n_mels=cfg["audio"]["n_mels"],
        n_classes=len(LABELS),
        channels=cfg["wake_word"]["channels"],
        fc=cfg["wake_word"]["fc"],
    ).to(args.device)

    print(f"model params: {model.n_params()/1e3:.1f}K")

    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for e in range(epochs):
        tl, ta = run_epoch(model, mel, train_dl, optim, loss_fn, args.device, train=True)
        vl, va = run_epoch(model, mel, val_dl, optim, loss_fn, args.device, train=False)
        sched.step()
        print(f"epoch {e:2d}  train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}")
        if va > best:
            best = va
            torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "last.pt")
    print(f"best val acc: {best:.3f}")


if __name__ == "__main__":
    main()
