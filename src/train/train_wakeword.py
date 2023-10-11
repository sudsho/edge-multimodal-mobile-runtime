"""
train the wake-word conv net on Speech Commands v2.
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

from src.audio.mel import MelFrontend
from src.models.wake_word_cnn import WakeWordCNN
from src.train.data_speech_commands import SpeechCommandsWW, LABELS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", required=True, help="path to speech_commands_v0.02/")
    p.add_argument("--out", default="runs/wakeword")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def run_epoch(model, mel, loader, optim, loss_fn, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    it = tqdm(loader, desc="train" if train else "val", leave=False)
    for wav, y in it:
        wav = wav.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            m = mel(wav)
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

    train_ds = SpeechCommandsWW(args.data, split="train")
    val_ds = SpeechCommandsWW(args.data, split="val")

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
    )

    mel = MelFrontend(
        sample_rate=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        win_ms=cfg["audio"]["window_ms"],
        hop_ms=cfg["audio"]["hop_ms"],
        n_mels=cfg["audio"]["n_mels"],
    ).to(args.device)

    model = WakeWordCNN(
        n_mels=cfg["audio"]["n_mels"],
        n_classes=len(LABELS),
        channels=cfg["wake_word"]["channels"],
        fc=cfg["wake_word"]["fc"],
    ).to(args.device)

    print(f"model params: {model.n_params()/1e3:.1f}K")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["train"]["epochs"])
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for e in range(cfg["train"]["epochs"]):
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
