"""
train the tiny ECAPA-TDNN speaker embedding on a VoxCeleb1 subset.

we train it as a closed-set classifier with AAM-softmax, then throw the
classifier away at export time and keep the L2-normalized embedding.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.audio.mel import MelFrontend
from src.models.speaker_ecapa_tiny import SpeakerECAPATiny
from src.train.data_voxceleb import VoxCeleb1Subset


class AAMSoftmax(nn.Module):
    def __init__(self, emb_dim: int, n_classes: int, margin: float = 0.2, scale: float = 30.0):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_normal_(self.W)
        self.m = margin
        self.s = scale

    def forward(self, e: torch.Tensor, y: torch.Tensor):
        w = F.normalize(self.W, dim=-1)
        cos = F.linear(e, w).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.zeros_like(cos).scatter_(1, y.unsqueeze(1), 1.0)
        cos_m = torch.cos(theta + self.m)
        logits = self.s * (target * cos_m + (1 - target) * cos)
        return logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", required=True, help="path to voxceleb1 wav/ dir")
    p.add_argument("--out", default="runs/speaker")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.manual_seed(cfg["train"]["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = VoxCeleb1Subset(args.data, sample_rate=cfg["audio"]["sample_rate"])
    dl = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], drop_last=True,
    )

    mel = MelFrontend(
        sample_rate=cfg["audio"]["sample_rate"],
        n_fft=cfg["audio"]["n_fft"],
        win_ms=cfg["audio"]["window_ms"],
        hop_ms=cfg["audio"]["hop_ms"],
        n_mels=cfg["audio"]["n_mels"],
    ).to(args.device)

    net = SpeakerECAPATiny(
        n_mels=cfg["audio"]["n_mels"],
        channels=cfg["speaker"]["channels"][-1],
        emb_dim=cfg["speaker"]["emb_dim"],
        attn_channels=cfg["speaker"]["attention_channels"],
    ).to(args.device)
    head = AAMSoftmax(cfg["speaker"]["emb_dim"], ds.n_speakers).to(args.device)

    optim = torch.optim.AdamW(
        list(net.parameters()) + list(head.parameters()),
        lr=cfg["train"]["lr"], weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["train"]["epochs"])
    ce = nn.CrossEntropyLoss()

    for e in range(cfg["train"]["epochs"]):
        net.train(); head.train()
        loss_sum, correct, total = 0.0, 0, 0
        for wav, y in tqdm(dl, desc=f"epoch {e}", leave=False):
            wav = wav.to(args.device); y = y.to(args.device)
            m = mel(wav)
            emb = net(m)
            logits = head(emb, y)
            loss = ce(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
        sched.step()
        print(f"epoch {e}  loss {loss_sum/total:.3f}  acc {correct/total:.3f}")
        torch.save(net.state_dict(), out / f"speaker_e{e}.pt")

    torch.save(net.state_dict(), out / "speaker_last.pt")


if __name__ == "__main__":
    main()
