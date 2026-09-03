"""Crash-safe checkpointing.

Overnight access to the lab PC is not guaranteed, so every training loop in this
project writes `last.pt` after each epoch and can resume from it. Writes go to a
temporary file first and are then renamed: a power cut mid-write leaves the
previous good checkpoint intact instead of a truncated one.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .config import CHECKPOINT_DIR

LAST = "last.pt"
BEST = "best.pt"


def stage_dir(stage: str, root=CHECKPOINT_DIR) -> Path:
    d = Path(root) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_checkpoint(path, model, epoch: int, optimizer=None, scaler=None,
                    scheduler=None, best_metric=None, history=None, extra=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history": history or [],
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic on Windows and POSIX alike
    return path


def load_checkpoint(path, model=None, optimizer=None, scaler=None, scheduler=None,
                    map_location="cpu", strict: bool = True) -> dict:
    """Restore into whichever objects are passed and return the raw payload."""
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(ckpt["model"], strict=strict)
    for obj, key in ((optimizer, "optimizer"), (scaler, "scaler"), (scheduler, "scheduler")):
        if obj is not None and ckpt.get(key):
            obj.load_state_dict(ckpt[key])
    return ckpt


def resume_if_possible(stage: str, model, optimizer=None, scaler=None, scheduler=None,
                       root=CHECKPOINT_DIR, map_location="cpu"):
    """Returns (start_epoch, best_metric, history). Starts from scratch when
    there is no `last.pt` yet."""
    path = Path(root) / stage / LAST
    if not path.exists():
        return 0, None, []
    ckpt = load_checkpoint(path, model, optimizer, scaler, scheduler, map_location=map_location)
    print(f"[resume] {path} at epoch {ckpt['epoch']} (best={ckpt.get('best_metric')})")
    return ckpt["epoch"] + 1, ckpt.get("best_metric"), ckpt.get("history", [])
