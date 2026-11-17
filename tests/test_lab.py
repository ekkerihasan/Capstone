"""The lab-PC workflow: drop the tar in data/, run one command, Ctrl+C to pause,
run it again to resume - with the same epoch plan every session."""
import json
import tarfile

import pandas as pd
import pytest

import src.lab as lab
import src.train_seg as train_seg
from src.phantom import write_phantom_dataset
from src.train_seg import epochs_for_budget

FAST = ["--patch", "32", "--batch", "1", "--device", "cpu", "--no-amp", "--max-steps", "1",
        "--workers", "0", "--limit-train", "2", "--limit-val", "1"]


def test_epochs_for_budget():
    assert epochs_for_budget(13, 300) == 156          # 13 h at 5 min/epoch
    assert epochs_for_budget(0.001, 300) == 2         # never fewer than 2


@pytest.fixture()
def lab_env(tmp_path, monkeypatch):
    """A lab layout in tmp: data/ holding only the tar, empty checkpoints/results."""
    src_root = write_phantom_dataset(tmp_path / "build" / "Task07_Pancreas", n_cases=6)
    data = tmp_path / "data"
    data.mkdir()
    with tarfile.open(data / "Task07_Pancreas.tar", "w") as tf:
        tf.add(src_root, arcname="Task07_Pancreas")
    monkeypatch.setattr(lab, "DATA_DIR", data)
    monkeypatch.setattr(lab, "CKPT", tmp_path / "checkpoints" / "stage62")
    monkeypatch.setattr(lab, "OUT", tmp_path / "results" / "final")
    monkeypatch.setattr(lab, "SPLIT", tmp_path / "split.json")   # created on first run
    return tmp_path


def test_find_dataset_extracts_the_tar_once(lab_env):
    root = lab.find_dataset()
    assert (root / "dataset.json").exists()
    (lab_env / "data" / "Task07_Pancreas.tar").unlink()        # second call needs no tar
    assert lab.find_dataset() == root


def test_missing_dataset_says_where_to_put_it(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "DATA_DIR", tmp_path)
    with pytest.raises(SystemExit, match="Task07_Pancreas.tar"):
        lab.find_dataset()


def test_pause_then_resume_keeps_the_epoch_plan(lab_env, monkeypatch, capsys):
    """Session 1 is paused (Ctrl+C) in epoch 2; session 2 resumes and finishes
    with the epoch count decided in session 1."""
    real = train_seg.train_one_epoch
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(train_seg, "train_one_epoch", flaky)
    # 1 hour of budget at a few CPU-seconds per epoch would be hundreds of epochs; cap it
    monkeypatch.setattr(train_seg, "epochs_for_budget", lambda h, s, minimum=2: 3)
    lab.main(["--allow-cpu", "--hours", "1"] + FAST)
    out = capsys.readouterr().out
    assert "PAUSED. 1/3 epochs" in out

    run = json.loads((lab.CKPT / "run.json").read_text())
    assert run["epochs"] == 3 and run["hours"] == 1.0

    # session 2: a different --hours must not change the plan
    lab.main(["--allow-cpu", "--hours", "99"] + FAST)
    hist = pd.read_csv(lab.OUT / "metrics_stage62.csv")
    assert list(hist["epoch"]) == [0, 1, 2]                 # continued, not restarted
    assert hist["lr"].is_monotonic_decreasing               # one cosine schedule, no restart
    assert "3/3 epochs done" in lab.status()


def test_status_before_any_training(lab_env):
    assert "no training saved yet" in lab.status()


def test_backup_copies_checkpoints_and_metrics(lab_env, monkeypatch):
    monkeypatch.setattr(train_seg, "epochs_for_budget", lambda h, s, minimum=2: 2)
    dest = lab_env / "usb" / "capstone"
    (lab_env / "usb").mkdir()
    lab.main(["--allow-cpu", "--hours", "1", "--backup", str(dest)] + FAST)
    assert (dest / "checkpoints" / "stage62" / "last.pt").exists()
    assert (dest / "results_final" / "metrics_stage62.csv").exists()
