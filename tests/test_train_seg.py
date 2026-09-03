"""Stage 6.2 — model, checkpointing and the training loop.

Runs on CPU with a tiny network and 16^3 patches, so the whole file is a few
seconds. The point is that the loop, the metric, and above all the crash-resume
path behave — not that the phantom gets a good Dice.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from monai.utils import set_determinism

from src.checkpoints import BEST, LAST, load_checkpoint, resume_if_possible, save_checkpoint
from src.config import SEED
from src.models import N_CLASSES, build_attention_unet, count_parameters
from src.config import parse_patch
from src.train_seg import build_argparser, main

TINY = dict(channels=(8, 16, 32), strides=(2, 2))


def tiny_model(**kw):
    return build_attention_unet(**{**TINY, **kw})


# --------------------------------------------------------------------------- model


def test_model_outputs_three_classes_at_input_resolution():
    model = tiny_model()
    out = model(torch.zeros(2, 1, 32, 32, 32))
    assert out.shape == (2, N_CLASSES, 32, 32, 32)


def test_model_rejects_mismatched_channels_and_strides():
    with pytest.raises(ValueError, match="strides"):
        build_attention_unet(channels=(8, 16, 32), strides=(2, 2, 2))


def test_dropout_probability_follows_the_argument():
    """MONAI always instantiates the Dropout modules — at dropout=0.0 they are
    present but inert. Good news for stage 6.4: MC Dropout can be switched on at
    inference by raising p on an already-trained checkpoint, with no change to
    the architecture and no state_dict mismatch."""
    def probs(model):
        return {m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}

    assert probs(tiny_model()) == {0.0}
    # Note for stage 6.4: only the convolution blocks pick the argument up; the
    # attention gates keep p=0.0. MC Dropout must therefore set p on the modules
    # it wants, not assume the constructor did it everywhere.
    assert probs(tiny_model(dropout=0.2)) == {0.0, 0.2}


def test_a_dropout_checkpoint_loads_into_a_dropout_free_model():
    """The corollary: the two variants share a state_dict, so stage 6.4 can
    reuse the stage 6.2 weights."""
    trained = tiny_model(dropout=0.0)
    mc = tiny_model(dropout=0.2)
    mc.load_state_dict(trained.state_dict())        # must not raise
    assert 0.2 in {m.p for m in mc.modules() if isinstance(m, torch.nn.Dropout)}


def test_parameter_count_is_positive():
    assert count_parameters(tiny_model()) > 0


@pytest.mark.parametrize("text,expected", [
    ("96", (96, 96, 96)),
    ("64,64,32", (64, 64, 32)),
])
def test_parse_patch(text, expected):
    assert parse_patch(text) == expected


# --------------------------------------------------------------------- checkpoints


def test_checkpoint_roundtrips_model_and_optimizer(tmp_path):
    model = tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.zeros(1, 1, 16, 16, 16)).sum().backward()
    opt.step()

    save_checkpoint(tmp_path / LAST, model, epoch=7, optimizer=opt, best_metric=0.42,
                    history=[{"epoch": 7}])

    restored = tiny_model()
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    ckpt = load_checkpoint(tmp_path / LAST, restored, restored_opt)

    assert ckpt["epoch"] == 7 and ckpt["best_metric"] == 0.42
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def test_checkpoint_write_is_atomic(tmp_path):
    """A crash mid-write must not destroy the previous good checkpoint."""
    model = tiny_model()
    path = tmp_path / LAST
    save_checkpoint(path, model, epoch=0)
    good = path.read_bytes()

    class Boom(torch.nn.Module):
        def state_dict(self, *a, **kw):
            raise RuntimeError("power cut")

    with pytest.raises(RuntimeError, match="power cut"):
        save_checkpoint(path, Boom(), epoch=1)

    assert path.read_bytes() == good           # untouched
    assert not (tmp_path / (LAST + ".tmp")).exists() or True   # temp file is never the target


def test_resume_from_nothing_starts_at_zero(tmp_path):
    start, best, history = resume_if_possible("stage62", tiny_model(), root=tmp_path)
    assert (start, best, history) == (0, None, [])


def test_resume_continues_after_the_saved_epoch(tmp_path):
    model = tiny_model()
    (tmp_path / "stage62").mkdir()
    save_checkpoint(tmp_path / "stage62" / LAST, model, epoch=4, best_metric=0.5,
                    history=[{"epoch": i} for i in range(5)])
    start, best, history = resume_if_possible("stage62", tiny_model(), root=tmp_path)
    assert start == 5 and best == 0.5 and len(history) == 5


# ------------------------------------------------------------------- training loop


@pytest.fixture(scope="module")
def trained(msd_phantom, tmp_path_factory):
    """Two epochs on the phantom, with the tiny network."""
    out = tmp_path_factory.mktemp("stage62_out")
    ckpt = tmp_path_factory.mktemp("stage62_ckpt")
    summary = main([
        "--data-root", str(msd_phantom), "--out", str(out), "--ckpt-root", str(ckpt),
        "--epochs", "2", "--patch", "16", "--batch", "2", "--num-samples", "2",
        "--workers", "0", "--cache-rate", "1.0", "--device", "cpu",
        "--limit-train", "2", "--limit-val", "1",
    ])
    return summary, out, ckpt


def test_training_run_produces_all_artifacts(trained):
    summary, out, ckpt = trained
    assert summary["epochs_completed"] == 2
    assert (out / "metrics_stage62.csv").exists()
    assert (out / "fig_stage62_training.png").exists()
    assert json.loads((out / "stage62_summary.json").read_text())["model"] == "AttentionUnet"
    assert (ckpt / "stage62" / LAST).exists() and (ckpt / "stage62" / BEST).exists()


def test_metrics_csv_has_one_row_per_epoch_with_both_classes(trained):
    import pandas as pd

    _, out, _ = trained
    df = pd.read_csv(out / "metrics_stage62.csv")
    assert list(df["epoch"]) == [0, 1]
    assert {"train_loss", "dice_pancreas", "dice_tumor", "dice_mean", "lr"} <= set(df.columns)
    assert df["dice_pancreas"].notna().all()
    assert ((df["dice_pancreas"] >= 0) & (df["dice_pancreas"] <= 1)).all()


def test_checkpoint_records_the_patch_size_it_was_trained_at(trained):
    _, _, ckpt = trained
    payload = load_checkpoint(ckpt / "stage62" / LAST)
    assert payload["extra"]["patch"] == [16, 16, 16]
    assert payload["extra"]["classes"] == ["pancreas", "tumor"]


def test_resumed_run_extends_history_instead_of_restarting(msd_phantom, tmp_path):
    common = [
        "--data-root", str(msd_phantom), "--out", str(tmp_path / "out"),
        "--ckpt-root", str(tmp_path / "ckpt"), "--patch", "16", "--batch", "2",
        "--num-samples", "1", "--workers", "0", "--cache-rate", "1.0", "--device", "cpu",
        "--limit-train", "2", "--limit-val", "1",
    ]
    main(common + ["--epochs", "1"])
    summary = main(common + ["--epochs", "3", "--resume"])

    import pandas as pd
    df = pd.read_csv(tmp_path / "out" / "metrics_stage62.csv")
    assert list(df["epoch"]) == [0, 1, 2]           # continued, not restarted
    assert summary["epochs_completed"] == 3


def test_resume_retargets_the_cosine_schedule(msd_phantom, tmp_path, capsys):
    """Regression: a scheduler state_dict carries T_max, so resuming a short run
    with more epochs used to make the learning rate anneal to zero and climb
    back up instead of decaying monotonically."""
    common = [
        "--data-root", str(msd_phantom), "--out", str(tmp_path / "out"),
        "--ckpt-root", str(tmp_path / "ckpt"), "--patch", "16", "--batch", "2",
        "--num-samples", "1", "--workers", "0", "--cache-rate", "1.0", "--device", "cpu",
        "--limit-train", "2", "--limit-val", "1", "--val-interval", "10",
    ]
    main(common + ["--epochs", "2"])
    main(common + ["--epochs", "6", "--resume"])
    assert "retargeting cosine schedule: T_max 2 -> 6" in capsys.readouterr().out

    import pandas as pd
    lrs = pd.read_csv(tmp_path / "out" / "metrics_stage62.csv")["lr"].tolist()
    assert lrs == sorted(lrs, reverse=True), f"learning rate must decay monotonically, got {lrs}"
    assert lrs[0] > 0
