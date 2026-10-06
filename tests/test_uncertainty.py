"""Stage 6.4 — MC Dropout."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models import build_attention_unet
from src.uncertainty import (enable_dropout, mc_dropout_predict, mc_dropout_table,
                             predictive_entropy)


class Net(nn.Module):
    """Linear -> BatchNorm -> Dropout -> Linear, so both the 'dropout on' and
    the 'batchnorm stays eval' behaviours are observable."""

    def __init__(self, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.bn = nn.BatchNorm1d(16)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.bn(self.fc1(x)))))


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(6, 4)


def test_enable_dropout_reports_the_modules_and_leaves_batchnorm_in_eval():
    net = Net()
    names = enable_dropout(net)
    assert names == ["drop"]
    assert net.drop.training and not net.bn.training and not net.training


def test_enable_dropout_sets_p_explicitly():
    net = Net(p=0.0)
    enable_dropout(net, p=0.3)
    assert net.drop.p == 0.3


def test_mc_predict_returns_valid_probabilities_with_spread(x):
    torch.manual_seed(0)
    res = mc_dropout_predict(Net(p=0.5), x, T=20)
    assert res["mean"].shape == (6, 2) and res["std"].shape == (6, 2)
    assert res["samples"].shape == (20, 6, 2)
    assert np.allclose(res["mean"].sum(1), 1.0, atol=1e-5)
    assert (res["std"] > 0).any(), "dropout was on, the passes must differ"
    assert res["entropy"].shape == (6,) and (res["entropy"] >= 0).all()


def test_mc_predict_is_deterministic_when_p_is_zero(x):
    res = mc_dropout_predict(Net(p=0.5), x, T=5, p=0.0)    # p override wins
    assert np.allclose(res["std"], 0.0)
    assert np.allclose(res["samples"][0], res["samples"][-1])


def test_mc_predict_restores_eval_mode(x):
    net = Net()
    mc_dropout_predict(net, x, T=2)
    assert not net.drop.training and not net.training


def test_mc_predict_rejects_zero_passes(x):
    with pytest.raises(ValueError):
        mc_dropout_predict(Net(), x, T=0)


def test_attention_unet_dropout_can_be_switched_on_after_training():
    """The caveat from test_train_seg: a stage 6.2 checkpoint trained with
    dropout=0.0 still has Dropout modules, and enable_dropout(p=...) must set p
    on them itself because MONAI leaves the attention gates at 0.0."""
    model = build_attention_unet(channels=(8, 16, 32), strides=(2, 2), dropout=0.0)
    names = enable_dropout(model, p=0.2)
    assert len(names) > 0
    assert all(m.p == 0.2 for m in model.modules() if isinstance(m, nn.Dropout))


def test_predictive_entropy_is_zero_for_certainty_and_max_for_uniform():
    e = predictive_entropy(np.array([[1.0, 0.0], [0.5, 0.5]]))
    assert e[0] == pytest.approx(0.0, abs=1e-9)
    assert e[1] == pytest.approx(np.log(2))


def test_mc_table_matches_the_cls_predictions_contract(x):
    items = [{"case": f"pancreas_{i:03d}", "x": x[i], "y_true": i % 2, "group": "val",
              "eq_diam_cm": 1.2} for i in range(6)]
    torch.manual_seed(0)
    rows = mc_dropout_table(Net(), items, T=4, batch=4)
    assert len(rows) == 6
    assert set(rows[0]) >= {"case", "group", "y_true", "prob_tumor", "eq_diam_cm", "mc_std", "entropy"}
    assert all(0.0 <= r["prob_tumor"] <= 1.0 for r in rows)
    assert [r["case"] for r in rows] == [it["case"] for it in items]   # order preserved


# --------------------------------------------------- regressions from review


def test_mc_predict_restores_the_dropout_rates_it_overrode(x):
    """enable_dropout(model, p) assigns p permanently, which is right for an
    explicit call and wrong for an inference helper. Before this was fixed, a
    T=20 pass at p=0.5 left the model training at 0.5 afterwards."""
    model = Net()
    before = {n: m.p for n, m in model.named_modules() if isinstance(m, nn.Dropout)}
    assert before, "the fixture model must actually have dropout"
    mc_dropout_predict(model, x, T=3, p=0.5)
    after = {n: m.p for n, m in model.named_modules() if isinstance(m, nn.Dropout)}
    assert after == before
    assert not model.training


def test_explicit_enable_dropout_still_sets_p_permanently():
    """The restore belongs to mc_dropout_predict, not to enable_dropout: a
    caller who switches dropout on by hand means it."""
    model = Net()
    enable_dropout(model, p=0.33)
    assert all(m.p == 0.33 for m in model.modules() if isinstance(m, nn.Dropout))


def test_entropy_uses_the_class_axis_not_the_last_one():
    """[N, C] and [N, C, D, H, W] disagree about where the class axis is.
    Summing over the last one silently summed over W for a segmentation model."""
    probs = np.full((2, 3, 4, 5, 6), 1 / 3)
    assert predictive_entropy(probs, class_axis=1).shape == (2, 4, 5, 6)
    assert predictive_entropy(probs, class_axis=1) == pytest.approx(np.log(3))
    # default stays -1, so the classifier call sites are unchanged
    assert predictive_entropy(np.array([[0.5, 0.5]])).shape == (1,)


def test_mc_predict_on_a_segmentation_shape_gives_per_voxel_entropy():
    class Seg(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout3d(0.2)
            self.conv = nn.Conv3d(1, 3, 3, padding=1)

        def forward(self, v):
            return self.conv(self.drop(v))

    res = mc_dropout_predict(Seg(), torch.randn(2, 1, 8, 8, 8), T=3)
    assert res["mean"].shape == (2, 3, 8, 8, 8)
    assert res["entropy"].shape == (2, 8, 8, 8)
