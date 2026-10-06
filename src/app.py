"""Stage 6.6: the demo UI.

    python -m src.app                      # http://127.0.0.1:7860
    python -m src.app --share              # public link, for showing it remotely

A volume goes in; a segmentation mask, a calibrated probability, a conformal
prediction set and one of three verdicts come out - tumor present, no tumor, or
FLAGGED FOR SPECIALIST REVIEW. The abstention rule itself lives in
src/selective.py, deliberately apart from the UI.

What this app is honest about, on screen, every time:

- the segmentation weights are whatever checkpoint it found, which today is a
  two-epoch phantom run, not a trained model;
- there is no stage 6.3 classifier yet, so p(tumor) is read off the
  segmentation model's own tumor channel rather than a dedicated classifier;
- stage 6.5 does not exist, so the fidelity signal is absent rather than passing;
- the data is a phantom.

Every one of those is shown in the provenance panel next to the verdict. The app
is the pipeline's wiring made visible, not evidence about pancreatic cancer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from .config import CHECKPOINT_DIR, DATA_DIR, LABEL_PANCREAS, LABEL_TUMOR, RESULTS_DIR
from .phantom import DEMO_STAMP
from .selective import ABSTAIN, decide, label_of

TUMOR_CHANNEL = 2           # 0 background / 1 pancreas / 2 tumor
PATCH = (32, 32, 32)        # inference window; must divide down through the U-Net depth
TOP_K = 32                  # voxels averaged to turn a mask into one probability
MC_PASSES = 20              # T in CLAUDE.md
DEFAULT_CKPT = RESULTS_DIR / "review2_demo" / "checkpoints" / "stage62" / "best.pt"
DEFAULT_Q_HAT = RESULTS_DIR / "review2_demo" / "stage64_summary.json"
DEFAULT_CURVE = RESULTS_DIR / "review2_demo" / "coverage_curve.csv"


# ----------------------------------------------------------------- model layer

def find_checkpoint(explicit: str = "") -> Path | None:
    for c in (Path(explicit) if explicit else None,
              DEFAULT_CKPT,
              CHECKPOINT_DIR / "stage62" / "best.pt"):
        if c and c.exists():
            return c
    return None


def load_model(checkpoint: Path | None):
    """The stage 6.2 Attention U-Net, with the checkpoint if there is one.
    Without one the app still runs - on random weights, which the UI says."""
    from .models import build_attention_unet

    model = build_attention_unet()
    trained = False
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
        trained = True
    model.eval()
    return model, trained


def training_patch(summary: Path = RESULTS_DIR / "review2_demo" / "stage62_summary.json"):
    """Match the window to whatever the checkpoint was trained with - a model
    trained on 32^3 patches and evaluated on 96^3 ones is a different model."""
    if summary.exists():
        patch = json.loads(summary.read_text(encoding="utf-8")).get("patch")
        if patch:
            return tuple(patch)
    return PATCH


def load_q_hat(path: Path = DEFAULT_Q_HAT):
    """The conformal threshold calibrated in stage 6.4. Without it the app shows
    no prediction set and the abstention rule simply loses that signal."""
    if not path.exists():
        return None, None
    d = json.loads(path.read_text(encoding="utf-8")).get("conformal", {})
    return d.get("q_hat"), d.get("alpha")


def load_coverage_curve(path: Path = DEFAULT_CURVE):
    """alpha -> q_hat, as calibrated in stage 6.4.

    Lets the UI move the target coverage and show the trade-off live: tighten it
    and more cases come back as a two-label set, which is an abstention. The
    q_hat for each alpha was calibrated on the held-out calibration group, so
    moving the slider is not re-fitting anything.
    """
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_csv(path)
    return sorted(zip(df["alpha"], df["q_hat"]))


def q_hat_for(curve, alpha: float, fallback=None):
    if not curve:
        return fallback
    return min(curve, key=lambda r: abs(r[0] - alpha))[1]


def _tumor_probability(logits) -> float:
    """One number from a segmentation output: the mean of the TOP_K highest
    tumor-channel probabilities.

    A plain max is one hot voxel away from 1.0 and a plain mean is ~0 because
    almost every voxel is background. Averaging the top K is the compromise, and
    K is fixed so two cases are comparable.
    """
    p = torch.softmax(logits, dim=1)[0, TUMOR_CHANNEL].flatten()
    k = min(TOP_K, p.numel())
    return float(torch.topk(p, k).values.mean())


@torch.no_grad()
def predict(model, x, mc_passes: int = MC_PASSES, dropout_p: float = 0.1, patch=PATCH):
    """Mask, probability and MC Dropout spread for one volume.

    The spread is taken over the *scalar* probability of each pass rather than
    voxel-wise, which also sidesteps uncertainty.predictive_entropy assuming the
    class axis is last - true for a classifier's [N, C], not for a segmentation
    model's [N, C, D, H, W].
    """
    from monai.inferers import sliding_window_inference

    from .uncertainty import enable_dropout

    # Whole preprocessed volumes are not divisible by the U-Net's depth, so the
    # app infers the same way validation does: a sliding window, which pads.
    def forward(v):
        return sliding_window_inference(v, patch, 2, model, overlap=0.25)

    logits = forward(x)
    mask = logits.argmax(dim=1)[0].cpu().numpy()
    prob = _tumor_probability(logits)

    mc_std = None
    if mc_passes > 1:
        originals = {n: m.p for n, m in model.named_modules() if hasattr(m, "p")}
        enable_dropout(model, dropout_p)
        try:
            draws = [_tumor_probability(forward(x)) for _ in range(mc_passes)]
        finally:
            model.eval()
            for n, m in model.named_modules():     # restore, see the note in selective.py
                if n in originals:
                    m.p = originals[n]
        mc_std = float(np.std(draws))
    return mask, prob, mc_std


# ------------------------------------------------------------------ case layer

def list_cases(root: Path) -> list:
    images = sorted((root / "imagesTr").glob("pancreas_*.nii.gz"))
    return [p.name for p in images if not p.name.startswith("._")]


def load_volume(image_path: Path):
    """Stage 6.1 preprocessing, exactly as the training pipeline applies it."""
    from .transforms import build_transforms

    label_path = image_path.parent.parent / "labelsTr" / image_path.name
    item = {"image": str(image_path)}
    if label_path.exists():
        item["label"] = str(label_path)
    d = build_transforms(train=False, keys=list(item))(item)
    x = torch.as_tensor(np.asarray(d["image"]))[None]
    gt = np.asarray(d["label"])[0] if "label" in d else None
    return x, gt


def overlay_figure(image, pred, gt, title: str):
    import matplotlib.pyplot as plt

    from .phantom import stamp_figure
    from .viz import best_axial_slice

    img = np.asarray(image)[0, 0]
    ref = gt if gt is not None else pred
    z = best_axial_slice(ref)
    panels = [("preprocessed CT", None), ("predicted mask", pred)]
    if gt is not None:
        panels.append(("ground truth", gt))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, mask) in zip(axes, panels):
        ax.imshow(np.rot90(img[:, :, z]), cmap="gray")
        if mask is not None:
            m = np.rot90(np.asarray(mask)[:, :, z])
            ax.contour(m >= LABEL_PANCREAS, colors="lime", linewidths=0.9)
            ax.contour(m == LABEL_TUMOR, colors="red", linewidths=0.9)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{title}  |  axial slice z={z}", fontsize=11)
    fig.tight_layout()
    stamp_figure(fig)
    return fig


# -------------------------------------------------------------------- analysis

def analyse(image_path: Path, model, trained: bool, q_hat, alpha,
            mc_passes: int = MC_PASSES, patch=PATCH):
    x, gt = load_volume(image_path)
    mask, prob, mc_std = predict(model, x, mc_passes=mc_passes, patch=patch)

    conformal_set = None
    if q_hat is not None:
        probs = [1.0 - prob, prob]
        conformal_set = [c for c, p in enumerate(probs) if (1.0 - p) <= q_hat]

    verdict = decide(prob, conformal_set=conformal_set, mc_std=mc_std,
                     fidelity_pass=None)          # stage 6.5 does not exist yet
    fig = overlay_figure(x, mask, gt, image_path.name.replace(".nii.gz", ""))
    return fig, verdict, gt, trained, alpha


def report_markdown(verdict: dict, gt, trained: bool, alpha, mc_passes: int) -> str:
    d = verdict["decision"]
    colour = {"tumor": "#d1495b", "no_tumor": "#2a9d8f", ABSTAIN: "#e09f3e"}[d]
    lines = [
        f"<h2 style='color:{colour};margin:0'>{label_of(d)}</h2>",
        "",
        f"**p(tumor)** {verdict['prob_tumor']:.3f}",
    ]
    if verdict["mc_std"] is not None:
        lines.append(f"**MC Dropout spread** {verdict['mc_std']:.3f} over T={mc_passes} passes")
    if verdict["conformal_set"] is not None:
        names = {0: "no tumor", 1: "tumor"}
        inside = ", ".join(names[c] for c in verdict["conformal_set"]) or "(empty)"
        lines.append(f"**Conformal set** at {int((1 - (alpha or 0.1)) * 100)} % target "
                     f"coverage: {{{inside}}}")
    if verdict["reasons"]:
        lines += ["", "**Why it was escalated**"] + [f"- {r}" for r in verdict["reasons"]]
    if gt is not None:
        truth = "tumor" if (np.asarray(gt) == LABEL_TUMOR).any() else "no tumor"
        lines += ["", f"*Phantom ground truth: {truth}.*"]

    lines += [
        "", "---", "",
        "**Provenance**",
        f"- segmentation weights: {'checkpoint loaded' if trained else 'UNTRAINED (random)'}",
        "- p(tumor): read off the segmentation tumor channel; **no stage 6.3 classifier exists yet**",
        "- fidelity signal: **absent**, stage 6.5 not built",
        f"- data: synthetic phantom. {DEMO_STAMP}",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------------------- UI

def build_ui(data_root: Path, checkpoint: str = "", mc_passes: int = MC_PASSES):
    import gradio as gr

    ckpt = find_checkpoint(checkpoint)
    model, trained = load_model(ckpt)
    q_hat, alpha = load_q_hat()
    curve = load_coverage_curve()
    patch = training_patch()
    cases = list_cases(data_root)

    banner = (
        f"### Pancreatic cancer early detection — stages 6.1 → 6.6\n"
        f"**{DEMO_STAMP}**\n\n"
        f"Segmentation checkpoint: `{ckpt if ckpt else 'none — random weights'}` · "
        f"conformal q̂: `{q_hat if q_hat is not None else 'not calibrated'}` · "
        f"MC Dropout T={mc_passes} · window {patch}"
    )

    def run(case_name, uploaded, target_alpha):
        path = Path(uploaded) if uploaded else (data_root / "imagesTr" / case_name)
        if not path.exists():
            return None, f"**No such case:** `{path}`"
        q = q_hat_for(curve, target_alpha, fallback=q_hat)
        fig, verdict, gt, is_trained, _ = analyse(path, model, trained, q, target_alpha,
                                                  mc_passes=mc_passes, patch=patch)
        return fig, report_markdown(verdict, gt, is_trained, target_alpha, mc_passes)

    with gr.Blocks(title="Pancreatic cancer early detection") as ui:
        gr.Markdown(banner)
        with gr.Row():
            with gr.Column(scale=1):
                case = gr.Dropdown(cases, value=cases[0] if cases else None,
                                   label="Phantom case")
                upload = gr.File(label="…or upload a NIfTI volume (.nii.gz)",
                                 file_types=[".gz", ".nii"], type="filepath")
                alpha_in = gr.Slider(0.02, 0.5, value=alpha or 0.1, step=0.02,
                                     label="conformal alpha (1 - target coverage)",
                                     info="tighter coverage -> larger sets -> more escalations")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                out_md = gr.Markdown()
        out_fig = gr.Plot(label="Segmentation")
        go.click(run, inputs=[case, upload, alpha_in], outputs=[out_fig, out_md])
    return ui


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=str(DATA_DIR / "phantom"))
    p.add_argument("--checkpoint", default="")
    p.add_argument("--mc-passes", type=int, default=MC_PASSES)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — run `python -m src.run_demo_pipeline` first")
    ui = build_ui(root, args.checkpoint, args.mc_passes)
    ui.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
