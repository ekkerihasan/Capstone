"""Stage 6.6: the demo UI.

    python -m src.app                      # http://127.0.0.1:7860
    python -m src.app --share              # public link, for showing it remotely

A volume goes in; a segmentation mask, a calibrated probability, a conformal
prediction set and one of three verdicts come out - tumor present, no tumor, or
FLAGGED FOR SPECIALIST REVIEW. The abstention rule itself lives in
src/selective.py, deliberately apart from the UI.

    python -m src.app --data-root data/Task07_Pancreas     # real MSD, trained weights

Two ways to get p(tumor), chosen by what exists on disk:

- Stage 6.3 classifier (checkpoints/stage63/best.pt, or --classifier): the 6.2
  mask locates the pancreas, the classifier's window slides over it at half-
  window stride, and p(tumor) is the highest-scoring window - with MC Dropout,
  Grad-CAM++ and F-Fidelity computed on that window. This is the real pipeline.
- Fallback, no classifier: p(tumor) is read off the segmentation model's own
  tumor channel, as in the Review 2 demo.

What this app is honest about, on screen, every time: which weights were
loaded (or that they are untrained), which of the two routes produced p(tumor),
where q-hat was calibrated, and whether the data is a phantom.

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

from .config import CHECKPOINT_DIR, DATA_DIR, LABEL_PANCREAS, LABEL_TUMOR, MSD_DIR, RESULTS_DIR
from .phantom import DEMO_STAMP
from .selective import ABSTAIN, decide, label_of

TUMOR_CHANNEL = 2           # 0 background / 1 pancreas / 2 tumor
PATCH = (32, 32, 32)        # inference window; must divide down through the U-Net depth
TOP_K = 32                  # voxels averaged to turn a mask into one probability
MC_PASSES = 20              # T in CLAUDE.md
DEFAULT_CKPT = RESULTS_DIR / "review2_demo" / "checkpoints" / "stage62" / "best.pt"
DEFAULT_Q_HAT = RESULTS_DIR / "review2_demo" / "stage64_summary.json"
DEFAULT_CURVE = RESULTS_DIR / "review2_demo" / "coverage_curve.csv"
DEFAULT_CLS = CHECKPOINT_DIR / "stage63" / "best.pt"
# where stage 6.4 put its calibration, best first: real end-to-end, then oracle
CALIBRATION_DIRS = (RESULTS_DIR / "final" / "real", RESULTS_DIR / "final" / "oracle")
MIN_PANCREAS_FRAC = 0.01    # a window must be >= 1 % pancreas to be classified


# ----------------------------------------------------------------- model layer

def find_checkpoint(explicit: str = "") -> Path | None:
    for c in (Path(explicit) if explicit else None,
              CHECKPOINT_DIR / "stage62" / "best.pt",
              DEFAULT_CKPT):
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


def checkpoint_patch(checkpoint: Path | None):
    """The training window recorded inside a stage 6.2 checkpoint, if any."""
    if checkpoint is None:
        return None
    extra = torch.load(checkpoint, map_location="cpu", weights_only=False).get("extra", {})
    return tuple(extra["patch"]) if extra.get("patch") else None


def find_classifier(explicit: str = "") -> Path | None:
    for c in (Path(explicit) if explicit else None, DEFAULT_CLS):
        if c and c.exists():
            return c
    return None


def load_classifier(checkpoint: Path):
    """The stage 6.3 classifier and its training config (model, roi_size, dropout)."""
    from .classifier import build_classifier

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("extra", {}))
    model = build_classifier(cfg.get("model", "densenet121"), dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.eval()
    cfg["roi_size"] = tuple(cfg.get("roi_size", (32, 32, 32)))
    cfg["path"] = str(checkpoint)
    return model, cfg


def calibration_dir(classifier_present: bool) -> Path:
    """Use stage 6.4's real calibration when the classifier is the one in use."""
    if classifier_present:
        for d in CALIBRATION_DIRS:
            if (d / "stage64_summary.json").exists():
                return d
    return DEFAULT_Q_HAT.parent


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


def pancreas_windows(mask, roi_size, min_frac: float = MIN_PANCREAS_FRAC) -> list:
    """Window corners tiling the pancreas bbox at half-window stride, keeping
    those that are at least `min_frac` pancreas. Half stride means any lesion
    sits in the central half of some window, as the training positives did."""
    from .roi import pancreas_bbox

    mask = np.asarray(mask)
    if not (mask >= LABEL_PANCREAS).any():
        return []
    bbox = pancreas_bbox(mask, margin_mm=0.0)
    size = np.asarray(roi_size, dtype=int)
    axes = []
    for sl, n in zip(bbox, size):
        lo, hi = sl.start, sl.stop
        first = lo - n // 4
        last = max(first, hi - 3 * n // 4)
        stride = max(1, n // 2)
        starts = list(range(first, last + 1, stride))
        if starts[-1] < last:
            starts.append(last)
        axes.append(starts)
    need = min_frac * float(np.prod(size))
    keep = []
    for a in axes[0]:
        for b in axes[1]:
            for c in axes[2]:
                win = mask[max(a, 0):a + size[0], max(b, 0):b + size[1], max(c, 0):c + size[2]]
                if (win >= LABEL_PANCREAS).sum() >= need:
                    keep.append((a, b, c))
    return keep


@torch.no_grad()
def classify_volume(classifier, cfg: dict, image, mask, mc_passes: int = MC_PASSES,
                    batch: int = 16):
    """Stage 6.3 on a whole volume: classify every pancreas window, keep the
    highest. Returns None when the mask has no pancreas to look in."""
    from .classifier import tumor_probability
    from .roi import crop_at
    from .uncertainty import mc_dropout_predict

    roi = cfg["roi_size"]
    starts = pancreas_windows(mask, roi)
    if not starts:
        return None
    img = np.asarray(image, dtype=np.float32)
    xs = torch.from_numpy(np.stack([crop_at(img, s, roi)[None] for s in starts]))
    probs = torch.cat([tumor_probability(classifier, xs[i:i + batch])
                       for i in range(0, len(xs), batch)]).numpy()
    i = int(probs.argmax())
    best = xs[i:i + 1]
    prob, mc_std = float(probs[i]), None
    if mc_passes > 1 and cfg.get("dropout", 0.0) > 0:
        r = mc_dropout_predict(classifier, best, T=mc_passes, p=cfg["dropout"])
        prob, mc_std = float(r["mean"][0, 1]), float(r["std"][0, 1])
    return {"prob": prob, "mc_std": mc_std, "start": starts[i], "x": best,
            "n_windows": len(starts), "window_probs": probs}


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


def overlay_figure(image, pred, gt, title: str, stamp: bool = True, box=None):
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
        if box is not None and box[2][0] <= z < box[2][1]:
            (x0, x1), (y0, y1) = box[0], box[1]
            w = img.shape[1]
            # np.rot90 shows element (i, j) at row w-1-j, column i
            ax.add_patch(plt.Rectangle((x0 - 0.5, w - y1 - 0.5), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="yellow", linewidth=1.2, linestyle="--"))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{title}  |  axial slice z={z}", fontsize=11)
    fig.tight_layout()
    if stamp:
        stamp_figure(fig)
    return fig


# -------------------------------------------------------------------- analysis

def explain_case(explainer, x, k_fractions=(0.01, 0.05, 0.10), n_random: int = 3):
    """Stage 6.5 for one case: Grad-CAM++, then the masking test that says
    whether the map is worth showing. Fewer k and fewer random masks than the
    offline default, because this runs while somebody is watching."""
    from .explain import gradcam_plusplus
    from .fidelity import fidelity_test

    cam, cls, _ = gradcam_plusplus(explainer, x)
    fid = fidelity_test(explainer, x, cam, class_idx=cls,
                        k_fractions=k_fractions, n_random=n_random)
    return cam, fid


def analyse(image_path: Path, model, trained: bool, q_hat, alpha,
            mc_passes: int = MC_PASSES, patch=PATCH, explainer=None,
            classifier=None, cls_cfg=None, want_saliency: bool = False, phantom: bool = True):
    if classifier is not None:
        return analyse_with_classifier(image_path, model, trained, q_hat, alpha, classifier,
                                       cls_cfg, mc_passes, patch, want_saliency, phantom)
    x, gt = load_volume(image_path)
    mask, prob, mc_std = predict(model, x, mc_passes=mc_passes, patch=patch)

    conformal_set = None
    if q_hat is not None:
        probs = [1.0 - prob, prob]
        conformal_set = [c for c, p in enumerate(probs) if (1.0 - p) <= q_hat]

    cam = fid = None
    if explainer is not None:
        cam, fid = explain_case(explainer, x)

    verdict = decide(prob, conformal_set=conformal_set, mc_std=mc_std,
                     fidelity_pass=None if fid is None else fid["passed"])
    verdict["saliency"] = cam
    verdict["fidelity"] = fid
    verdict["phantom"] = phantom
    fig = overlay_figure(x, mask, gt, image_path.name.replace(".nii.gz", ""), stamp=phantom)
    return fig, verdict, gt, trained, alpha


def analyse_with_classifier(image_path: Path, model, trained: bool, q_hat, alpha,
                            classifier, cls_cfg: dict, mc_passes: int = MC_PASSES,
                            patch=PATCH, want_saliency: bool = False, phantom: bool = False):
    """The real pipeline: 6.2 mask -> 6.3 windows -> 6.4 MC/conformal ->
    6.5 Grad-CAM++/F-Fidelity on the deciding window -> 6.6 rule."""
    from .explain import gradcam_plusplus
    from .fidelity import fidelity_test

    x, gt = load_volume(image_path)
    mask, _, _ = predict(model, x, mc_passes=0, patch=patch)
    res = classify_volume(classifier, cls_cfg, np.asarray(x)[0, 0], mask, mc_passes=mc_passes)
    name = image_path.name.replace(".nii.gz", "")

    if res is None:                       # nothing to classify: never guess
        verdict = decide(0.5)
        verdict.update({"decision": ABSTAIN, "abstain": True,
                        "reasons": ["segmentation found no pancreas to examine"],
                        "saliency": None, "fidelity": None, "source": "classifier",
                        "classifier": cls_cfg, "phantom": phantom, "window": None,
                        "n_windows": 0})
        return overlay_figure(x, mask, gt, name, stamp=phantom), verdict, gt, trained, alpha

    prob = res["prob"]
    conformal_set = None
    if q_hat is not None:
        probs = [1.0 - prob, prob]
        conformal_set = [c for c, p in enumerate(probs) if (1.0 - p) <= q_hat]

    cam = fid = None
    if want_saliency:
        cam, cls, _ = gradcam_plusplus(classifier, res["x"])
        fid = fidelity_test(classifier, res["x"], cam, class_idx=cls,
                            k_fractions=(0.01, 0.05, 0.10), n_random=3)

    verdict = decide(prob, conformal_set=conformal_set, mc_std=res["mc_std"],
                     fidelity_pass=None if fid is None else fid["passed"])
    box = [(s, s + n) for s, n in zip(res["start"], cls_cfg["roi_size"])]
    verdict.update({"saliency": cam, "fidelity": fid, "source": "classifier",
                    "classifier": cls_cfg, "phantom": phantom, "window": box,
                    "window_x": res["x"], "n_windows": res["n_windows"]})
    fig = overlay_figure(x, mask, gt, name, stamp=phantom, box=box)
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
    fid = verdict.get("fidelity")
    if fid is not None:
        mark = "PASS" if fid["passed"] else "FAIL"
        lines.append(f"**F-Fidelity** {mark} — masking the most salient voxels moves the "
                     f"score {fid['mean_margin']:+.3f} more than masking at random "
                     f"(needs > {fid['threshold']:.3f})")
    if verdict["reasons"]:
        lines += ["", "**Why it was escalated**"] + [f"- {r}" for r in verdict["reasons"]]
    phantom = verdict.get("phantom", True)
    if gt is not None:
        truth = "tumor" if (np.asarray(gt) == LABEL_TUMOR).any() else "no tumor"
        lines += ["", f"*{'Phantom ground' if phantom else 'Ground'} truth: {truth}.*"]

    lines += ["", "---", "", "**Provenance**",
              f"- segmentation weights: {'checkpoint loaded' if trained else 'UNTRAINED (random)'}"]
    if verdict.get("source") == "classifier":
        cfg = verdict.get("classifier") or {}
        lines += [
            f"- p(tumor): stage 6.3 classifier `{cfg.get('model', '?')}` "
            f"(roi {tuple(cfg.get('roi_size', ()))}), highest of "
            f"{verdict.get('n_windows', 0)} pancreas windows (yellow box)",
            "- fidelity signal: Grad-CAM++ on the 6.3 classifier, deciding window, "
            "F-Fidelity vs random masking",
            "- conformal q̂ was calibrated on patches; here it is applied to the "
            "highest-scoring window",
        ]
    else:
        lines += [
            "- p(tumor): read off the segmentation tumor channel; "
            "**no stage 6.3 classifier exists yet**",
            "- fidelity signal: Grad-CAM++ on the segmentation score head, "
            "F-Fidelity vs random masking",
        ]
    lines.append(f"- data: synthetic phantom. {DEMO_STAMP}" if phantom
                 else "- data: real CT (MSD Task07); research prototype, not for clinical use")
    return "\n".join(lines)


# -------------------------------------------------------------------------- UI

def build_ui(data_root: Path, checkpoint: str = "", mc_passes: int = MC_PASSES,
             classifier: str = ""):
    import gradio as gr

    from .explain import ScoreHead

    ckpt = find_checkpoint(checkpoint)
    model, trained = load_model(ckpt)
    explainer = ScoreHead(model)
    cls_path = find_classifier(classifier)
    clf, cls_cfg = load_classifier(cls_path) if cls_path else (None, None)
    calib = calibration_dir(clf is not None)
    q_hat, alpha = load_q_hat(calib / "stage64_summary.json")
    curve = load_coverage_curve(calib / "coverage_curve.csv")
    patch = checkpoint_patch(ckpt) or training_patch()
    cases = list_cases(data_root)
    phantom = "phantom" in str(data_root).lower()

    banner = (
        "### Pancreatic cancer early detection — stages 6.1 → 6.6\n"
        + (f"**{DEMO_STAMP}**\n\n" if phantom
           else "**Real CT (MSD Task07) · research prototype, not for clinical use**\n\n")
        + f"Segmentation: `{ckpt if ckpt else 'none — random weights'}` · "
        f"classifier: `{cls_path if cls_path else 'none — segmentation tumor-channel fallback'}` · "
        f"conformal q̂: `{q_hat if q_hat is not None else 'not calibrated'}` ({calib.name}) · "
        f"MC Dropout T={mc_passes} · window {patch}"
    )

    def run(case_name, uploaded, target_alpha, want_saliency):
        path = Path(uploaded) if uploaded else (data_root / "imagesTr" / case_name)
        if not path.exists():
            return None, None, f"**No such case:** `{path}`"
        q = q_hat_for(curve, target_alpha, fallback=q_hat)
        fig, verdict, gt, is_trained, _ = analyse(
            path, model, trained, q, target_alpha, mc_passes=mc_passes, patch=patch,
            explainer=explainer if want_saliency else None,
            classifier=clf, cls_cfg=cls_cfg, want_saliency=want_saliency, phantom=phantom)

        sal_fig = None
        if verdict.get("saliency") is not None:
            from .explain import overlay_saliency
            title = path.name.replace(".nii.gz", "")
            if verdict.get("window_x") is not None:      # classifier: the deciding window
                sal_fig = overlay_saliency(verdict["window_x"], verdict["saliency"],
                                           title=title + " (deciding window)", stamp=phantom)
            else:
                x, _ = load_volume(path)
                sal_fig = overlay_saliency(x, verdict["saliency"], gt=gt, title=title,
                                           stamp=phantom)
        return fig, sal_fig, report_markdown(verdict, gt, is_trained, target_alpha, mc_passes)

    with gr.Blocks(title="Pancreatic cancer early detection") as ui:
        gr.Markdown(banner)
        with gr.Row():
            with gr.Column(scale=1):
                case = gr.Dropdown(cases, value=cases[0] if cases else None,
                                   label="Case")
                upload = gr.File(label="…or upload a NIfTI volume (.nii.gz)",
                                 file_types=[".gz", ".nii"], type="filepath")
                alpha_in = gr.Slider(0.02, 0.5, value=alpha or 0.1, step=0.02,
                                     label="conformal alpha (1 - target coverage)",
                                     info="tighter coverage -> larger sets -> more escalations")
                saliency_in = gr.Checkbox(value=True, label="Grad-CAM++ and F-Fidelity (6.5)",
                                          info="adds ~20 forward passes")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=2):
                out_md = gr.Markdown()
        out_fig = gr.Plot(label="Segmentation")
        out_sal = gr.Plot(label="Grad-CAM++ saliency (stage 6.5)")
        go.click(run, inputs=[case, upload, alpha_in, saliency_in],
                 outputs=[out_fig, out_sal, out_md])
    return ui


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = MSD_DIR if (MSD_DIR / "imagesTr").exists() else DATA_DIR / "phantom"
    p.add_argument("--data-root", default=str(default_root))
    p.add_argument("--checkpoint", default="", help="stage 6.2 checkpoint")
    p.add_argument("--classifier", default="",
                   help="stage 6.3 checkpoint (default: checkpoints/stage63/best.pt if present)")
    p.add_argument("--mc-passes", type=int, default=MC_PASSES)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args(argv)

    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f"{root} does not exist — run `python -m src.run_demo_pipeline` first")
    ui = build_ui(root, args.checkpoint, args.mc_passes, args.classifier)
    ui.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
