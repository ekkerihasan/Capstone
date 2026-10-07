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
from .selective import ABSTAIN, MAX_MC_STD, decide, label_of

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
            mc_passes: int = MC_PASSES, patch=PATCH, explainer=None):
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
    fig = overlay_figure(x, mask, gt, image_path.name.replace(".nii.gz", ""))
    return fig, verdict, gt, trained, alpha


CARD = ("display:inline-block;vertical-align:top;width:31%;min-width:230px;margin:0 1% 10px 0;"
        "padding:14px 16px;border-radius:10px;background:#1b2730;border-left:4px solid {edge}")

VERDICT_STYLE = {
    "tumor": ("#D1495B", "The system is reporting a tumour in this scan."),
    "no_tumor": ("#2A9D8F", "The system found no tumour in this scan."),
    ABSTAIN: ("#E09F3E", "The system will not answer this scan. It is asking for a human."),
}


def _card(edge: str, state: str, title: str, value: str, meaning: str) -> str:
    return (f"<div style='{CARD.format(edge=edge)}'>"
            f"<div style='font-size:12px;letter-spacing:1px;color:{edge};font-weight:600'>{state}</div>"
            f"<div style='font-size:15px;font-weight:600;margin-top:2px'>{title}</div>"
            f"<div style='font-size:19px;margin-top:6px'>{value}</div>"
            f"<div style='font-size:13px;color:#9aa8b2;margin-top:4px;line-height:1.4'>{meaning}</div>"
            "</div>")


def verdict_html(verdict: dict, gt, alpha, mc_passes: int) -> str:
    """The verdict as three labelled checks instead of one paragraph.

    Each check states in words what it measured and whether that was a problem.
    A reader who does not already know what a conformal set is cannot act on
    "set size 2", and the people this has to convince are radiologists and
    examiners, not the three of us who wrote it.
    """
    d = verdict["decision"]
    colour, lead = VERDICT_STYLE[d]
    ok, flag = "#2A9D8F", "#E09F3E"

    out = [
        f"<div style='padding:16px 18px;border-radius:12px;background:{colour}1A;"
        f"border:1px solid {colour}'>"
        f"<div style='font-size:26px;font-weight:700;color:{colour};line-height:1.2'>{label_of(d)}</div>"
        f"<div style='font-size:15px;color:#c8d4da;margin-top:6px'>{lead}</div></div>",
        "<div style='margin-top:14px'>",
    ]

    # 1 - the calibrated answer
    cset = verdict["conformal_set"]
    if cset is None:
        out.append(_card("#6b7a85", "NOT AVAILABLE", "Calibrated answer",
                         "&mdash;", "No calibration file was found."))
    else:
        names = {0: "no tumour", 1: "tumour"}
        inside = ", ".join(names[c] for c in cset) or "empty"
        single = len(cset) == 1
        out.append(_card(
            ok if single else flag,
            "OK" if single else "NEEDS A HUMAN",
            f"Calibrated answer &mdash; {int((1 - (alpha or 0.1)) * 100)}% confidence",
            "{" + inside + "}",
            "Narrowed to one answer at the confidence we demand." if single
            else "At the confidence we demand, both answers are still possible."))

    # 2 - do the repeated runs agree with each other
    std = verdict["mc_std"]
    if std is None:
        out.append(_card("#6b7a85", "NOT AVAILABLE", "Model agreement",
                         "&mdash;", "Repeat runs were switched off."))
    else:
        steady = std <= MAX_MC_STD
        out.append(_card(
            ok if steady else flag,
            "OK" if steady else "NEEDS A HUMAN",
            f"Model agreement &mdash; {mc_passes} repeat runs",
            f"spread {std:.3f}",
            f"The {mc_passes} runs agreed with each other." if steady
            else f"The {mc_passes} runs disagreed (over {MAX_MC_STD:.2f})."))

    # 3 - does the explanation survive being tested
    fid = verdict.get("fidelity")
    if fid is None:
        out.append(_card("#6b7a85", "NOT AVAILABLE", "Explanation check",
                         "&mdash;", "Saliency was switched off for this run."))
    else:
        passed = fid["passed"]
        out.append(_card(
            ok if passed else flag,
            "OK" if passed else "NEEDS A HUMAN",
            "Explanation check &mdash; F-Fidelity",
            "PASS" if passed else "FAIL",
            "Hiding the highlighted region really does change the answer." if passed
            else "Hiding the highlighted region changed the answer no more than hiding "
                 "random tissue, so the heat map is not trustworthy."))

    out.append("</div>")

    if verdict["reasons"]:
        items = "".join(f"<li style='margin:3px 0'>{r}</li>" for r in verdict["reasons"])
        out.append("<div style='margin-top:6px;font-size:14px;color:#c8d4da'>"
                   f"<b>Escalated because:</b><ul style='margin:6px 0 0 18px'>{items}</ul></div>")

    if gt is not None:
        truth = "tumour" if (np.asarray(gt) == LABEL_TUMOR).any() else "no tumour"
        agrees = (truth == "tumour") == (d == "tumor")
        note = ("the system agreed" if d != ABSTAIN and agrees else
                "the system disagreed" if d != ABSTAIN else
                "the system declined to answer rather than guess")
        out.append("<div style='margin-top:10px;font-size:14px;color:#9aa8b2'>"
                   f"<b>Known answer for this test case:</b> {truth} &mdash; {note}.</div>")
    return "".join(out)


def provenance_html(trained: bool) -> str:
    """Folded away by default, but always one click from the verdict."""
    rows = [
        ("Segmentation weights", "checkpoint loaded" if trained else "UNTRAINED (random)"),
        ("Tumour probability", "read off the segmentation tumour channel &mdash; "
                               "<b>no stage 6.3 classifier exists yet</b>"),
        ("Explanation", "Grad-CAM++ on the segmentation score head, "
                        "F-Fidelity vs random masking"),
        ("Data", f"synthetic phantom &mdash; {DEMO_STAMP}"),
    ]
    body = "".join(
        f"<tr><td style='padding:4px 14px 4px 0;color:#9aa8b2;white-space:nowrap'>{k}</td>"
        f"<td style='padding:4px 0'>{v}</td></tr>" for k, v in rows)
    return ("<details style='margin-top:14px;font-size:13px;color:#c8d4da'>"
            "<summary style='cursor:pointer;color:#9aa8b2'>What this run is actually based on"
            "</summary>"
            f"<table style='margin-top:8px;border-collapse:collapse'>{body}</table></details>")


HOW_TO_READ_SEG = (
    "<div style='font-size:14px;color:#c8d4da;line-height:1.55'>"
    "<b>How to read this.</b> Left is the CT after preprocessing. Middle is what the model "
    "found: <span style='color:#4ade80'><b>green outline = pancreas</b></span>, "
    "<span style='color:#f87171'><b>red outline = tumour</b></span>. Right is the correct "
    "answer, for comparison.</div>")

HOW_TO_READ_CAM = (
    "<div style='font-size:14px;color:#c8d4da;line-height:1.55'>"
    "<b>How to read this.</b> The heat map shows <i>where the model looked</i> to reach its "
    "answer &mdash; <span style='color:#f87171'><b>red is where it looked hardest</b></span>, "
    "blue is where it barely looked. The <b>white outline</b> is the real tumour. A "
    "trustworthy explanation puts the red inside the white."
    "<br><br><b>But looking right is not proof.</b> So the system hides the red region and "
    "checks whether the answer actually changes, then compares that against hiding the same "
    "amount of <i>random</i> tissue. If hiding the red region matters no more than hiding "
    "anything else, the heat map was decoration &mdash; and the Explanation check above "
    "reports FAIL.</div>")


def report_markdown(verdict: dict, gt, trained: bool, alpha, mc_passes: int) -> str:
    """The whole report as one string, for callers that want it in one piece."""
    return verdict_html(verdict, gt, alpha, mc_passes) + provenance_html(trained)


# -------------------------------------------------------------------------- UI

def build_ui(data_root: Path, checkpoint: str = "", mc_passes: int = MC_PASSES):
    import gradio as gr

    from .explain import ScoreHead

    ckpt = find_checkpoint(checkpoint)
    model, trained = load_model(ckpt)
    explainer = ScoreHead(model)
    q_hat, alpha = load_q_hat()
    curve = load_coverage_curve()
    patch = training_patch()
    cases = list_cases(data_root)

    header = (
        "<div style='padding:2px 0 10px'>"
        "<div style='font-size:24px;font-weight:700'>Pancreatic cancer early detection</div>"
        "<div style='font-size:15px;color:#9aa8b2;margin-top:4px'>"
        "A CT volume goes in. The system answers &mdash; or refuses, and says why. "
        "Stages 6.1 to 6.6.</div>"
        f"<div style='margin-top:10px;display:inline-block;padding:5px 11px;border-radius:6px;"
        f"background:#C1453A22;border:1px solid #C1453A;color:#E8938B;font-size:12px;"
        f"letter-spacing:1px;font-weight:600'>{DEMO_STAMP}</div></div>")

    waiting = ("<div style='padding:28px 18px;border-radius:12px;background:#1b2730;"
               "color:#9aa8b2;font-size:15px;line-height:1.6'>"
               "Pick a case on the left and press <b>Analyse</b>.<br>"
               "The system will run all six stages and show you the verdict, the three checks "
               "behind it, and where the model looked.</div>")

    def run(case_name, uploaded, target_alpha, want_saliency):
        path = Path(uploaded) if uploaded else (data_root / "imagesTr" / case_name)
        if not path.exists():
            return None, None, f"**No such case:** `{path}`", ""
        q = q_hat_for(curve, target_alpha, fallback=q_hat)
        fig, verdict, gt, is_trained, _ = analyse(
            path, model, trained, q, target_alpha, mc_passes=mc_passes, patch=patch,
            explainer=explainer if want_saliency else None)

        sal_fig = None
        if verdict.get("saliency") is not None:
            from .explain import overlay_saliency
            x, _ = load_volume(path)
            sal_fig = overlay_saliency(x, verdict["saliency"], gt=gt,
                                       title=path.name.replace(".nii.gz", ""))
        report = (verdict_html(verdict, gt, target_alpha, mc_passes)
                  + provenance_html(is_trained))
        return fig, sal_fig, report, (HOW_TO_READ_CAM if sal_fig is not None else "")

    with gr.Blocks(title="Pancreatic cancer early detection") as ui:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                case = gr.Dropdown(cases, value=cases[0] if cases else None,
                                   label="1. Choose a scan")
                upload = gr.File(label="…or upload a NIfTI volume (.nii.gz)",
                                 file_types=[".gz", ".nii"], type="filepath")
                alpha_in = gr.Slider(0.02, 0.5, value=alpha or 0.1, step=0.02,
                                     label="2. How confident must it be?",
                                     info="Left = stricter, so more scans get sent to a human")
                saliency_in = gr.Checkbox(value=True,
                                          label="Explain the decision (stage 6.5)",
                                          info="Adds about a minute: Grad-CAM++ and the "
                                               "masking test")
                go = gr.Button("3. Analyse", variant="primary", size="lg")
                gr.HTML("<div style='margin-top:10px;font-size:12px;color:#6b7a85;"
                        "line-height:1.5'>"
                        f"Weights: {'checkpoint' if trained else 'UNTRAINED'} &middot; "
                        f"repeat runs: {mc_passes} &middot; window {patch}</div>")
            with gr.Column(scale=2, min_width=420):
                out_md = gr.HTML(waiting)

        gr.HTML("<div style='margin:18px 0 4px;font-size:17px;font-weight:600'>"
                "What the model found</div>")
        out_fig = gr.Plot(label="", show_label=False)
        gr.HTML(HOW_TO_READ_SEG)

        gr.HTML("<div style='margin:22px 0 4px;font-size:17px;font-weight:600'>"
                "Why it decided that &mdash; and whether that reason holds up</div>")
        out_sal = gr.Plot(label="", show_label=False)
        cam_help = gr.HTML("")

        go.click(run, inputs=[case, upload, alpha_in, saliency_in],
                 outputs=[out_fig, out_sal, out_md, cam_help])
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
