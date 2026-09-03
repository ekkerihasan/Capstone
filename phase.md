# Review 1 — Gameplan
**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Phase-II, Review 1 · target: **minimum 30% implementation** · review on **4 Sept 2026, afternoon/evening**
Team: Hasan (2AB23CS045), Mustafa (2AB23CS047), Wasih (2AB23CS006) · Guide: Mrs. Arzoo

---

## 1. What "30%" means for us

Two of the six pipeline stages, evidenced with real outputs:

- **Stage 6.1 Preprocessing** — implemented and demonstrated on real CT volumes.
- **Stage 6.2 ROI segmentation** — a pretrained baseline running and scored on held-out cases. Our own Attention U-Net training is Review 2.
- Plus dataset groundwork: PanTS metadata analysis, patient-wise split, repo and environment.

Stages 6.3–6.6 are **not started**. Say so.

---

## 2. Tonight — three parallel tracks

### Track A — Wasih (code)
| Order | Task | Est. |
|---|---|---|
| A1 | Unzip repo, `git init`, push to GitHub, invite Hasan + Mustafa | 15 min |
| A2 | Open `notebooks/00_pants_metadata_eda.ipynb` (CPU, 1.3 MB download). Fix the column-name mapping cell against the real `df.columns`. Save `fig_pants_metadata.png` | 30 min |
| A3 | Open `notebooks/01_preprocessing_and_baseline.ipynb` in Colab, **T4 GPU runtime**. Run cells 0–1 (MSD download, ~12 GB on Colab's link) | 20 min |
| A4 | Run cells 2–4: split, stats table, tumor-size histogram, overlay figures, augmentation figure | 45 min |
| A5 | Run cell 5: download the MONAI DiNTS bundle, inference on the 4 held-out cases, Dice table, GT-vs-prediction figures | 1–2 h |
| A6 | Confirm everything is in Drive under `pancreas_project/results/review1/`. Commit the executed notebook **with outputs saved** | 15 min |

### Track B — Hasan (deck)
Start from the Phase-I deck. Fix: framework → PyTorch + MONAI only (no TensorFlow); title → exact report title; literature → one comparative-table slide, not fifteen.
Build the four new slides (text in §5 below), leave screenshot placeholders for Track A.

### Track C — Mustafa (process + data)
- Logbook entry #1 (template in §6). The 10-day rule started 26 Aug — this entry is already due.
- Confirm the RTX 4060 booking and overnight-access policy with the department.
- `requirements.txt` sanity check; add `.gitignore`; create `results/review1/` in the repo.

---

## 3. Tomorrow morning (2–3 hours)

1. Paste Track A screenshots into the deck placeholders.
2. Fill real numbers into the status slide: subset size, number of sub-2 cm cases, baseline Dice.
3. One full rehearsal out loud, timed. Target 10–12 minutes.
4. Print the logbook page; have the Project File ready — the circular requires it at every review.
5. Offline backup: PDF of the deck + executed notebook HTML on a pen drive. Assume the room has no internet.

---

## 4. Demo running order (10–12 min)

| # | Show | Say |
|---|---|---|
| 1 | GitHub repo | structure, notebooks numbered by stage |
| 2 | `nvidia-smi` + MONAI config | environment is set up and reproducible |
| 3 | Dataset summary + stats table + tumor-size histogram | "N of our 20 cases are sub-2 cm — that's the case the project targets" |
| 4 | Overlay figures (before/after preprocessing) + augmentation figure | **Stage 6.1 complete** — walk through resampling, HU window, patient-wise split |
| 5 | Baseline Dice table + GT-vs-prediction figures | **Stage 6.2 baseline** — "this is the bar our Attention U-Net must beat at Review 2" |
| 6 | PanTS metadata figure | scale of the target dataset, and why we validate rather than train on it |

**Do not attempt a live training run.** Everything is pre-run; re-execute one cell only if the room's internet is good.

---

## 5. Deck structure (~16 slides, mapped to the circular)

| Circular item | Slide | Status |
|---|---|---|
| 1 Project title | 1 | from Phase-I |
| 2 Existing & proposed system | 2–3 | from Phase-I, framework fixed |
| 3 Background | 4 | from Phase-I |
| 4 Problem statement & objectives | 5–6 | from report §1.3, §1.5 |
| 5 Literature survey | 7 | comparative table 3.13; "15 papers, detailed in the report" |
| 6 Project modules | 8 | **new — text below** |
| 7 Methodology | 9 | architecture diagram (`docs/architecture_diagram.html`) |
| — | 10 | **new — datasets & Phase-II scope** |
| — | 11 | PanTS at a glance (metadata figure) |
| 8 Minimum 30% implementation | 12 | **new — status checklist** |
| 9 Proof of implementation | 13–15 | screenshots |
| — | 16 | **new — plan to Review 2** |

### Slide 8 — Project modules
> The pipeline has six stages, following Chapter 6 of the report.
> **6.1 Preprocessing** — resampling, orientation, HU windowing, augmentation, patient-wise split · *done*
> **6.2 ROI segmentation** — Attention U-Net for pancreas and duct · *baseline running, training next*
> **6.3 Feature extraction** — 3D ResNet / DenseNet on the ROI · *Review 2*
> **6.4 Calibrated uncertainty** — MC Dropout + conformal prediction · *Review 2*
> **6.5 Verified explanation** — Grad-CAM++ checked with F-Fidelity · *final review*
> **6.6 Selective prediction** — predict, or abstain and escalate · *final review*

### Slide 10 — Datasets and scope for Phase-II
> **Training and development:** MSD Task07 Pancreas — 281 labelled CT volumes with pancreas and tumor masks.
> **External validation:** PanTS test shard — 901 volumes, with pancreatic duct annotations.
> PanTS in full is 36,390 scans and roughly 300 GB. Training on all of it needs cluster hardware; we validate on it instead, which is the stronger test of generalisation anyway.
> **Future work, not this phase:** MRI, blood biomarkers, radiomics fusion.

### Slide 12 — Implementation status: 30%
> ✓ Environment: PyTorch + MONAI on Colab T4; RTX 4060 arranged for training
> ✓ Data pipeline: MSD Pancreas loaded, 20-case working subset, patient-wise split saved to JSON
> ✓ Stage 6.1 complete: resampling, RAS orientation, HU windowing, elastic/flip/rotation augmentation
> ✓ Stage 6.2 baseline: pretrained pancreas segmentation scored on 4 held-out cases — Dice ___ pancreas, ___ tumor
> ✓ PanTS metadata analysed: ___ % of tumors are under 2 cm
> ✓ Repository, documentation and logbook in place
> → Next: train our Attention U-Net and beat the baseline

### Slide 16 — Plan to Review 2
> Weeks 1–2: full MSD locally, PanTS shard staged, Attention U-Net trained (stage 6.2)
> Weeks 3–4: ROI classifier, sub-2 cm evaluation (stage 6.3)
> Weeks 5–6: MC Dropout + conformal prediction, calibration metrics (stage 6.4)
> **At Review 2: stages 6.1–6.4 end-to-end on MSD, externally checked on PanTS.**

---

## 6. Logbook entry #1

> **Date:** 3–4 Sept 2026 · **Present:** Hasan, Mustafa, Wasih
> **Work done:** Set up PyTorch + MONAI environment and project repository. Implemented the Stage 6.1 preprocessing pipeline (resampling to 1.5×1.5×2.0 mm, RAS orientation, HU windowing −100…240, elastic/flip/rotation augmentation) and a patient-wise split on the MSD Task07 Pancreas dataset. Ran a pretrained pancreas segmentation baseline on held-out cases and measured Dice. Analysed the PanTS metadata to quantify the sub-2 cm tumor population.
> **Guide's suggestions:** _(fill in after the review)_
> **Next actions:** train the Attention U-Net (Stage 6.2) on the college RTX 4060; stage the PanTS test shard.

---

## 7. Answers to keep ready

**Why not train on PanTS?** 300 GB and an 8 GB GPU. We use its test shard for external validation, which tests generalisation better than training on it would.

**Why is the tumor Dice low?** That is the pretrained baseline's published performance, not our model. Small isodense tumors are precisely the hard case this project exists to address — the number is the problem statement in numeric form.

**What is novel here?** None of the 15 papers reviewed combines calibrated confidence, faithfulness-verified explanations, and the ability to abstain in one pipeline. Each solves one piece.

**Is 30% really done?** Two of six stages, with running code and measured outputs, plus dataset and infrastructure work.

**Where is the MRI part?** Moved to future work — no suitable annotated MRI dataset is available to us this phase.

---

## 8. Risks tonight

| Risk | Response |
|---|---|
| Colab gives no GPU | Preprocessing and figures run on CPU; skip the baseline, present stage 6.1 only and say the baseline is running |
| MSD download stalls | Retry; `download_and_extract` resumes. Fallback: use 5 cases and note the subset size |
| Bundle inference OOM | Add `--inferer#sw_batch_size 1`; if it still fails, drop to 2 cases |
| `torch.load` weights_only error | Add `--arch_ckpt "$torch.load(@arch_ckpt_path, map_location=torch.device('cuda'), weights_only=False)"` |
| PanTS metadata columns not found | Print `df.columns`, set `COL_*` manually; the histogram is optional, the scan count is not |
| Everything fails | Present the plan, architecture diagram, repo and preprocessing code honestly. A clear plan with real code beats a fabricated result |

---

## 9. Hard rules for tomorrow

1. Do not claim any of stages 6.3–6.6 exist.
2. Do not present the pretrained baseline as our trained model.
3. Do not claim PanTS full-scale training.
4. Do not claim MRI support.
5. Every number on a slide must trace to a file in `results/review1/`.

---

## 10. For Claude Code sessions tonight

Open the repo in VS Code and start with:

> Read CLAUDE.md and docs/REVIEW1_GAMEPLAN.md. We are in Track A, task A3. Help me run notebooks/01_preprocessing_and_baseline.ipynb on the Colab T4 through the VS Code extension, and fix errors as they come.

Scope for tonight — do only this:
- Get `00_pants_metadata_eda.ipynb` and `01_preprocessing_and_baseline.ipynb` to run end-to-end.
- Fix errors in place; keep cell structure and stage numbering intact.
- Make sure every figure and table is written to `results/review1/`.
- Commit the executed notebooks with outputs.

Out of scope tonight — do **not** start these, even if asked in passing:
- training the Attention U-Net (that is Week 1–2)
- anything in stages 6.3, 6.4, 6.5, 6.6
- refactoring notebooks into `src/` modules
- changing the pipeline design, dataset roles, or the project title

If a task looks out of scope, say so and stop rather than starting it.