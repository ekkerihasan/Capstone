# CLAUDE.md — project context for Claude Code

Read this fully before doing anything. It is the source of truth for scope, data, hardware and deadlines.

## Project
**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
VTU 2022 scheme, final-year CSE major project (Phase-II), Anjuman Institute of Technology and Management, Bhatkal.
Team: Hasan Ekkeri (2AB23CS045), Mustafa Hajeeb (2AB23CS047), Abdul Wasih (2AB23CS006). Guide: Mrs. Arzoo.
The approved Phase-I report defines the design. Do not change the title or the approach (CNN pipeline, not Transformer).

## Goal
Detect small (sub-2 cm) pancreatic tumors on CT and return, for each case: a prediction, a calibrated confidence, a fidelity-verified saliency map, or an explicit abstention ("flag for specialist review").

## Pipeline — six stages (Chapter 6 of the report). Keep this numbering in code and notebooks.
| Stage | What | Status | Notebook |
|---|---|---|---|
| 6.1 | Preprocessing: resample (1.5,1.5,2.0 mm), RAS, HU window −100…240 → [0,1], patient-wise split, aug (flip, rot90, elastic) | done (Review 1) | `01_preprocessing_and_baseline.ipynb` |
| 6.2 | ROI segmentation: pancreas + duct. Baseline = MONAI `pancreas_ct_dints_segmentation` bundle; target = `monai.networks.nets.AttentionUnet` | baseline done; training = Week 1–2 | `02_segmentation_attention_unet.ipynb` |
| 6.3 | ROI crop → 3D ResNet/DenseNet classifier, tumor present/absent, multi-depth feature fusion | Week 3–4 | `03_classifier_resnet_densenet.ipynb` |
| 6.4 | MC Dropout (T=20) + split-conformal prediction; report ECE, coverage, reliability diagram | Week 5–6 (Review 2) | `04_uncertainty_mc_conformal.ipynb` |
| 6.5 | Grad-CAM++ on classifier + F-Fidelity masking test (top-k mask vs random mask, measure output drop) | Week 7–8 | `05_explainability_gradcam_ffidelity.ipynb` |
| 6.6 | Selective prediction: abstain on low confidence / large conformal set / failed fidelity; Gradio UI | Week 9–10 | `06_selective_prediction_ui.ipynb` |

## Datasets — roles are fixed, do not reassign
- **MSD Task07 Pancreas** — PRIMARY training/dev set. 281 labelled portal-venous CTs, labels 0/1/2 = bg/pancreas/tumor. ~12 GB. Path on lab PC: `data/Task07_Pancreas/`. Cases are patients → case-level split = patient-wise split.
- **PanTS** (HF `BodyMaps/PanTSMini`, CC-BY-NC-SA 4.0) — EXTERNAL VALIDATION only, plus duct/anatomy labels and metadata. One shard = 28–37 GB compressed. Only the test shard (901 cases) + `PanTSMini_Label.tar.gz` are staged. Path: `data/PanTS/`.
- **PanTS `metadata.xlsx`** — used for EDA and sub-2 cm case selection.
- **Out of scope for Phase-II:** MRI, blood biomarkers, radiomics fusion, PanTS full-scale (36,390-scan) training. Never write code, docs or slides that imply these are done or in progress.

## Hardware and its consequences
- Lab PC: RTX 4060, **8 GB VRAM**, 1 TB disk. Overnight access not confirmed → every training loop must checkpoint every epoch and resume from `checkpoints/<stage>/last.pt`.
- Colab free T4 (16 GB) for short runs only; sessions die. Never rely on it for a run longer than ~2 h.
- Defaults that fit 8 GB: 3D patches 96×96×96, batch size 2, `torch.cuda.amp` on, `CacheDataset` with `cache_rate` ≤ 0.3, `num_workers=4`.
- If OOM: reduce batch to 1, then patch to 64³, before touching the model.

## Stack
Python 3.10+, PyTorch 2.x, MONAI ≥ 1.3, nibabel, SimpleITK, pydicom, pandas, matplotlib, scikit-learn, pytorch-ignite (for MONAI bundles), gradio (stage 6.6). PyTorch only — no TensorFlow anywhere.

## Repo conventions
```
notebooks/   numbered by stage (00_ … 06_). Keep them runnable top-to-bottom.
src/         importable code pulled out of notebooks (transforms.py, data.py, models.py, metrics.py, uncertainty.py, explain.py)
results/     per-review folders: results/review1/, results/review2/, results/final/ — metrics as CSV/JSON, figures as PNG @150 dpi
checkpoints/ gitignored
data/        gitignored
docs/        plan, decks, logbook scans
```
- Metrics always reported overall AND for the sub-2 cm bucket (equivalent spherical diameter from mask volume).
- Every figure that goes in a review deck is saved to `results/<review>/` with a descriptive filename, never only shown inline.
- Seeds fixed (`monai.utils.set_determinism(42)`), splits saved to JSON and reused.
- Small, tested changes. Run the notebook or script after editing; do not claim something works without an output.

## Deadlines
- Review 1 (30%): 4 Sept 2026 — done: stages 6.1, 6.2 baseline, PanTS EDA.
- Review 2 (60%): ~mid-Oct 2026 (TBC) — stages 6.1–6.4 end-to-end on MSD, external check on PanTS shard.
- Final (100%): TBC (~late Nov 2026) — all six stages, Gradio demo, final report, viva.
- Logbook entry every 10 days. When a milestone is reached, add a bullet to `docs/logbook_notes.md`.

## Current task (3–4 Sept 2026)
**Review 1 is tomorrow.** Read `docs/REVIEW1_GAMEPLAN.md` before doing anything — section 10 defines exactly what is in and out of scope tonight. In short: get notebooks `00` and `01` running end-to-end, save every figure and table to `results/review1/`, commit the executed notebooks. Do not start Attention U-Net training or stages 6.3–6.6.

## When asked to "continue the project"
1. Read `docs/Phase-II_Project_Plan.md` and the latest `results/` folder.
2. Identify the current week and the stage it maps to above.
3. Work on that stage's notebook/`src` module. Do not skip ahead to later stages.
4. Finish by updating the status table in this file and `docs/logbook_notes.md`.