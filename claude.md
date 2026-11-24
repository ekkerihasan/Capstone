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
| 6.2 | ROI segmentation: pancreas/tumor Attention U-Net; MSD has no duct labels | trained 200 epochs; val whole-pancreas Dice 0.736, sub-2 cm tumor Dice 0.202 (below 0.75 target) | `02_segmentation_attention_unet.ipynb` |
| 6.3 | 3D DenseNet121 patch classifier; tumor vs non-tumor pancreas tissue | checkpoint saved at epoch 39; best val patch AUROC 0.833 at epoch 37; oracle and predicted-mask inference complete | `03_classifier_resnet_densenet.ipynb` |
| 6.4 | MC Dropout (T=20) + split-conformal prediction; ECE and coverage | real patch-level results generated; predicted-mask val AUROC 0.846, 0.929 patch coverage; not a patient-level guarantee | `04_uncertainty_mc_conformal.ipynb` |
| 6.5 | Grad-CAM++ + F-Fidelity masking test | evaluated on 168 val patches; pass rate 47.0% overall, 54.5% for 11 sub-2 cm patches | `05_explainability_gradcam_ffidelity.ipynb` |
| 6.6 | Selective prediction and Gradio UI | CUDA demo verified; val evaluation: 18/42 accepted, 57.1% abstained; 100% accuracy among accepted, on tumor-only MSD patients | `06_selective_prediction_ui.ipynb` |

Stage 6.3 is **patch-level**, not a patient-level tumor-presence detector: every MSD patient has a tumor, and negative patches are tumor-free tissue from those same patients. Stage 6.4 calibrates on patches and applies the threshold to the app's maximum-scoring window, so its app-level coverage is approximate.

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

## Current task (24 Nov 2026)
**Final review is 25 Nov.** Real-MSD compute for stages 6.1–6.6 is complete. The Phase-II report addendum, 12-slide deck (PDF/PPTX), rehearsal run sheet, results handoff, and 222-test report are prepared; the offline bundle is on Wasih's USB at `D:\capstone\FINAL_REVIEW_2026-11-25\repo\`. The app returned HTTP 200 locally with CUDA. Still perform a human timed rehearsal and review the exports on the presentation computer, send the handoff to Mustafa, and complete PR review/merge. Preserve the patch-level and validation-reuse limitations; do not claim PanTS external validation or clinical readiness. Current evidence and remaining human/external actions are tracked in `docs/REMAINING_WORK.md`.

## When asked to "continue the project"
1. Read `docs/Phase-II_Project_Plan.md` and the latest `results/` folder.
2. Identify the current week and the stage it maps to above.
3. Work on that stage's notebook/`src` module. Do not skip ahead to later stages.
4. Finish by updating the status table in this file and `docs/logbook_notes.md`.