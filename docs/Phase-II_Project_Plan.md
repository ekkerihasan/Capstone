# Phase-II Project Plan
**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Team: Hasan Ekkeri (2AB23CS045), Mustafa Hajeeb (2AB23CS047), Abdul Wasih (2AB23CS006)
Guide: Mrs. Arzoo, Asst. Professor, Dept. of CSE, AITM Bhatkal
Ref: Circular AITM/CSE/2026/08/01 (26 Aug 2026), VTU 2022 Scheme
Plan date: 3 Sept 2026

---

## 1. Milestones (per circular)

| Review | Target | Planned date | What must exist |
|---|---|---|---|
| Review 1 | 30% | 4 Sept 2026 | Stages 6.1–6.2 running on MSD Pancreas, PanTS metadata EDA, module map, corrected deck |
| Review 2 | 60% | ~mid-Oct 2026 (TBC) | Stages 6.1–6.4 end-to-end on MSD, external check on a PanTS shard, intermediate metrics |
| Final Review | 100% | TBC (~late Nov) | All six stages integrated, demo UI, final report, viva |

Logbook: update at least every 10 days. First entry due now, covering Review 1 preparation.

---

## 2. Data strategy

| Dataset | Role | Size | Where it lives |
|---|---|---|---|
| MSD Task07 Pancreas | Primary training/dev set (281 labelled portal-venous CTs, pancreas + tumor masks) | ~12 GB | Colab (Review 1), then college RTX 4060 |
| PanTS (HF: BodyMaps/PanTSMini) | External validation + duct/anatomy labels; sub-2 cm evaluation | 28–37 GB per shard, 346 GB total | Test shard (901 cases) + labels on 4060 machine by Week 2 |
| PanTS metadata.xlsx | EDA: tumor size, phase, hospital distribution | 1.3 MB | Downloaded locally now |

Decisions:
- PanTS is **not** the primary training set. Full-scale training is out of reach on a 4060; state this plainly in every review.
- **MRI** moves to "future work". No MRI dataset is available; do not present it as in-scope.
- Biomarkers / radiomics fusion (mentioned in Phase-I deck) is out of Phase-II scope.
- At 40 Mbps: MSD ≈ 45 min local download; one PanTS shard ≈ 2 h download + ~70 GB free disk to extract.

---

## 3. Week-by-week schedule

**Week 0 (3–4 Sept) — Review 1**
- Colab notebook: MONAI setup, MSD subset load, preprocessing (6.1), pretrained pancreas baseline + Attention U-Net first training run (6.2).
- PanTS metadata EDA notebook.
- Review 1 deck (9 items per circular), GitHub repo, logbook entry.

**Weeks 1–2 (7–20 Sept) — Data foundation**
- Book the RTX 4060; install CUDA/PyTorch/MONAI; clone repo.
- Full MSD locally. Download PanTS test shard + `PanTSMini_Label.tar.gz` overnight.
- Train Attention U-Net properly on MSD (target pancreas Dice ≥ 0.75). Validate on PanTS shard.
- Logbook entry #2.

**Weeks 3–4 (21 Sept – 4 Oct) — Stage 6.3 CNN feature extraction / classification**
- ROI crops from segmentation masks. 3D ResNet/DenseNet (MONAI) for tumor present/absent.
- Tumor size from mask voxel volume → sub-2 cm stratum for evaluation.
- Multi-depth feature fusion. Metrics: AUROC overall and per size bucket.
- Logbook entry #3.

**Weeks 5–6 (5–18 Oct) — Stage 6.4 calibrated uncertainty → Review 2**
- MC Dropout at inference (T = 20). Split-conformal prediction on held-out calibration set.
- Metrics: ECE, reliability diagram, empirical coverage vs. target.
- Review 2 deck: working modules, intermediate results, testing so far, screenshots.
- Logbook entry #4.

**Weeks 7–8 (19 Oct – 1 Nov) — Stage 6.5 fidelity-verified explainability**
- Grad-CAM++ on the classifier (slice-wise 2D; 3D CAM as stretch goal).
- F-Fidelity test: mask top-k salient regions, measure output change, compare against random masks. Pass/fail rule per saliency map.
- Logbook entry #5.

**Weeks 9–10 (2–15 Nov) — Stage 6.6 selective prediction + integration**
- Abstention rule combining conformal set size, MC variance, and fidelity pass.
- Gradio/Streamlit UI: input volume → mask, confidence interval, saliency, prediction or "flagged for specialist review".
- Second PanTS shard if disk permits. Logbook entry #6.

**Weeks 11–12 (16–29 Nov) — Final report and viva**
- Final report: Phase-I chapters + Implementation, Results, Testing, Screenshots, Conclusion & Future Work.
- End-to-end demo rehearsal. Logbook entry #7.

---

## 4. Work split

| Member | Primary | Secondary |
|---|---|---|
| Wasih | Pipeline code (6.1, 6.2, 6.4), Colab/4060 runs | Integration |
| Hasan | Review decks, report chapters, screenshots | 6.5 explainability |
| Mustafa | Data staging, GitHub/repo hygiene, logbook, 6.3 classifier | 6.6 UI |

(Adjust as the team prefers — but every stage needs one named owner.)

---

## 5. Repository layout

```
pancreas-early-detection/
  notebooks/
    00_pants_metadata_eda.ipynb
    01_preprocessing_msd.ipynb
    02_segmentation_attention_unet.ipynb
    03_classifier_resnet_densenet.ipynb
    04_uncertainty_mc_conformal.ipynb
    05_explainability_gradcam_ffidelity.ipynb
    06_selective_prediction_ui.ipynb
  src/            reusable functions pulled out of notebooks
  results/        metrics json, plots, screenshots per review
  docs/           decks, logbook scans, this plan
  requirements.txt
  README.md
```

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Colab free disconnects mid-training | Checkpoint to Google Drive every epoch; important runs repeat on the 4060 |
| PanTS shard too large for 4060 machine disk | Extract only a subset of cases from the tar; keep test shard only |
| Sub-2 cm tumors too few → weak results | Oversample; always report per-size-bucket metrics; use PanTS metadata to select small-tumor cases |
| Panel asks about MRI / PanTS full training | Pre-empt with an explicit scope slide at Review 1 |
| Work stuck on one laptop | Everything committed to GitHub; notebooks numbered by stage |
| Guide corrections not incorporated | Review guide feedback at each logbook entry; track as issues in the repo |

---

## 7. Phase-I inconsistencies to fix in Review 1 materials

- Framework: report says PyTorch; deck says TensorFlow/PyTorch → PyTorch + MONAI only.
- Title: use exactly the report/certificate title.
- Literature count: report has 12 papers, deck has 15 → align (keep 15, add the 3 to the report's Chapter 3 later).
- Deck filename mentions "Transformer" → rename; the approach is the CNN pipeline as approved.
