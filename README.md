# AI-Based Pancreatic Cancer Prediction Using Deep Learning
### Early Detection from Medical Imaging Data — VTU Final-Year Project (Phase-II)

Anjuman Institute of Technology and Management, Bhatkal · Dept. of Computer Science & Engineering · 2026-27
**Team:** Hasan Ekkeri (2AB23CS045), Mustafa Hajeeb (2AB23CS047), Abdul Wasih (2AB23CS006)
**Guide:** Mrs. Arzoo, Assistant Professor

## What this project does
Pancreatic ductal adenocarcinoma is usually found too late; tumors under 2 cm are nearly isodense with normal pancreas on CT. This system detects small pancreatic tumors and, for every case, returns either a prediction with a calibrated confidence and a verified saliency map, or an explicit abstention that flags the case for specialist review.

## Pipeline
```
CT volume
  → 6.1 Preprocessing (resample, RAS, HU window, augmentation)
  → 6.2 ROI segmentation — Attention U-Net (pancreas + duct)
  → 6.3 Feature extraction — 3D ResNet / DenseNet on the ROI
  → 6.4 Calibrated uncertainty — MC Dropout + Conformal Prediction
  → 6.5 Explainability — Grad-CAM++ verified with F-Fidelity
  → 6.6 Selective prediction — predict, or abstain and escalate
```

## Datasets
| Dataset | Role | Notes |
|---|---|---|
| MSD Task07 Pancreas | Training / development | 281 labelled CTs, pancreas + tumor masks |
| PanTS (Li et al., NeurIPS 2025) | External validation, duct labels, metadata | test shard (901 cases); full-scale training is out of scope |

Data is not committed. Place it under `data/` (see `CLAUDE.md` for paths).

## Setup
```bash
git clone <repo-url> && cd pancreas-early-detection
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
GPU: PyTorch with CUDA 12.x for the RTX 4060 — install the matching wheel from pytorch.org before `requirements.txt` if the default one is CPU-only.

## Run
Notebooks are numbered by stage and run top-to-bottom:
```
notebooks/00_pants_metadata_eda.ipynb          PanTS metadata exploration
notebooks/01_preprocessing_and_baseline.ipynb  Stage 6.1 + pretrained baseline for 6.2 (Review 1)
notebooks/02_segmentation_attention_unet.ipynb Stage 6.2 training (Review 2)
...
```
Outputs go to `results/<review>/`; checkpoints to `checkpoints/` (gitignored).

## Status
| Review | Target | Date | State |
|---|---|---|---|
| Review 1 | 30% | 4 Sept 2026 | preprocessing pipeline, patient-wise split, augmentation, pretrained pancreas baseline scored on held-out cases, PanTS EDA |
| Review 2 | 60% | TBC (~mid-Oct) | stages 6.1–6.4 end-to-end |
| Final | 100% | TBC | all stages + Gradio demo + report |

## Project documents
- `docs/Phase-II_Project_Plan.md` — timeline, work split, risks
- `CLAUDE.md` — working context for AI-assisted development
- Phase-I report and review decks in `docs/`

## References (core)
- Li et al., *PanTS: The Pancreatic Tumor Segmentation Dataset*, NeurIPS 2025.
- Oktay et al., *Attention U-Net: Learning Where to Look for the Pancreas*, MIDL 2018.
- Gal & Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016.
- Zheng et al., *F-Fidelity: A Robust Framework for Faithfulness Evaluation of Explainable AI*, ICLR 2025.
- Antonelli et al., *The Medical Segmentation Decathlon*, Nat. Commun. 2022.
