# Final results handoff

Prepared 24 Nov 2026 for Mustafa's paper. Use the linked CSV/JSON files as the source of exact values.

## Core tables

| Stage | Measure | Overall | Sub-2 cm |
|---|---|---:|---:|
| 6.2 val | Whole-pancreas Dice | 0.736 | — |
| 6.2 val | Tumor Dice | 0.415 | 0.202 (n=11) |
| 6.3 | Best validation patch AUROC | 0.833 at epoch 37 | Check Stage 6.4 rows below for AUROC against all negatives |
| 6.4 oracle | AUROC | 0.823 | 0.810 |
| 6.4 predicted masks | AUROC | 0.846 | 0.811 |
| 6.4 oracle | ECE / coverage / mean set size / abstention | 0.2556 / 0.899 / 1.542 / 0.542 | coverage 1.000 / abstention 0.545 (n=11) |
| 6.4 predicted masks | ECE / coverage / mean set size / abstention | 0.2412 / 0.929 / 1.571 / 0.571 | coverage 1.000 / abstention 0.545 (n=11) |
| 6.5 | F-Fidelity pass rate | 0.470 (168 patches) | 0.545 (11 tumor patches) |
| 6.6 | Accepted / abstained / accepted accuracy | 18/42 / 0.571 / 1.000 | 4/11 accepted / 0.636 abstention / 1.000 accepted accuracy |

## Source artifacts

- Stage 6.2: `results/final/seg_val/seg_summary.json`, `seg_by_size_bucket.csv`
- Stage 6.3/6.4 oracle: `results/final/oracle/stage64_summary.json`,
  `stage64_by_size_bucket.csv`
- Stage 6.3/6.4 predicted masks: `results/final/real/stage64_summary.json`,
  `stage64_by_size_bucket.csv`
- Stage 6.5: `results/final/stage65/stage65_summary.json`
- Stage 6.6: `results/final/stage66/stage66_summary.json`
- Figures and UI screenshots: `results/final/`

## Do not overstate

- AUROC and conformal metrics are **patch-level**. The app applies patch calibration to its
  highest-scoring window, so app-level coverage is approximate.
- MSD Task07 patients all have tumors. Negative examples are tumor-free pancreatic tissue
  patches from tumor-bearing patients; these data do not establish patient-level specificity.
- 6.2 chose its checkpoint on `val`, which is also the 6.4 evaluation group. There is no
  independent fourth split.
- The whole-pancreas Dice is below the 0.75 target; sub-2 cm tumor Dice is 0.202.
- PanTS external validation was not completed. Do not claim clinical readiness.
