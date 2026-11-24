# Final Review Deck — Slide Outline and Presenter Notes

The accompanying PDF and PPTX are generated from this outline and the figures in `results/final/`.
Keep the final claims narrow: this is an MSD research prototype, not a clinical diagnostic tool.

## Slide 1 — Title

**AI-Based Pancreatic Cancer Prediction Using Deep Learning**<br>
Early detection from CT imaging · Phase-II final review · 25 Nov 2026

Presenter: This is a six-stage CNN pipeline evaluated on MSD Task07 Pancreas.

## Slide 2 — Problem and scope

- Target: support detection of small pancreatic lesions on portal-venous CT.
- Primary/dev data: MSD Task07, 281 labelled cases.
- Out of scope: MRI, biomarkers, radiomics fusion, full-scale PanTS training.
- PanTS external validation was not completed.

Presenter: MSD cases all contain tumors; the 6.3 negatives are tumor-free tissue windows, not
healthy patients.

## Slide 3 — Six-stage pipeline

1. 6.1 preprocessing and patient-wise split.
2. 6.2 pancreas/tumor Attention U-Net.
3. 6.3 DenseNet121 tumor-vs-tumor-free tissue patch classifier.
4. 6.4 MC Dropout and split-conformal prediction.
5. 6.5 Grad-CAM++ with F-Fidelity verification.
6. 6.6 selective prediction and Gradio UI.

Presenter: A failed fidelity check suppresses the saliency map and can cause abstention.

## Slide 4 — Data and evaluation protocol

- MSD Task07: 281 cases; split 197 train / 42 val / 42 calibration.
- Patient-wise split; labels: background, pancreas, tumor.
- Sub-2 cm defined from equivalent spherical diameter.
- Report all results overall and by size when available.

Presenter: The same validation group is reused for segmentation model selection and subsequent
evaluation; state this limitation.

## Slide 5 — Stage 6.2 segmentation

- Val whole-pancreas Dice: 0.736 (target 0.75 not met).
- Val tumor Dice: 0.415.
- Tumor <2 cm Dice: 0.202 (n=11); ≥2 cm Dice: 0.491 (n=31).
- Include `results/final/fig_stage62_training.png` and a representative overlay.

Presenter: Whole-pancreas Dice includes labels 1+2; small tumors remain the weakest group.

## Slide 6 — Stage 6.3 patch classifier

- DenseNet121, ROI 32³, dropout 0.2, three negative tissue windows/case.
- Best checkpoint: epoch 37; validation patch AUROC 0.833.
- Oracle and predicted-mask evaluation are reported separately.
- Include the real reliability plot from `results/final/real/`.

Presenter: Not patient-level presence/absence classification; all patients in MSD have tumors.

## Slide 7 — Stage 6.4 uncertainty

| Measure | Oracle ROI | Predicted masks |
|---|---:|---:|
| Patch AUROC | 0.823 | 0.846 |
| Sub-2 cm AUROC | 0.810 | 0.811 |
| ECE | 0.2556 | 0.2412 |
| Patch coverage (target 0.90) | 0.899 | 0.929 |
| Mean set size | 1.542 | 1.571 |

Note: In both columns the tumor (positive) window is placed from the ground-truth tumor; only the tumor-free windows come from the ground-truth vs the predicted pancreas. So this comparison does not measure whether the system finds the tumor inside a predicted mask - the patient-level Stage 6.6 run (sliding windows over the predicted pancreas, as in the app) is the end-to-end test.

Presenter: Calibration/coverage is patch-level; applying it to the app's maximum window is
approximate, not a patient-level coverage guarantee.

## Slide 8 — Stage 6.5 explanation fidelity

- Grad-CAM++ maps are checked by top-k masking against random masking.
- Pass rate: 47.0% overall (79/168).
- Sub-2 cm: 54.5% (6/11); ≥2 cm: 71.0% (22/31).
- Include a real saliency/F-Fidelity plot from `results/final/stage65/`.

Presenter: Do not present every heatmap as verified; the app suppresses failed maps.

## Slide 9 — Stage 6.6 selective prediction

- 42 validation patients, all tumor-bearing.
- 18 accepted; 24 flagged for specialist review (57.1% abstention).
- Accepted accuracy: 100% on this cohort.
- Sub-2 cm: 4/11 accepted; 7/11 abstained.
- Include detected-small and abstained UI screenshots.

Presenter: Accepted accuracy on a tumor-only set does not establish specificity or clinical
performance.

## Slide 10 — Integrated demo

- Use the local Gradio app with the real checkpoints.
- Show one small-tumor detection, one larger-tumor detection, and one abstention.
- Offline screenshots are in `results/final/screenshots/`.

Presenter: Keep the screenshots as fallback if the app or GPU is unavailable.

## Slide 11 — Limitations and responsible interpretation

- Small-tumor Dice 0.202; whole-pancreas Dice below target.
- Patch-level classification; no healthy-patient specificity.
- Validation reuse; no fourth independent split.
- Approximate case-level conformal behavior.
- F-Fidelity fails on 53% of patches overall.
- No PanTS external validation or clinical validation.

## Slide 12 — Conclusion and future work

- End-to-end six-stage pipeline and local demo are implemented.
- Results support a technical prototype, not a clinical claim.
- Next: independent tumor-negative cohort, external validation, better small-lesion segmentation,
  independent patient-level testing, case-level calibration, and clinician-supervised evaluation.
