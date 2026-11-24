# Final Review Report — Phase-II Results Addendum

**Project:** AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)<br>
**Dataset:** Medical Segmentation Decathlon (MSD), Task07 Pancreas<br>
**Review date:** 25 November 2026

This addendum contains the Phase-II implementation, testing, results, limitations, conclusion,
and future-work material. It supplements the approved Phase-I report; it does not replace or
reconstruct its literature-review and design chapters.

## Implementation

The prototype follows the six numbered stages in Chapter 6. Stage 6.1 resamples portal-venous
CT, reorients it to RAS, clips the stated HU window and normalizes it, then uses patient-wise
train/validation/calibration partitions. MSD Task07 contains 281 labelled cases; the saved
partition is 197 train, 42 validation, and 42 calibration cases.

Stage 6.2 is a three-class Attention U-Net for background, pancreas, and tumor. Stage 6.3 is a
3D DenseNet121 **patch classifier**, not a patient-level tumor-presence detector. Training
samples one tumor window and up to three tumor-free pancreatic tissue windows per patient. Its
recorded checkpoint configuration is 32×32×32 voxels, dropout 0.2, seed 42, and a best
validation AUROC checkpoint at epoch 37 (AUROC 0.833); the last checkpoint is epoch 39.

Stage 6.4 applies 20-pass MC Dropout and split-conformal sets to classifier patches. Stage 6.5
computes Grad-CAM++ and compares top-saliency masking against random masking; a failed
F-Fidelity check suppresses the map. Stage 6.6 combines the available confidence, conformal,
and fidelity signals into an accept-or-abstain decision and exposes the pipeline through a
local Gradio interface.

## Testing

The CPU test suite uses synthetic MSD-shaped phantoms and verifies preprocessing, data splits,
training/checkpoint behavior, patch sampling, metrics, uncertainty, conformal prediction,
explainability, fidelity, app behavior, and stage runners. The final count and per-test outcomes
are recorded by the generated [test report](../results/final/test_report.md).

These are software-behavior tests, not evidence of clinical performance. They do not establish
generalization to healthy patients, performance on another dataset, or clinical safety.

## Results

### Stage 6.2 — segmentation

| Group | Whole-pancreas Dice | Tumor Dice | Tumor ≥2 cm | Tumor <2 cm |
|---|---:|---:|---:|---:|
| Validation (42 cases) | 0.736 | 0.415 | 0.491 (n=31) | 0.202 (n=11) |
| Calibration (42 cases) | 0.734 | 0.399 | 0.479 (n=27) | 0.255 (n=15) |

Whole-pancreas Dice counts pancreas and tumor labels together. The validation score is below
the 0.75 target. The much lower small-tumor Dice is a key failure mode.

### Stages 6.3–6.4 — patch classification and uncertainty

The 6.3 best checkpoint reached 0.833 validation patch AUROC at epoch 37. On the 168 validation
patches, the Stage 6.4 runner reported the following oracle (ground-truth ROI) and predicted-mask
results:

| Measure | Oracle ROI | Predicted masks |
|---|---:|---:|
| AUROC, all patches | 0.823 | 0.846 |
| AUROC, sub-2 cm positives vs all negatives | 0.810 | 0.811 |
| Sensitivity at 0.5 | 0.929 | 0.929 |
| Specificity at 0.5 | 0.619 | 0.603 |
| ECE (15 bins) | 0.2556 | 0.2412 |
| Patch conformal coverage (nominal 0.90) | 0.899 | 0.929 |
| Mean conformal set size | 1.542 | 1.571 |
| Fraction with non-singleton/empty set | 0.542 | 0.571 |

Note: In both columns the tumor (positive) window is placed from the ground-truth tumor; only the tumor-free windows come from the ground-truth vs the predicted pancreas. So this comparison does not measure whether the system finds the tumor inside a predicted mask - the patient-level Stage 6.6 run (sliding windows over the predicted pancreas, as in the app) is the end-to-end test.

Size-stratified predicted-mask coverage was 0.968 for tumors ≥2 cm (n=31), 1.000 for tumors
<2 cm (n=11), and 0.913 for tumor-free tissue patches (n=126). The small-tumor result has only
11 positive patches and should not be read as a stable patient-level guarantee.

### Stage 6.5 — fidelity-checked explanations

F-Fidelity passed for 79/168 validation patches (47.0%). It passed for 28/42 tumor patches
(66.7%), 51/126 tumor-free tissue patches (40.5%), 6/11 sub-2 cm tumor patches (54.5%), and
22/31 tumors ≥2 cm (71.0%). Only passing saliency maps are shown by the app.

### Stage 6.6 — selective prediction

On the 42 validation cases, 18 were accepted and 24 abstained (57.1%). Accuracy among accepted
cases was 100%. For the 11 sub-2 cm cases, 4 were accepted and 7 abstained (63.6%); accuracy
among those four accepted cases was 100%.

Every MSD validation case in this experiment has a tumor. Therefore the accepted accuracy is
conditional on this tumor-bearing cohort and is not an estimate of specificity or performance
on healthy patients.

## Limitations

1. **Small-lesion segmentation:** sub-2 cm tumor Dice is 0.202; the whole-pancreas Dice of
   0.736 also misses its 0.75 target.
2. **Patch-level classification:** the classifier distinguishes tumor from tumor-free pancreas
   tissue in tumor-bearing patients. It does not answer whether a patient has cancer.
3. **No healthy-patient evaluation:** MSD Task07 has tumors in every case. PanTS external
   validation was not completed.
4. **Validation reuse:** 6.2 checkpoint selection and 6.4 evaluation both use the `val` group;
   the available split has no fourth independent test group.
5. **Approximate app calibration:** conformal calibration is patch-level, but the app applies
   the threshold to the maximum-scoring window. The patch guarantee does not transfer directly
   to this patient/case-level aggregation.
6. **Calibration and explanation quality:** ECE is approximately 0.24–0.26; F-Fidelity passes
   for fewer than half of all evaluated patches.
7. **Small sample sizes:** only 11 sub-2 cm validation cases/patches contribute to relevant
   small-tumor subgroup estimates.
8. **Research prototype only:** the pipeline is not clinically validated or intended to guide
   diagnosis or treatment.

## Conclusion

The project implements and evaluates a six-stage CT pipeline on MSD Task07, including
segmentation, tissue-patch classification, uncertainty estimation, explanation checks, and an
abstaining interface. Real-data results show that the pipeline runs end-to-end, but performance
is limited by small-tumor segmentation, imperfect calibration, and a low fidelity pass rate.
The selective interface abstains on more than half of the validation cases, and its reported
accepted accuracy comes from a tumor-only cohort. These results demonstrate a research
prototype, not a clinically reliable patient-level cancer detector.

## Future work

- Obtain an independent, patient-level external validation cohort including tumor-negative
  controls; evaluate PanTS only under its permitted external-validation role.
- Add an independent patient-level test split before making generalization claims.
- Improve and independently validate sub-2 cm lesion segmentation.
- Revisit calibration at the case-level aggregation actually used by the app.
- Improve explanation fidelity and evaluate its relation to clinically meaningful tumor regions.
- Conduct prospective, clinician-supervised usability and safety evaluation before any clinical
  use.

## Reproducibility pointers

- Metrics and size-bucket tables: `../results/final/`
- Saved checkpoint settings: `../checkpoints/stage63/best.pt` (local/ignored; do not commit)
- Test command: `python -m src.run_test_report --out results/final`
- Explicit remaining actions: `./REMAINING_WORK.md`
