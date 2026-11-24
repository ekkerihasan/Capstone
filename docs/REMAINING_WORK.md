# Final review status (24 Nov 2026)

The real-MSD compute for stages 6.1–6.6 is complete. This file supersedes the
22 Nov snapshot in [REMAINING_WORK_2026-11-22.md](./REMAINING_WORK_2026-11-22.md).

## Completed in the workspace

- Stages 6.1–6.6 have real-MSD outputs under `results/final/`.
- Stage notebooks 02–06 are present and stage 07 contains the final-results Colab workflow.
- Stage 6.6 app screenshots are saved under `results/final/screenshots/`.
- The 6.3 checkpoint and settings are available locally; see `checkpoints/stage63/` and
  `results/final/`.
- The Phase-II report addendum and 12-slide outline are in `docs/FINAL_REPORT_ADDENDUM.md`
  and `docs/FINAL_REVIEW_DECK.md`; the PDF and PPTX are generated alongside them.
- The final full suite passes 222/222 tests; its report is `results/final/test_report.md`.
- The team handoff and review run sheet are prepared in `docs/`.

## Results to present

| Stage | Result |
|---|---|
| 6.2 | Val whole-pancreas Dice 0.736; sub-2 cm tumor Dice 0.202 (n=11). The 0.75 pancreas target was not met. |
| 6.3 | DenseNet121, 32³ ROI, 3 negatives/case, dropout 0.2; best checkpoint epoch 37, val patch AUROC 0.833. |
| 6.4 | Patch-level val AUROC: oracle 0.823, predicted masks 0.846; target 90% conformal coverage observed 0.899 / 0.929. |
| 6.5 | F-Fidelity pass rate 47.0% overall and 54.5% for the 11 sub-2 cm tumor patches. |
| 6.6 | 18/42 cases accepted; 57.1% abstention; accepted accuracy 100% on this tumor-only MSD validation set. |

Use the CSV/JSON artifacts in `results/final/oracle/`, `real/`, `stage65/`, and
`stage66/` for exact values and denominators.

## Still needs a person or external access

- Review/sign off the report and deck, and verify the exports on the review computer. The USB
  copy is at `D:\capstone\FINAL_REVIEW_2026-11-25\repo\`.
- Send [RESULTS_HANDOFF_FOR_MUSTAFA.md](./RESULTS_HANDOFF_FOR_MUSTAFA.md) to Mustafa; the
  workspace has no contact details or messaging integration.
- Run a timed human rehearsal and verify the app on the actual presentation computer. The local
  Gradio endpoint returned HTTP 200 on CUDA, but the human timed run is not performed.
- Push/open/merge the PR after reviewing the complete changes. No PR exists yet.

## Mandatory caveats

- Stage 6.3 is **patch-level tumor-vs-tumor-free pancreatic tissue**, not patient-level cancer detection.
- Every MSD Task07 patient has a tumor; the results do not estimate specificity on healthy patients.
- Stage 6.4 conformal calibration/testing is patch-level, while the app uses the maximum-scoring window; app-level coverage is approximate.
- 6.2 model selection and 6.4 evaluation both reuse `val`; this split has no fourth group.
- PanTS external validation was not completed. No clinical-readiness claim is supported.
- F-Fidelity failure suppresses the saliency map; report its measured pass rate rather than implying every map was verified.
