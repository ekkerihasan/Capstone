# Final review run sheet

**Status:** prepared for rehearsal; a human timed run on the review computer is still required.

## Before the review

- Copy [FINAL_REVIEW_DECK.pdf](./FINAL_REVIEW_DECK.pdf) and
  [FINAL_REVIEW_DECK.pptx](./FINAL_REVIEW_DECK.pptx) to the review USB.
- Copy `results/final/screenshots/` and keep the repository/results folder locally available.
- Confirm MSD data and `checkpoints/stage62/best.pt`, `checkpoints/stage63/best.pt` exist on the demo computer.
- Start the app from the repository root:

  ```bat
  .venv\Scripts\python.exe -m src.app --device cuda
  ```

- Open `http://127.0.0.1:7860` and test one large case, one sub-2 cm case, and one abstention before presenting.
- If the app or GPU is unavailable, use the saved screenshots. Do not claim the screenshots are a live run.

## 10-minute presentation timing

| Time | Slides | Focus |
|---|---|---|
| 0:00–1:00 | 1–2 | Title, problem, scope |
| 1:00–2:00 | 3–4 | Six stages, data and split |
| 2:00–3:30 | 5 | 6.2 Dice, below-target and small-tumor limitations |
| 3:30–5:00 | 6–7 | Patch classifier, oracle-vs-predicted ROI, uncertainty |
| 5:00–6:00 | 8 | Fidelity pass rate; explain map suppression |
| 6:00–7:30 | 9–10 | Selective prediction and short demo |
| 7:30–9:00 | 11 | Limitations and honest interpretation |
| 9:00–10:00 | 12 | Conclusion and future work |

## Required verbal caveats

- The classifier is patch-level, and every MSD case in this evaluation contains a tumor.
- Patch-level conformal coverage is not a patient-level guarantee when applied to the maximum window.
- Validation is reused for 6.2 checkpoint selection and 6.4 evaluation.
- PanTS external validation and clinical validation were not completed.
