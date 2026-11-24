# Offline final-review bundle

The bundle contains the repository source, final results, saved checkpoints, report, deck,
and screenshots. The extracted MSD dataset remains at `D:\Task07_Pancreas` on Wasih's USB
to avoid a second copy of the large dataset.

## Launch the local demo

From `D:\capstone\FINAL_REVIEW_2026-11-25\repo`, use the existing project environment on this
laptop:

```bat
cd /d D:\capstone\FINAL_REVIEW_2026-11-25\repo
"%USERPROFILE%\Documents\capstone-pancreas\.venv\Scripts\python.exe" -m src.app --data-root D:\Task07_Pancreas --checkpoint checkpoints\stage62\best.pt --classifier checkpoints\stage63\best.pt --device cuda
```

If that computer does not have CUDA, change `--device cuda` to `--device cpu`; CPU inference is
slower. On a different computer, first create a virtual environment and install dependencies
from `requirements.txt` with a compatible PyTorch build.

Open `http://127.0.0.1:7860`. If the app cannot be started, the three saved cases in
`results\final\screenshots\` are the offline visual fallback. They are saved results, not a
live demo.

## Review files

- `docs\FINAL_REVIEW_DECK.pdf` — offline presentation
- `docs\FINAL_REVIEW_DECK.pptx` — editable presentation
- `docs\FINAL_REPORT_ADDENDUM.md` — results and limitations chapter draft
- `docs\FINAL_REVIEW_REHEARSAL.md` — timed run sheet
- `results\final\test_report.md` — final automated test report (222/222 passed)
