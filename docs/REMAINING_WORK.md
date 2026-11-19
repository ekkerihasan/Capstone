# Remaining Work: 19 Nov to Final Review (25 Nov)

Team: Wasih and Hasan. Mustafa is on the paper; send him numbers as they land.
Branch: `feat/stage63-msd-negatives`. Earlier plan: `docs/FINAL_2DAY_PLAN.md`.

## Status on 19 Nov
| Stage | State | Evidence |
|---|---|---|
| 6.1 Preprocessing | Done, real data | `results/review1_local/full281/` |
| 6.2 Attention U-Net | **Trained, real data**, 200 epochs, 14 h on the RTX 4060 | `results/final/metrics_stage62.csv`, `fig_stage62_training.png`, `stage62_summary.json` |
| 6.3 Classifier | Trained by Hasan on Colab, **not shared yet** | — |
| 6.4 MC Dropout + conformal | Oracle run by Hasan (ground-truth ROIs); real run pending | — |
| 6.5 Grad-CAM++ / F-Fidelity | Code done, wired into the app; no offline batch script for the classifier | — |
| 6.6 Abstention + Gradio UI | Code done and wired to the 6.3 classifier; not yet run on real weights | — |

### Stage 6.2 results (42 validation patients)
| | Pancreas Dice | Tumor Dice | Mean |
|---|---|---|---|
| `best.pt` (epoch 125) | **0.697** | **0.415** | **0.556** |
| Mean of the last 20 epochs | 0.704 | 0.389 | 0.546 |

- Pancreas Dice is below the 0.75 target. Say so at the review.
- The tumor class was learned only from about epoch 45 onwards. Both curves are flat from about epoch 110, so more epochs would not help at this setup.
- `best.pt` was chosen on `val`, and `val` is also the 6.4 test group (the split has no fourth group). State this as a limitation.

---

## 1. Hasan, now
- [ ] Put these on the USB or Drive. `.pt` files are gitignored, so pushing will **not** carry them:
  - `checkpoints/stage63/best.pt`
  - `results/final/oracle/` (cls_predictions.csv + stage 6.4 outputs)
  - any 6.5 figures already made
- [ ] Push any code or notebook changes to `feat/stage63-msd-negatives`.
- [ ] Write down the 6.3 training settings you used: `--model`, `--roi-size`, `--neg-per-case`, `--dropout`, `--epochs`, and the best val AUROC.

## 2. Lab PC: compute steps (Wasih)
Open cmd in the `Capstone` folder. `F:` is the USB here; use whatever letter it gets on the lab PC.

**Step 1: update the code**
```bat
git pull
```
If it says "diverged" or "not possible to fast-forward" (the commits were rewritten on 18 Nov), run once:
```bat
git fetch
git reset --hard origin/feat/stage63-msd-negatives
```

**Step 2: 6.2 masks and Dice tables**, a few minutes each on the GPU:
```bat
.venv\Scripts\python -m src.predict_seg --split results/review1_local/full281/split_281cases.json --group val --save-masks results/final/masks --out results/final/seg_val
.venv\Scripts\python -m src.predict_seg --split results/review1_local/full281/split_281cases.json --group calib --save-masks results/final/masks --out results/final/seg_calib
```
Check that `results\final\masks` holds 84 `.npy` files. `results\final\seg_val\seg_by_size_bucket.csv` is **the 6.2 table for the deck** (overall and sub-2 cm).

**Step 3: Hasan's classifier into place**
```bat
mkdir checkpoints\stage63
copy F:\best.pt checkpoints\stage63\best.pt
xcopy /E /I /Y F:\oracle results\final\oracle
```
Adjust the source paths to where Hasan's files actually are.

**Step 4: real end-to-end, 6.3 to 6.4, on predicted masks**
```bat
.venv\Scripts\python -m src.predict_cls --split results/review1_local/full281/split_281cases.json --pred-masks results/final/masks --mc 20 --groups val calib --out results/final/real
.venv\Scripts\python -m src.run_stage64 --predictions results/final/real/cls_predictions.csv --out results/final/real
```
- If it fails with "needs a checkpoint trained with --dropout > 0", rerun the first command without `--mc 20` and note that MC Dropout is absent.
- `results\final\real` = the headline numbers: AUROC, sensitivity, specificity, conformal coverage, abstention rate, reliability diagram.

**Step 5: demo screenshots**
```bat
.venv\Scripts\python -m src.app
```
- Open http://127.0.0.1:7860. The banner must list both checkpoints and say **Real CT**.
- Screenshot three cases: a sub-2 cm tumor that was detected, a large tumor, and one **flagged for specialist review**. For each, capture the verdict, the segmentation with the yellow window, and the saliency panel.
- Save the screenshots to `results\final\screenshots\`. Stop the app with Ctrl+C.

**Step 6: take everything home**
```bat
xcopy /E /I /Y results\final F:\capstone\results_final
```

## 3. Code still to write (laptop, no GPU needed)
- [ ] `src/run_stage65.py`: batch Grad-CAM++ + F-Fidelity over the real 6.3 classifier's val patches. Output: a fidelity pass-rate table and 3–4 saliency figures for the deck. **Without it, 6.5's only evidence is the app screenshots.**
- [ ] Notebooks `02`–`06`, one per stage as CLAUDE.md requires. These are thin wrappers that call `src/` and display the saved `results/final/` figures; no training inside them.
- [ ] Update the status table in `CLAUDE.md`, which still says "Current task (3–4 Sept)".
- [ ] Open a PR from `feat/stage63-msd-negatives` to `master` and merge it once the real numbers are in.

## 4. Write-up (20–24 Nov)
- [ ] **Results tables** from `results/final/`:
  - 6.2: Dice overall and sub-2 cm, from `seg_val`;
  - 6.3: AUROC, oracle vs real;
  - 6.4: coverage, set size, abstention rate per size bucket;
  - 6.5: fidelity pass rate;
  - 6.6: accuracy on accepted cases vs abstention rate.
- [ ] **Report chapters:** Implementation, Testing (`python -m src.run_test_report` regenerates the test report from a real pytest run), Results, Limitations, Conclusion and Future Work.
- [ ] **Final deck:** problem → pipeline → data → one slide per stage with a real figure → demo screenshots → limitations → future work.
- [ ] **Logbook:** catch up the entries since 7 Oct (the 10-day rule), plus one for training and one for results.
- [ ] Send Mustafa the tables and figures for the paper.
- [ ] Rehearse once, timed, with an offline backup: deck as PDF, screenshots, and the app runnable on a laptop.

## Say this honestly at the review
- **6.3 is patch-level** (tumor vs non-tumor *tissue*), because every MSD patient has a tumor. The negatives come from tumor-bearing patients. A healthy-pancreas test (PanTS) is future work.
- **Oracle vs real:** report both separately. The gap between them is the cost of segmentation error.
- **Conformal on max-window:** q̂ is calibrated on patches but applied to the highest-scoring window in the app, so its coverage guarantee is approximate there.
- **Validation reuse:** 6.2 model selection and the 6.4 test both use `val`.
- **Figures:** each one is either real (`results/final/`) or carries the phantom stamp. Never mix them.
