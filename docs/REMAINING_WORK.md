# Final Review Plan: status and remaining work (review on 25 Nov)

**Last updated 22 Nov 2026.**
- **Team:** Wasih and Hasan. Mustafa is on the paper; send him numbers as they land.
- **Branch:** `feat/stage63-msd-negatives` (all work below is pushed there).
- **History:** `docs/FINAL_2DAY_PLAN.md` (17–19 Nov plan).

---

## Hasan: catch-up summary
1. **6.2 is done on real data.** Wasih trained the Attention U-Net on the lab RTX 4060: 200 epochs, about 14 h, 18–19 Nov. He then evaluated it on all 84 val and calib patients and saved the predicted masks.
2. **Stages 6.3 to 6.6 are blocked on one file: your `checkpoints/stage63/best.pt`.** It is about 46 MB for densenet121 (57 MB for resnet10). Send it by Drive link, or on WhatsApp or Telegram as a *Document*. If you trained with the plan's command, it is already on your Drive at `MyDrive/capstone/checkpoints/stage63/best.pt`.
3. Also send your `results/final/oracle/` folder if you have it (a few MB), and the 6.3 training settings you used (see "What we still need from Hasan").
4. **Once the file arrives,** the rest runs on Wasih's laptop in about an hour (section 3).

---

## Status
| Stage | State | Evidence in the repo |
|---|---|---|
| 6.1 Preprocessing | ✅ Done, real data (281 cases) | `results/review1_local/full281/` |
| 6.2 Attention U-Net | ✅ **Trained and evaluated, real data** | `results/final/metrics_stage62.csv`, `fig_stage62_training.png`, `stage62_summary.json`, `seg_val/`, `seg_calib/` |
| 6.3 Classifier | ✅ Trained by Hasan on Colab. ⏳ **File not shared yet** | — |
| 6.4 MC Dropout + conformal | ⏳ Code done; real run waiting for the 6.3 file | — |
| 6.5 Grad-CAM++ / F-Fidelity | ⏳ Code done, including a batch script (`src/run_stage65.py`); waiting for the 6.3 file | — |
| 6.6 Abstention + Gradio UI | ⏳ Code done and wired to the 6.3 classifier; waiting for the 6.3 file | — |
| Tests | ✅ 219 pass, all on CPU with no data | `python -m pytest -q` |

### Stage 6.2 results: Attention U-Net, `best.pt` (epoch 125), real MSD
| Group | Whole-pancreas Dice | Tumor Dice | Tumor ≥ 2 cm | **Tumor < 2 cm** |
|---|---|---|---|---|
| val (42 patients) | 0.736 | 0.415 | 0.491 (n=31) | **0.202** (n=11) |
| calib (42 patients) | 0.734 | 0.399 | 0.479 (n=27) | **0.255** (n=15) |

- **Whole-pancreas Dice** counts labels 1 + 2 (organ including tumor). The training log's **0.697** counts label 1 only (tissue excluding tumor). Both are correct; recomputed from the masks, the latter is 0.6972. Always say which one you quote.
- **Below target:** 0.736 is under the 0.75 pancreas target. Say so.
- **Small tumors are the weak point:** tumor Dice for tumors under 2 cm is about half that for larger ones, while pancreas Dice is the same for both. This is exactly the problem stages 6.3 to 6.6 address.
- **Training behaviour:** the tumor class was learned only from about epoch 45. Both curves are flat from about epoch 110.

---

## Who trained what, and where it lives
Ask the owner if you need a file that is not in git. `.pt` files and the masks are gitignored.

| Artifact | Made by | How | Where it is | Size |
|---|---|---|---|---|
| MSD Task07 dataset (tar) | Wasih (downloaded) | `msd-for-monai` S3 | Wasih's laptop `repo/data/Task07_Pancreas.tar`, USB `F:\` | 12.3 GB |
| 6.1 preprocessing run (281 cases) | Hasan | Colab + `src/run_preprocessing.py`, 4 Sept | in git: `results/review1_local/full281/` | small |
| **6.2 Attention U-Net `best.pt` / `last.pt`** | **Wasih** | lab RTX 4060 via `lab.bat`, 18–19 Nov, 200 epochs, 96³ patches, batch 2, AMP | Wasih's laptop `repo/checkpoints/stage62/`; USB `F:\capstone\checkpoints\stage62\`; lab PC `C:\Users\datas\Documents\capstone-pancreas\checkpoints\stage62\` | 71 MB each |
| 6.2 predicted masks, val + calib (84 `.npy`) | Wasih | laptop CPU, `predict_seg --save-masks`, 22 Nov | Wasih's laptop `repo/results/final/masks/` (not in git) | ~700 MB |
| 6.2 Dice tables and figures | Wasih | as above | in git: `results/final/seg_val/`, `seg_calib/` | small |
| **6.3 classifier `best.pt`** | **Hasan** | Colab T4, `train_cls --sampling patch` | Hasan's Drive, probably `MyDrive/capstone/checkpoints/stage63/` | ~46 MB (densenet121) |
| 6.3/6.4 oracle results | Hasan | Colab | Hasan's `results/final/oracle/` | small |

### Who wrote the code (from `git log`)
- **Mustafa (Oct):** 6.3 classifier, ROI crop, training and inference scripts.
- **Wasih (Oct):** 6.4 MC Dropout, conformal prediction, calibration metrics.
- **Hasan (Oct):** phantom harness, 6.1→6.4 demo pipeline, 6.5 Grad-CAM++ and F-Fidelity, 6.6 abstention rule and Gradio app, test report.
- **Wasih (Nov), this branch:**
  - patch-level negatives for real MSD;
  - OS-independent splits;
  - the 6.2→6.3 mask hand-off;
  - the `lab.bat` launcher with pause/resume;
  - the 6.3 classifier wired into the app;
  - the `run_stage65` batch script;
  - notebook 07;
  - `predict_seg --reuse-masks`.

### What we still need from Hasan
- [ ] `checkpoints/stage63/best.pt`
- [ ] `results/final/oracle/`, if made
- [ ] The 6.3 settings, for the report:

| Setting | Value used |
|---|---|
| `--model` | |
| `--roi-size` | |
| `--neg-per-case` | |
| `--dropout` | |
| `--epochs` | |
| best val AUROC (and epoch) | |
| number of `only k/n tumor-free windows fit` warnings | |

- [ ] Any code or notebook changes made on Colab, pushed to `feat/stage63-msd-negatives`.

---

## 3. Remaining compute: Wasih's laptop, about 1 h on CPU, after the 6.3 file arrives
The data (84 val and calib cases) and the 6.2 masks are already on the laptop.
1. Put Hasan's file at `repo\checkpoints\stage63\best.pt`.
2. Run 6.3 → 6.4 twice, oracle (ROIs from the ground truth) and real (ROIs from our 6.2 masks):
```bat
.venv\Scripts\python -m src.predict_cls --split results/review1_local/full281/split_281cases.json --mc 20 --groups val calib --out results/final/oracle
.venv\Scripts\python -m src.run_stage64 --predictions results/final/oracle/cls_predictions.csv --out results/final/oracle
.venv\Scripts\python -m src.predict_cls --split results/review1_local/full281/split_281cases.json --mc 20 --groups val calib --pred-masks results/final/masks --out results/final/real
.venv\Scripts\python -m src.run_stage64 --predictions results/final/real/cls_predictions.csv --out results/final/real
```
   If it says "needs a checkpoint trained with --dropout > 0", drop `--mc 20` and note that MC Dropout is absent.
3. Run 6.5:
```bat
.venv\Scripts\python -m src.run_stage65 --split results/review1_local/full281/split_281cases.json --ckpt checkpoints/stage63/best.pt --pred-masks results/final/masks --group val --out results/final/stage65
```
4. Run the 6.6 demo with `.venv\Scripts\python -m src.app` and open http://127.0.0.1:7860. Each case takes about 70 s on the laptop CPU.
   - Screenshot three cases: a detected sub-2 cm tumor, a large tumor, and one **flagged for specialist review**. Save them to `results/final/screenshots/`.
   - For a **live** demo at the review, use the lab PC (4060) or notebook 07 on Colab. The laptop is too slow to demo live.

**Fallbacks:**
- Lab PC: the same commands, after `git pull`.
- Colab: `notebooks/07_final_results_colab.ipynb` through the VS Code Colab extension.

---

## 4. Code and docs still to do
- [ ] Notebooks `02`–`06`, one per stage. Thin wrappers that call `src/` and show the saved `results/final/` figures; no training inside them.
- [ ] Update the status table and the "Current task" section in `CLAUDE.md` (still says 3–4 Sept).
- [ ] Open a PR from `feat/stage63-msd-negatives` to `master` and merge it once the real numbers are in.

## 5. Write-up (by 24 Nov)
- [ ] **Results tables:**
  - 6.2 Dice (above, done);
  - 6.3 AUROC, oracle vs real;
  - 6.4 coverage, set size and abstention per size bucket;
  - 6.5 fidelity pass rate;
  - 6.6 accuracy on accepted cases vs abstention rate.
- [ ] **Report chapters:** Implementation, Testing (`python -m src.run_test_report` regenerates the test report), Results, Limitations, Conclusion and Future Work.
- [ ] **Final deck:** problem → pipeline → data → one slide per stage with a real figure → demo screenshots → limitations → future work.
- [ ] **Logbook:** catch up the entries since 7 Oct, plus one for training (18–19 Nov) and one for results.
- [ ] Send Mustafa the tables and figures.
- [ ] One timed rehearsal, with an offline backup: deck PDF, screenshots, and the app runnable.

## Say this honestly at the review
- **6.3 is patch-level** (tumor vs non-tumor *tissue*), because every MSD patient has a tumor. The negatives come from tumor-bearing patients. A healthy-pancreas test (PanTS) is future work.
- **Oracle vs real:** report both. The gap between them is the cost of segmentation error.
- **Conformal on max-window:** q̂ is calibrated on patches but applied to the highest-scoring window in the app, so its coverage guarantee is approximate there.
- **Validation reuse:** 6.2 model selection and the 6.4 test both use `val`; the split has no fourth group.
- **Below-target pancreas Dice:** 0.736 whole-organ vs the 0.75 target. Small-tumor Dice is the clear weakness.
- **Figures:** each one is either real (`results/final/`) or carries the phantom stamp. Never mix them.
