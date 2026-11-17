# Final Review: 2-Day Plan (two people, real data, stages 6.2 to 6.6)

Written 17 Nov 2026, updated 18 Nov. Final review: **25 Nov**. Real-data results due by the evening of **19 Nov**.
The days from 20 to 24 Nov are buffer for the report, deck, rehearsal and logbook.
**Team for these two days: Wasih and Hasan.** Mustafa is writing the paper; send him numbers as they land.

## Starting point
- All six stages are coded and wired. Branch `feat/stage63-msd-negatives`: **215/215 tests pass**.
- Only 6.1 has run on real data so far (281 MSD cases). Everything else is still phantom-only.
- Changes on this branch:
  - **6.3 negatives:** `--sampling patch`. One jittered tumor window per case plus `--neg-per-case` tumor-free windows of the same pancreas. A test guarantees no negative window touches the tumor.
  - **Split paths fixed:** the 281-case split was saved on Windows and would not have loaded on Colab (Linux). It now rebases onto `--data-root` on any OS.
  - **6.2 to 6.3 hand-off:** `predict_seg --save-masks DIR` writes the predicted masks, and `predict_cls --pred-masks DIR` reads them.
  - **`lab.bat`:** one command on the lab PC. It starts or resumes training, and Ctrl+C pauses safely (see below).
  - **App wired to 6.3:** `src/app.py` now uses the classifier. It segments, slides the classifier window over the pancreas, takes p(tumor) from the highest-scoring window, and runs MC Dropout, conformal, Grad-CAM++ and F-Fidelity on that window. If no classifier checkpoint exists, it falls back to the old segmentation-channel route.
- The split everywhere is `results/review1_local/full281/split_281cases.json`: 197 train / 42 val / 42 calib, patient-wise.

## The lab PC (9 to 5, no overnight)
**First visit only, once:**
```bat
git clone -b feat/stage63-msd-negatives https://github.com/ekkerihasan/Capstone.git
```
Then copy `Task07_Pancreas.tar` from the USB drive into `Capstone\data\`. There's no need to extract it.

**Every visit, the whole procedure:**
```bat
cd Capstone
lab.bat --backup E:\capstone
```
`E:` is the USB drive; leave out `--backup` if no drive is plugged in.

`lab.bat` does everything else:
- pulls the latest code;
- on the first run, creates `.venv` with the CUDA build of PyTorch (about 10 min);
- extracts the tar on the first run (about 10 min);
- starts training or resumes it automatically.

**Pausing:** press **Ctrl+C** whenever the lab has to be freed. Finished epochs are already saved; only the epoch in progress is lost, which is a few minutes. On exit it copies the checkpoints and metrics to the USB drive. If Windows asks "Terminate batch job (Y/N)?", either answer is fine.

**How long it trains:**
- The first session times one epoch on the 4060, then fixes the total number of epochs to fit **11 hours** of training. That is about 6.5 h today plus about 4.5 h tomorrow, finishing before the 14:00 inference.
- The plan is stored in `checkpoints/stage62/run.json`, so every resume continues the same learning-rate schedule. Nobody chooses or types an epoch count.
- Each epoch line shows `~X h left`.
- `lab.bat --status` shows progress without training.
- If you get an out-of-memory error: `lab.bat --fresh --batch 1`. If it still fails: add `--patch 64`.

## Day 1: 18 Nov
### Wasih: lab PC
- 9:00: clone, copy the tar, run `lab.bat --backup E:\capstone` (see above). Check that it prints `GPU: NVIDIA GeForce RTX 4060` and the line `epoch 0 took …s -> 11.0 h budget = N epochs`.
- While it trains, at the lab or on the laptop:
  1. Notebooks `02`–`06` as thin wrappers around `src/` (CLAUDE.md requires one per stage).
  2. Logbook entries #5 and #6 (about four entries are overdue).
- 17:00: Ctrl+C. The backup to the USB happens automatically.

### Hasan: Colab T4, stage 6.3 (plus the DiNTS baseline)
```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/ekkerihasan/Capstone.git && cd Capstone && git checkout feat/stage63-msd-negatives
%cd Capstone
!pip -q install -r requirements.txt
!wget -q https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar && mkdir -p data && tar -xf Task07_Pancreas.tar -C data && rm Task07_Pancreas.tar
!python -m src.train_cls --data-root data/Task07_Pancreas \
   --split results/review1_local/full281/split_281cases.json \
   --sampling patch --neg-per-case 3 --roi-size 32 --model densenet121 --dropout 0.2 \
   --epochs 40 --batch 16 --out results/final \
   --ckpt-root /content/drive/MyDrive/capstone/checkpoints
```
- Checkpoints go to Drive. If the session dies, re-run the setup cells and add `--resume`.
- Lines saying `only k/n tumor-free windows fit` are fine. If **most** cases print them, use `--roi-size 32,32,24`. That needs `--model resnet10`, because DenseNet needs at least 32 voxels per axis.
- `--dropout 0.2` is required. Without it, MC Dropout (6.4) has nothing to sample.

Then the oracle results for 6.3 and 6.4 (ROIs cut from ground-truth masks):
```python
!python -m src.predict_cls --split results/review1_local/full281/split_281cases.json \
   --data-root data/Task07_Pancreas --out results/final/oracle \
   --ckpt /content/drive/MyDrive/capstone/checkpoints/stage63/best.pt --mc 20
!python -m src.run_stage64 --predictions results/final/oracle/cls_predictions.csv --out results/final/oracle
```
Copy `results/final/` to Drive. **If time is left:** run DiNTS baseline inference on the 42 val cases (notebook 01, section 5) to get the baseline Dice for the comparison table.

## Day 2: 19 Nov
**Wasih, lab PC:**
- 9:00: `lab.bat --backup E:\capstone`. It resumes by itself, and should finish around 14:00 (the `~X h left` counter shows when).
- **14:00**, stop (Ctrl+C) if it hasn't finished. Then save predicted masks for the val and calib groups:
```bat
.venv\Scripts\python -m src.predict_seg --split results/review1_local/full281/split_281cases.json --group val   --save-masks results/final/masks --out results/final/seg_val
.venv\Scripts\python -m src.predict_seg --split results/review1_local/full281/split_281cases.json --group calib --save-masks results/final/masks --out results/final/seg_calib
```
`seg_val/seg_by_size_bucket.csv` holds **the 6.2 Dice table for the deck**.
- 15:00: copy `results/final/masks` to Drive for Hasan. Copy Hasan's `stage63/best.pt` into `checkpoints\stage63\` and his `results/final/real` folder into `results\final\real`.
eal`.
eal`.
- 16:00, **the demo** with real weights:
```bat
.venv\Scripts\python -m src.app
```
It picks up `checkpoints/stage62/best.pt`, `checkpoints/stage63/best.pt`, the real q-hat and the MSD data on its own. Open http://127.0.0.1:7860 and take screenshots for the deck.

**Hasan, Colab and laptop:**
- 9:00: 6.5 figures (Grad-CAM++ + F-Fidelity) on the 6.3 model. Start the deck skeleton.
- 15:00: **real end-to-end**, using Wasih's masks on Drive:
```python
!python -m src.predict_cls --split results/review1_local/full281/split_281cases.json \
   --data-root data/Task07_Pancreas --pred-masks /content/drive/MyDrive/capstone/masks \
   --ckpt /content/drive/MyDrive/capstone/checkpoints/stage63/best.pt --mc 20 \
   --groups val calib --out results/final/real
!python -m src.run_stage64 --predictions results/final/real/cls_predictions.csv --out results/final/real
```
- 16:00: put the numbers into the deck and report, and send them to Mustafa.

**Fallback.** If our 6.2 model is weak at 14:00 (pancreas Dice below about 0.6), produce the masks with DiNTS instead. 6.3 to 6.6 do not care which model drew the mask. Present the Attention U-Net as the comparison.

## Say this honestly at the review
- 6.3 is **patch-level**: tumor vs non-tumor *tissue*, because every MSD patient has a tumor. The negatives come from tumor-bearing patients. A healthy-pancreas external test (PanTS) is future work unless time remains.
- The oracle numbers (ground-truth ROI) and the real numbers (predicted ROI) are reported separately. The gap between them is the cost of segmentation error.
- Tumor Dice on MSD is low even for state-of-the-art methods. Always report the sub-2 cm bucket separately.
- 6.2 picks its best epoch on `val`, and 6.4 also tests on `val`. State this as a limitation, since the split has no fourth group.
- Every figure is either real (`results/final/`) or carries the phantom stamp. Never mix them.
