# Final Review: 2-Day Plan (two people, real data, stages 6.2 to 6.6)

Written 17 Nov 2026. Final review: **25 Nov**. Real-data results due by the evening of **19 Nov**.
The days from 20 to 24 Nov are buffer for the report, deck, rehearsal and logbook.
**Team for these two days: Wasih and Hasan.** Mustafa is writing the paper; send him numbers as they land.

## Starting point
- All six stages are coded. Branch `feat/stage63-msd-negatives`: **204/204 tests pass** (17 Nov).
- Only 6.1 has run on real data so far (281 MSD cases). Everything else is still phantom-only.
- Changes on this branch since the Review 2 code:
  - **6.3 negatives:** `--sampling patch`. One jittered tumor window per case plus `--neg-per-case` tumor-free windows of the same pancreas. A test guarantees that no negative window touches the tumor.
  - **Split paths fixed:** the 281-case split was saved on Windows and would not have loaded on Colab (Linux). It now rebases onto `--data-root` on any OS.
  - **6.2 to 6.3 hand-off:** `predict_seg --save-masks DIR` writes the predicted masks, and `predict_cls --pred-masks DIR` cuts ROIs from them. This is the real end-to-end path.
- The split to use everywhere is `results/review1_local/full281/split_281cases.json`: 197 train / 42 val / 42 calib, patient-wise.

## Lab PC rules (9 am to 5 pm, no overnight)
- Training saves `checkpoints/stage62/last.pt` after **every epoch**. Stopping loses at most the current epoch.
- **5 pm:** press Ctrl+C. Copy `checkpoints/` and `results/final/` to the USB drive.
- **9 am:** run the same command with `--resume` added. **Never change `--epochs`** after the first start, because the LR schedule is built on it.

## Tonight (17 Nov)
| Who | Task |
|---|---|
| Wasih | Copy `data/Task07_Pancreas.tar` (12.3 GB) to a USB drive, along with the repo. |
| Hasan | Pull `feat/stage63-msd-negatives`. Open a Colab notebook and check the T4 is assigned. |

## Day 1: 18 Nov
### Wasih: lab RTX 4060, stage 6.2
```bash
git clone https://github.com/ekkerihasan/Capstone.git && cd Capstone && git checkout feat/stage63-msd-negatives
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124    # before requirements
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0))"
mkdir data && tar -xf E:\Task07_Pancreas.tar -C data          # E: = the USB drive
python -m pytest -q                                             # expect 204 passed
```
**9:45, timing run.** Run 2 epochs to measure the time per epoch:
```bash
python -m src.train_seg --data-root data/Task07_Pancreas \
  --split results/review1_local/full281/split_281cases.json \
  --out results/final --epochs 2 --ckpt-root checkpoints_timing
```
**About 10:15, the real run.** Set `E = floor(13 h × 3600 / seconds_per_epoch)`. That is about 13 h of training across both days. Write E down and use the same E both days.
```bash
python -m src.train_seg --data-root data/Task07_Pancreas \
  --split results/review1_local/full281/split_281cases.json \
  --out results/final --epochs E
```
- If you get an OOM error: add `--batch 1`. If it still fails: `--patch 64`.
- Watch `results/final` for the per-epoch CSV. The pancreas Dice should be rising by the afternoon.
- 5 pm: Ctrl+C, then back up `checkpoints/` and `results/final/`.

**Wasih, while the GPU trains:**
1. **Wire the 6.3 classifier into `src/app.py`.** About 2 to 3 hours, and it is the main code task left. Today the app reads p(tumor) off the *segmentation* tumor channel. The `--data-root` default is the phantom, and `PATCH` is hard-coded to 32. The app needs to:
   - load `stage63/best.pt`;
   - slide `roi_size` windows over the predicted pancreas;
   - set p(tumor) = max over the windows, with MC Dropout on that window;
   - run Grad-CAM++ and F-Fidelity on the classifier.
2. Notebooks `02`–`06` as thin wrappers around `src/` (CLAUDE.md requires one per stage).
3. Logbook entries #5 and #6.

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
| Time | Wasih (lab PC) | Hasan (Colab + laptop) |
|---|---|---|
| 9:00 | `--resume` 6.2 with the same E | 6.5 (Grad-CAM++ + F-Fidelity) on the 6.3 model; save figures |
| 14:00 | Stop. Use `best.pt` for both runs below. | Deck skeleton for the final review |
| 14:15 | `predict_seg --split … --group val --save-masks results/final/masks --out results/final`, then the same with `--group calib` | |
| 15:00 | Copy `results/final/masks` to Hasan (USB or Drive) | **Real end-to-end:** `predict_cls … --pred-masks results/final/masks --out results/final/real --mc 20`, then `run_stage64` on that output |
| 16:00 | Gradio UI (`src/app.py`) on the trained weights; screenshots | Fill numbers into the deck and report; send them to Mustafa |

**Fallback.** If our 6.2 model is weak at 14:00 (pancreas Dice below about 0.6), produce the masks with DiNTS instead. 6.3 to 6.6 do not care which model drew the mask. Present the Attention U-Net as the comparison.

## Say this honestly at the review
- 6.3 is **patch-level**: tumor vs non-tumor *tissue*, because every MSD patient has a tumor. The negatives come from tumor-bearing patients. A healthy-pancreas external test (PanTS) is future work unless time remains.
- The oracle numbers (ground-truth ROI) and the real numbers (predicted ROI) are reported separately. The gap between them is the cost of segmentation error.
- Tumor Dice on MSD is low even for state-of-the-art methods. Always report the sub-2 cm bucket separately.
- 6.2 picks its best epoch on `val`, and 6.4 also tests on `val`. State this as a limitation, since the split has no fourth group.
- Every figure is either real (`results/final/`) or carries the phantom stamp. Never mix them.
