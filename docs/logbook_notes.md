# Project Logbook — Phase-II

Entry every 10 days per the department circular. Newest entry at the top.
Each entry: dates covered, stage worked on, what exists as evidence, what is blocked.

---

## Entry #2 — 4 Sept 2026 · Stage 6.2 training loop

**Stage:** 6.2 ROI segmentation — Attention U-Net.

**Done**
- `src/models.py` — `build_attention_unet()`, 3-class output (bg/pancreas/tumor), 5.9 M parameters,
  channels (16,32,64,128,256) sized for 96^3 patches at batch 2 inside 8 GB with AMP.
- `src/checkpoints.py` — atomic save (temp file + rename, so a power cut cannot truncate `last.pt`),
  `resume_if_possible()` returning (start_epoch, best_metric, history).
- `src/train_seg.py` — the full loop: DiceCELoss, AdamW + cosine schedule, AMP on CUDA,
  `CacheDataset`, sliding-window validation, per-class Dice, per-epoch checkpoint **and** per-epoch
  metrics CSV, training-curve figure, `--resume`.
- 16 new tests (43 total, all passing on CPU in ~27 s, no dataset and no GPU needed).

**Two defects found and fixed by running it**
1. `RandCropByPosNegLabeld` raised "proposed random crop ROI is larger than the image size" when a
   volume was under 96 voxels in an axis after foreground cropping — would have crashed mid-epoch on
   the lab PC. Fixed with `SpatialPadd` in front of the sampler.
2. `scheduler.load_state_dict()` restores `T_max` from the checkpoint, so resuming a short run with
   a longer `--epochs` made the cosine LR anneal to zero and then climb back up. The run's horizon
   now overrides the checkpoint's. Regression test asserts the LR decays monotonically across a resume.

**Verified**
- 3 epochs on a phantom, interrupted, resumed to epoch 5: history continued rather than restarting,
  loss fell monotonically, `last.pt`/`best.pt` both written, training-curve figure spans the resume.

**Still not done**
- No real MSD run yet: no Task07 data on this laptop and no GPU here. Stage 6.2 needs the RTX 4060
  (or a Colab T4) for the actual training run and the pancreas Dice >= 0.75 target.
- MONAI DiNTS bundle baseline still unexecuted.

**Next**
- Book the lab PC; download MSD; run `src.run_preprocessing` then `src.train_seg` for real.
- Then stage 6.3 (ROI crop -> 3D ResNet/DenseNet classifier), Weeks 3-4.

---

## Entry #1 — 3 Sept 2026 · Review 1 preparation · Stage 6.1

**Stage:** 6.1 Data preprocessing (and the pretrained baseline for 6.2).

**Done**
- Repository set up: `notebooks/`, `src/`, `docs/`, requirements pinned, data and checkpoints gitignored.
- Phase-II plan written (`docs/Phase-II_Project_Plan.md`): milestones, data strategy, week-by-week schedule, work split, risks.
- Stage 6.1 pulled out of the notebook into an importable, tested package:
  - `src/config.py` — the Chapter 6 constants in one place (1.5/1.5/2.0 mm, HU -100...240, seed 42, 96^3 patches).
  - `src/transforms.py` — the report's preprocessing pipeline plus train-only augmentation and 96^3 patch sampling.
  - `src/data.py` — MSD datalist loading (skipping the macOS `._` files inside the official tar), patient-wise split, split JSON save/load, leakage guard.
  - `src/metrics.py` — Dice, voxel volume, equivalent spherical diameter, sub-2 cm bucketing.
  - `src/viz.py`, `src/run_preprocessing.py` — review figures and a CPU-only end-to-end runner.
- `tests/` — the pipeline is verified against a synthetic MSD-shaped phantom, so stage 6.1 can be checked on any laptop without the 12 GB download.
- Notebooks written for Review 1: `00_pants_metadata_eda.ipynb`, `01_preprocessing_and_baseline.ipynb`.

**Not yet done / honest status**
- The two notebooks have **not been executed** — no cell outputs, no `results/review1/` from real MSD data.
  They need one Colab T4 session: MSD download (~12 GB), preprocessing figures, DiNTS baseline Dice on the held-out cases.
- Stage 6.2 Attention U-Net training has not started (Weeks 1-2).

**Blocked on**
- Lab RTX 4060 booking; PanTS test shard + `PanTSMini_Label.tar.gz` download (Week 2).

**Next**
- Run `01_preprocessing_and_baseline.ipynb` on Colab end-to-end and commit `results/review1/`.
- Begin `02_segmentation_attention_unet.ipynb` against `src.transforms.build_transforms(train=True, patches=True)`.
