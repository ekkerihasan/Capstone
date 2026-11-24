# Project Logbook — Phase-II

Entry every 10 days per the department circular. Newest entry at the top.
Each entry: dates covered, stage worked on, what exists as evidence, what is blocked.

---

## Entry #7 — 23–24 Nov 2026 · Real-data evaluation and final review preparation

**Stages:** 6.3–6.6 evaluation, result verification, and final-review documentation.

**Evidence recorded**
- Stage 6.3 best checkpoint: DenseNet121, 32×32×32 ROI, 3 tumor-free tissue windows per case, dropout 0.2; best validation patch AUROC 0.833 at epoch 37. The checkpoint metadata records the final saved epoch as 39.
- Stage 6.4 patch-level validation: oracle ROI AUROC 0.823 and predicted-mask AUROC 0.846. ECE was 0.2556 and 0.2412 respectively. Observed conformal coverage was 0.899 and 0.929 against a nominal 0.90 target.
- Stage 6.5 F-Fidelity passed for 47.0% of 168 patches overall and 54.5% of 11 sub-2 cm tumor patches.
- Stage 6.6 evaluated 42 tumor-bearing MSD validation cases: 18 accepted, 24 abstained; accepted accuracy was 100%. For the 11 sub-2 cm cases, 4 were accepted and 7 abstained.
- Tables, summaries, figures, and demo screenshots are stored under `results/final/`. The final test report is generated from the current full suite.

**Limitations stated**
- 6.3 is a patch classifier for tumor-containing versus tumor-free pancreatic tissue, not patient-level tumor-presence detection.
- MSD contains tumor-bearing cases only; patient-level specificity is not measured.
- Conformal calibration is patch-level but the UI uses the maximum-scoring window. The UI coverage is approximate.
- The same `val` group was used for 6.2 model selection and 6.4 evaluation; there is no fourth split.
- PanTS external validation was not completed; no clinical-readiness claim is made.

**Still requiring people/external systems**
- Send the results handoff to Mustafa.
- Review the report and deck, verify the PDF/PPTX in the USB bundle, and run a timed rehearsal on the demo computer.
- Push the branch, open a PR, and merge after review.

---

## Entry #6 — 18–22 Nov 2026 · Real-MSD segmentation and integration

**Stages:** 6.2 training/evaluation and real-data integration for 6.3–6.6.

- Wasih trained the Attention U-Net on the lab RTX 4060 for 200 epochs; the selected checkpoint is epoch 125. Evaluation on val and calibration groups produced val whole-pancreas Dice 0.736 and tumor Dice 0.415; sub-2 cm tumor Dice was 0.202 (n=11). The whole-pancreas score is below the 0.75 target.
- A predicted-mask handoff was added for 6.3 inference. The lab launcher supports resumable training and checkpoint backup.
- The stage 6.3 classifier checkpoint was supplied from the Colab training run; real-MSD oracle and predicted-mask prediction files were generated.
- Stage 6.5 batch evaluation and stage 6.6 case-level evaluation ran using real checkpoints. See `results/final/` for evidence.

**Limitations/blockers**
- The selected 6.2 checkpoint does not meet the pancreas Dice target, and small-lesion segmentation remains weak.
- PanTS external validation was not run.
- Do not describe the 6.3 patch results or conformal coverage as patient-level guarantees.

---

## Entry #5 — 8 Oct–16 Nov 2026 · Documentation gap

The repository history contains no commits after 7 Oct until 17 Nov, and no dated run logs or result artifacts were found for this interval. This entry records the gap rather than reconstructing undocumented activity. The final review plan resumed on 17 Nov.

---

## Entry #4 — 7 Oct 2026 · Stages 6.4 and 6.6 built against the phantom

**Period covered:** 6–7 Oct 2026.
**Stage:** 6.4 calibrated uncertainty, 6.6 selective prediction and UI. Both on phantom data.

**Why this is possible without the GPU**
`roi.py` and the stages after it take a *mask*; they do not care which model produced it. So
6.4 and 6.6 were built and tested against the synthetic phantom and ground-truth labels while
6.2 waits for the lab 4060. The code does not change when real data arrives — only the input
does. Roughly 70 % of the remaining work was code that needed no GPU; this entry is that 70 %
being taken.

**Done — Wasih, stage 6.4**
- `src/metrics.py` extended: AUROC, confusion, sensitivity/specificity, reliability curve, ECE,
  and `auroc_by_size_bucket`. The last exists because AUROC is not a per-case mean — a sub-2 cm
  AUROC has to rank the small positives against *all* negatives, so `by_size_bucket` cannot
  produce it.
- `src/uncertainty.py`: MC Dropout at T=20, with `enable_dropout` setting p explicitly because
  MONAI leaves the Attention U-Net gates at 0.0.
- `src/conformal.py`: split conformal with q_hat the ceil((n+1)(1-alpha))-th smallest
  nonconformity score, and q_hat = 1.0 when the calibration set is too small for the requested
  alpha — the method refuses to promise what the data cannot support.
- `src/run_stage64.py`: one command from `cls_predictions.csv` to the summary, the four CSVs and
  the two deck figures.
- 35 tests, including the marginal-coverage guarantee on exchangeable data.

**Reviewed, two defects open**
1. `enable_dropout(model, p)` permanently changes the model's dropout rate — eval mode is
   restored, p is not. Harmless for inference, wrong if MC Dropout is ever called from a
   validation step inside training.
2. `predictive_entropy` sums over the last axis. Correct for a classifier's [N, C], silently
   wrong for a segmentation model's [N, C, D, H, W], which the roadmap plans to use.

**Done — Hasan, integration and stage 6.6**
- `src/phantom.py`: the generator moved out of `tests/` so the demo and the tests share one
  phantom; `tumor_radius` and `pancreas_axes` added, because the default phantom pancreas is
  only ~1.6 cm across and every tumor inside it was sub-2 cm, leaving the size-bucket table
  with an empty row.
- `src/run_demo_pipeline.py`: 6.1 -> 6.4 in one command, ~96 s on CPU, 37 artifacts.
  `manifest.json` records which stages were real; stage 6.3 is a flagged placeholder until that
  lane lands.
- `src/run_test_report.py`: the test report is generated from an actual pytest run, so it
  cannot drift away from the suite.
- `src/selective.py` and `src/app.py`: the abstention rule, and the Gradio UI that serves it.
  Verified against a running server. An alpha slider reads the calibrated q_hat sweep, so the
  coverage trade-off is demonstrable live.

**The honest result**
With the current two-epoch checkpoint the UI abstains on nearly every case. That is the system
working: an untrained model refuses rather than guesses. Every figure and every screen carries
the phantom stamp, and the app states on each result that the weights are untrained, that no
6.3 classifier exists, and that the fidelity signal is absent.

**Status**
~40 % by weight, up from ~22 % on 6 Oct. 125 tests pass on CPU.

**Still blocked / outstanding**
- Stage 6.3 (Mustafa) — now the single critical path: 6.2 needs no mask from it, but the real
  numbers for 6.4 and 6.6 do.
- The lab RTX 4060 is still unbooked. Nothing in this entry is a result.
- No Review 2 deck yet, and no report chapters.
- The two open decisions from entry #3 — negative sampling for 6.3, and duct scope — are still
  open and still free to settle.

---

## Entry #3 — 6 Oct 2026 · Status reassessment, phantom harness, work split

**Period covered:** 5 Sept – 6 Oct 2026.
**Stage:** nothing advanced on real data. Groundwork laid for 6.3 and 6.4.

**Gap in the record — stated plainly**
No commits between 4 Sept and 6 Oct. Review 1 was delivered on 4 Sept and development stopped
there: the lab RTX 4060 booking was never secured, and stage 6.2 cannot produce a real result
without it. This entry records the gap rather than filling it, because the commit history is
public and a logbook that disagreed with it would be worse than a logbook with a hole.

**Status reassessment (6 Oct)**
- Roughly **22%** of the technical work is complete measured by what has actually *run*;
  nearer **27%** on the VTU review scale, which also credits plan, repo and deck.
- 6.1 preprocessing — complete, on all 281 real MSD cases.
- 6.2 segmentation — all modules written, 56 tests pass on CPU, **never run on real data**.
- 6.3 through 6.6 — not started.

**Defect found in the Review 1 record**
Cells 18–24 of `notebooks/01_preprocessing_and_baseline.ipynb` — the whole MONAI DiNTS
baseline section — carry no outputs. The Colab session stopped after section 4 (preprocessing
figures on a 20-case subset); the full-281 numbers came from `src/run_preprocessing.py`
separately. **Stage 6.2 therefore has no real results at all, baseline included.** It must not
be listed as done in the Review 2 deck.

**Done today**
- `docs/REVIEW2_TEMP_PLAN.md` — interim plan: build every stage against synthetic phantoms
  now, swap in real data when the GPU frees. Includes the rule that every phantom-derived
  number and figure is stamped "SYNTHETIC PHANTOM — PIPELINE DEMONSTRATION, NOT A RESULT".
- `src/phantom.py` — the synthetic MSD-shaped generator lifted out of `tests/` so the demo
  pipeline and notebooks can use it, not only pytest; `tests/conftest.py` now imports it, so
  there is one implementation. Half the phantom cases carry no tumor, which gives stage 6.3
  the negatives that real MSD does not have, and the tumor stays near-isodense (35 HU in a
  55 HU pancreas) so the task is not trivially separable on intensity.
- Three-lane work split with the module interfaces frozen in advance — Mustafa on 6.3,
  Wasih on 6.4 and the training run, Hasan on docs, integration and the UI.

**Blocked on**
- **Lab RTX 4060 booking.** Still the single hard blocker: no trained segmentation model means
  no ROI, and no ROI means 6.3 through 6.6 cannot run on real data. Kaggle Notebooks
  (12 h sessions, datasets mounted rather than downloaded) is the fallback; the atomic
  checkpoint and resume in `src/checkpoints.py` was built for exactly that.

**Two decisions still open — both cost nothing now and stall training later**
1. Negative samples for 6.3: every real MSD case has a tumor, so negatives must come from
   PanTS `tumor? = 0` cases or from non-tumor pancreas crops.
2. Duct segmentation: `CLAUDE.md` scopes 6.2 as "pancreas + duct", but MSD Task07 labels only
   background / pancreas / tumor and duct annotations live in PanTS, the *validation* set.
   Settle with the guide; the clean answer is probably to record duct as future work.

**Next**
- Three lanes in parallel against the phantom, no lane waiting on another.
- Book the 4060 (or start the Kaggle run) — this is the item that moves the percentage.

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
