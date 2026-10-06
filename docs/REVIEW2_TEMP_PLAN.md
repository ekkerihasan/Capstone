# Interim Plan - Synthetic-Phantom Demo Route to Review 2

**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Team: Hasan Ekkeri (2AB23CS045), Mustafa Hajeeb (2AB23CS047), Abdul Wasih (2AB23CS006)
Guide: Mrs. Arzoo, Dept. of CSE, AITM Bhatkal

Written 6 Oct 2026. **Temporary plan** - it covers the stretch between now and the real
training run on the college RTX 4060. Supersedes nothing; `docs/Phase-II_Project_Plan.md`
remains the schedule of record and `docs/MODULE_ROADMAP.md` the module list.

---

## 1. Where the project actually stands

Measured by what has been *run*, not what has been written.

| Stage | Weight | State | Earned |
|---|---|---|---|
| 6.1 Preprocessing | 10 | Complete - ran on all 281 real MSD cases | 10 |
| 6.2 Segmentation | 20 | Code + tests done and running in the demo chain, but never on real data | 8 |
| 6.3 Classifier | 20 | Not started - a placeholder is not progress on 6.3 | 0 |
| 6.4 Uncertainty | 15 | MC Dropout, split conformal, ECE and coverage implemented and tested; run on synthetic input only | 9 |
| 6.5 Explainability | 12 | Not started, deferred past Review 2 | 0 |
| 6.6 Selective prediction + UI | 13 | Abstention rule and Gradio UI built and running on phantoms | 8 |
| Report / deck / logbook | 10 | Plans, roadmap, logbook current, generated test report; **no Review 2 deck, no report chapters** | 5 |
| | **100** | | **~40** |

*Updated 7 Oct 2026. It was ~22 on 6 Oct, before the 6.4 and 6.6 lanes landed.*

On the VTU review scale it reads nearer **27%**, because Review 1's 30% includes process
marks (plan, repo, literature, deck) that were earned.

**Correction to the Review 1 record.** In `notebooks/01_preprocessing_and_baseline.ipynb`,
cells 18-24 - the entire MONAI DiNTS baseline section - have no outputs. The Colab run
stopped after section 4 (preprocessing figures, 20-case subset); the full-281 numbers came
from `src/run_preprocessing.py` separately. **Stage 6.2 has zero real results, baseline
included.** Do not list the DiNTS baseline as done in the Review 2 deck.

---

## 2. The constraint

Training needs the college PC (RTX 4060, 8 GB). It is not available before the next review.
So the plan is to build and wire **every** stage end-to-end against synthetic phantoms now,
and swap in real data when the GPU is free. The code does not change between the two.

### Why the existing phantom is good enough for this

`tests/conftest.py` already generates MSD-shaped volumes with:

- the correct label convention (0 background / 1 pancreas / 2 tumor),
- deliberately wrong spacing (0.8/0.8/2.5) and orientation, so preprocessing has real work,
- `with_tumor=(i % 2 == 0)` - **3 positive and 3 negative cases**, which is exactly what the
  6.3 present/absent classifier needs,
- a near-isodense tumor (35 HU inside a 55 HU pancreas), as in real PDAC - not a trivially
  easy toy.

---

## 3. The hard rule

> Every number and every figure produced from phantom data carries the stamp
> **"SYNTHETIC PHANTOM - PIPELINE DEMONSTRATION, NOT A RESULT"**, on the figure itself and
> on the slide.

Dice on a phantom blob will be ~0.9 and AUROC will be 1.0. Those numbers mean nothing. An
unlabelled one that a panel member spots puts every other claim in doubt. Labelled, it reads
as engineering discipline - the integration harness was built before the data was available.

The real stage 6.1 results on 281 cases stay clearly separated as the genuine results.

---

## 4. Build order

| Pri | Module | Why | Estimate |
|---|---|---|---|
| P0 | `src/phantom.py` | **DONE 6 Oct.** Generator lifted out of `tests/`; `conftest.py` now imports it, so tests and demo share one phantom. Also provides `DEMO_STAMP` and `stamp_figure()` | - |
| P0 | `src/roi.py`, `src/classifier.py`, `src/train_cls.py` | The 6.3 chain. **Outstanding - Mustafa's lane, and the critical path for everything else** | 2-3 h |
| P1 | `src/uncertainty.py`, `src/conformal.py`, ECE + reliability in `src/metrics.py` | **DONE 6 Oct (Wasih, fd1e46b).** Two defects open: `enable_dropout` does not restore `p`, and `predictive_entropy` assumes the class axis is last | - |
| P1 | `src/run_demo_pipeline.py` | **DONE 6 Oct.** 6.1 -> 6.4 in ~96 s on CPU, 37 artifacts, `manifest.json` records which stages were real | - |
| P2 | `src/app.py` (Gradio shell) | **DONE 7 Oct**, with `src/selective.py` for the abstention rule. Verified against a running server; screenshots in `results/review2_demo/screenshots/` | - |
| P2 | `src/explain.py` (Grad-CAM++) | Can be stubbed for now | - |

P0 + P1 is one working evening and yields a demonstrable 6.1 -> 6.4 chain.

### Develop 6.3 and 6.4 against ground-truth masks

`roi.py` takes a mask and returns a crop; it does not care which model produced it. So on real
data the same two paths exist:

```
oracle path   GT mask  -> roi.py crop -> classifier -> uncertainty   <- build and validate this first
real path     6.2 mask -> roi.py crop -> classifier -> uncertainty   <- swap in when 6.2 is trained
```

This is not a shortcut. It separates classifier error from segmentation error, and the gap
between the two rows is a genuine result for the report ("ROI quality costs us X AUROC").

---

## 5. What the phantom will NOT catch

Training is not a formality once the modules exist. These only surface on real data:

- **Convergence.** The Dice >= 0.75 target may need LR and epoch tuning. A phantom converging
  says nothing about 281 real volumes.
- **Negative sampling for 6.3.** Every real MSD case has a tumor. The phantom supplies
  negatives for free; reality does not. Negatives must come from PanTS `tumor? = 0` cases or
  from non-tumor pancreas crops. **Still undecided - settle before training.**
- **Memory.** Phantoms are 64x64x40. Real volumes at 96^3 patches on 8 GB is where
  `--batch 1` and `--patch 64` get exercised.
- **Calibration.** ECE on real logits is often poor out of the box and needs temperature
  scaling. Synthetic logits look perfectly calibrated because we generated them.

Rough split of the work that remains: **~70% is code that can be written without a GPU,
~30% is runs and tuning that need the 4060.**

---

## 6. Open decisions (cost nothing now, stall training later)

1. **Negative samples for 6.3** - PanTS `tumor? = 0` cases, or non-tumor pancreas crops from
   MSD. Decide before `train_cls.py` is finalised.
2. **Duct segmentation.** `CLAUDE.md` scopes 6.2 as "pancreas + duct", but MSD Task07 labels
   only background / pancreas / tumor. Duct annotations live in PanTS, which is the
   *validation* set. Settle with the guide; the clean answer is probably "duct scoped out of
   Phase-II, recorded as future work".

---

## 7. When the college PC is free

1. Measure one epoch first, then set the epoch budget from the measured number:
   `python -m src.train_seg --epochs 2 --max-steps 20 --limit-val 4`
2. Full run with `--val-interval 5` (sliding-window validation on 42 cases is the expensive
   part, not training). Every loop checkpoints per epoch and resumes from
   `checkpoints/stage62/last.pt`.
3. `python -m src.predict_seg --split <split.json> --group val` for Dice overall and for the
   sub-2 cm bucket.
4. Re-run `run_demo_pipeline.py` against real data - the stamped figures become real figures.

**Fallback if the lab PC slips again.** Kaggle Notebooks give 12 h sessions and ~30 GPU-h per
week with datasets mounted rather than downloaded - better than free Colab for a 7-10 h run.
The atomic checkpoint and resume in `src/checkpoints.py` was built for exactly this; a dropped
session costs an epoch, not a run.

---

## 8. Line for the panel

> All six stages are implemented and run end-to-end, verified on synthetic phantoms with the
> correct label convention and deliberate preprocessing faults. Stage 6.1 has real results on
> all 281 MSD cases. Training on real data is scheduled on the lab RTX 4060; every loop
> checkpoints and resumes, so a dropped session costs an epoch, not a run.

Do not hide the GPU booking as the blocker. It is a scheduling fact, not a design failure.

---

## 9. Work split - three members, three Claudes, one repo

Lanes follow the owners already named in `docs/Phase-II_Project_Plan.md` section 4, so the
guide sees the same allocation she has seen before. `src/phantom.py` is done and pushed, so
all three lanes start in parallel - nobody waits.

### Mustafa - Stage 6.3, the ROI and classifier chain

Owns, and is the only one who edits: `src/roi.py`, `src/classifier.py`, `src/train_cls.py`,
`src/predict_cls.py`, `tests/test_roi.py`, `tests/test_classifier.py`.

Deliverable: phantom -> ground-truth mask -> ROI crop -> 3D ResNet/DenseNet -> tumor
present/absent, with per-case probabilities written to CSV. Trains on CPU in seconds at
64x64x40, so this is a laptop job.

Build against the **ground-truth** mask, not a predicted one. `roi.py` takes a mask and does
not care which model produced it, so the same code serves the real run once 6.2 is trained.

### Wasih - Stage 6.4, calibration, plus the real training run

Owns, and is the only one who edits: `src/uncertainty.py`, `src/conformal.py`,
`src/metrics.py`, `tests/test_uncertainty.py`, `tests/test_conformal.py`,
`tests/test_metrics.py`.

Deliverable: MC Dropout at T=20, split-conformal calibration on the 42-case calibration
group, and the ECE / reliability-diagram / coverage figures. All of it is post-hoc on saved
probabilities, so it can be written and tested against synthetic logits before 6.3 produces
real ones.

**Do the `metrics.py` additions first and push them within the hour** - Mustafa's AUROC
reporting is blocked on them, and `metrics.py` has exactly one owner precisely so two Claudes
never rewrite it at once.

Wasih also owns the lab PC booking and the stage 6.2 training run when the GPU frees up.
One caveat already pinned by `tests/test_train_seg.py`: only the convolution blocks pick up
the `dropout` constructor argument, the attention gates stay at p=0.0. `enable_dropout()`
must set p on the modules it wants rather than trusting the constructor.

### Hasan - documentation first, then integration and the UI

Owns, and is the only one who edits: `src/run_demo_pipeline.py`, `src/app.py`,
`docs/logbook_notes.md`, `results/review2_demo/`, the Review 2 deck.

Order matters here, because integration cannot start until the other two lanes land:

1. **Logbook entries #3 and #4 - overdue.** The circular wants one every 10 days; the last is
   #2 on 4 Sept. Three are missing. Independent of all code, and the panel checks it.
2. **Test report** for `results/review2_demo/` - module, what is covered, result. "Testing so
   far" is an explicit Review 2 deliverable and 56 tests already pass.
3. **`run_demo_pipeline.py`** - one command runs 6.1 -> 6.4 on phantoms and writes every
   figure through `phantom.stamp_figure()`.
4. **`app.py`** - the Gradio shell. Volume in, mask + confidence + abstain decision out.
   Highest visual impact per hour, but only after the three above.

Stage 6.5 explainability (`src/explain.py`) stays Hasan's per the project plan, but it is P2
and deferred until after the review.

### Frozen interfaces

Three Claudes working in parallel will invent three incompatible APIs unless the signatures
are fixed in advance. These are fixed. Anyone who needs a change asks in the group first.

```python
# src/roi.py                                                   (Mustafa)
def pancreas_bbox(mask, label=LABEL_PANCREAS, margin_mm=10.0,
                  spacing_mm=SPACING_MM) -> tuple          # tuple of slices
def crop_roi(image, mask, out_size=(96, 96, 96), margin_mm=10.0,
             spacing_mm=SPACING_MM) -> dict                # {"image", "mask", "bbox", "spacing_mm"}

# src/classifier.py                                            (Mustafa)
def build_classifier(name="densenet121", in_channels=1,
                     num_classes=2, dropout=0.0)           # -> torch.nn.Module

# src/predict_cls.py  ->  results/<run>/cls_predictions.csv     (Mustafa)
#   columns: case, group, y_true, prob_tumor, eq_diam_cm
#   ^ this CSV is the contract between stage 6.3 and stage 6.4. Nothing else crosses the line.

# src/metrics.py  (additions)                                   (Wasih)
def auroc(y_true, y_score) -> float
def sensitivity_specificity(y_true, y_pred) -> tuple
def confusion(y_true, y_pred)                              # -> 2x2 ndarray
def expected_calibration_error(y_true, y_prob, n_bins=15) -> float
def reliability_curve(y_true, y_prob, n_bins=15) -> tuple  # (bin_centre, accuracy, confidence)

# src/uncertainty.py                                            (Wasih)
def enable_dropout(model, p=None) -> list                  # names of the modules switched on
def mc_dropout_predict(model, x, T=20, device=None) -> dict  # {"mean", "std", "samples"}

# src/conformal.py                                              (Wasih)
def calibrate(probs_calib, y_calib, alpha=0.1) -> float    # qhat
def predict_set(probs, qhat) -> list                       # one set of labels per case
def coverage(sets, y_true) -> float

# src/run_demo_pipeline.py                                      (Hasan)
#   imports all of the above, writes results/review2_demo/, stamps every figure
```

Existing names stay as they are: `metrics.by_size_bucket()` is metric-agnostic, so the sub-2 cm
reporting that `CLAUDE.md` requires comes free in 6.3 and 6.4 - do not reimplement it per stage.

### Rules that keep three Claudes from fighting

1. **Never edit a file in someone else's lane.** Ask in the group instead. A two-minute
   message beats a merge conflict in a file none of you have read.
2. **One branch per lane**: `feat/stage63-mustafa`, `feat/stage64-wasih`, `feat/demo-hasan`.
   `git pull --rebase origin master` before every push.
3. **`src/config.py` is shared.** New constants are append-only, and say so in the group
   before pushing.
4. **Tests ship with the code, in the same commit.** `.venv/Scripts/python.exe -m pytest -q`
   must pass before any push - 56 tests pass today and that number should only go up.
5. **No unstamped phantom figure leaves a lane.** Section 3 is not negotiable.
6. Give your Claude `CLAUDE.md` plus this section. The frozen interfaces above are what make
   three separately-written modules fit together on the first try.
