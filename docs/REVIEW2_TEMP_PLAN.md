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
| 6.2 Segmentation | 20 | ~40% - all modules written, 43 tests pass, nothing run on real data | 8 |
| 6.3 Classifier | 20 | Not started | 0 |
| 6.4 Uncertainty | 15 | Not started (the 197/42/42 split exists, nothing else) | 0 |
| 6.5 Explainability | 12 | Not started | 0 |
| 6.6 Selective prediction + UI | 13 | Not started | 0 |
| Report / deck / logbook | 10 | ~40% - plan, roadmap, Review 1 deck, 2 logbook entries | 4 |
| | **100** | | **~22** |

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
| P0 | `src/phantom.py` | Lift the generator out of `tests/` so scripts and notebooks can use it, not only pytest. Unblocks everything below | 20 min |
| P0 | `src/roi.py`, `src/classifier.py`, `src/train_cls.py` | The 6.3 chain. On 64x64x40 phantoms this trains on CPU in seconds | 2-3 h |
| P1 | `src/uncertainty.py`, `src/conformal.py`, ECE + reliability in `src/metrics.py` | Stage 6.4. Post-hoc on logits, pure CPU, and the source of the two best figures in the deck | 2 h |
| P1 | `src/run_demo_pipeline.py` | One command: 6.1 -> 6.4 on phantoms -> `results/review2_demo/`, every figure stamped | 1 h |
| P2 | `src/app.py` (Gradio shell) | Volume in -> mask, confidence, abstain decision. Highest visual impact per hour | 1-2 h |
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
