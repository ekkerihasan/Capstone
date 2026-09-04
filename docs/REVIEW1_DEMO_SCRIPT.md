# Review 1 — 5-Minute Demo Script

**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Phase-II · Review 1 · 4 Sept 2026 · AITM Bhatkal, Dept. of CSE
Two presenters. Target 5:00, hard ceiling 5:30. Everything below traces to a file in the repo.

> **Speaker 1 (S1)** — narrative: problem, premise, scope, plan. Suggested: Hasan.
> **Speaker 2 (S2)** — implementation: the notebooks, the numbers. Suggested: Wasih or Mustafa.

---

## Before you walk in

- [ ] Repo open in the browser at `github.com/ekkerihasan/Capstone`
- [ ] `notebooks/00_pants_metadata_eda.ipynb` open, scrolled to the figure
- [ ] `notebooks/01_preprocessing_and_baseline.ipynb` open, scrolled to the overlay figures
- [ ] `results/review1/fig_pants_metadata.png` open in an image viewer (fallback if GitHub won't render)
- [ ] `results/review1_local/overlay_pancreas_001.png` open in an image viewer
- [ ] PDF of the deck + notebook HTML **on a pen drive** — assume the room has no internet
- [ ] Logbook printed, Project File in hand

**Do not run anything live.** Every figure below is pre-computed and committed.

---

## 0:00 – 1:00 · S1 — The problem, and the number behind it

> "Pancreatic cancer is usually found too late. The reason is visible in the data: tumours under 2 cm are nearly isodense with normal pancreas on CT — the radiologist is looking for something almost the same brightness as the tissue around it.
>
> We wanted to know whether that's actually the common case or the rare one, so we measured it in two independent datasets.
>
> In **PanTS**, the external-validation set, the metadata has no tumour-size column at all — the sizes are buried in free-text radiology reports. We parsed them out. Of the tumours we could measure, **37.7% are under 2 cm**.
>
> In **MSD Task07 Pancreas**, our training set, we computed size directly from the segmentation masks — a completely different method. **39.1% are under 2 cm.**
>
> Two datasets, two measurement methods, **1.4 percentage points apart**. The sub-2 cm case isn't an edge case. It's roughly two in five."

**On screen:** `results/review1/fig_pants_metadata.png`

**Point at:** the "37.7%" tile, then the red 2 cm line on the histogram.

---

## 1:00 – 1:30 · S1 — Scope, stated up front

> "Six stages in the pipeline. For Review 1 we are showing **two**: preprocessing, and the segmentation baseline.
>
> Stage 6.1 is **complete and running on real CT volumes**. Stage 6.2's pretrained baseline is **still running** — I'll be precise about that when we get to it.
>
> Stages 6.3 to 6.6 have **not been started**. They're Review 2 and the final review.
>
> Three things we've deliberately moved out of Phase-II scope: **MRI**, blood **biomarkers**, and training on **PanTS at full scale** — 36,390 scans, about 300 GB, which needs cluster hardware. We use PanTS to *validate* instead, which is the stronger test of generalisation anyway."

**Why this slide exists:** it pre-empts the three questions the panel would otherwise ask, and it establishes that you know exactly what is and isn't done.

---

## 1:30 – 3:30 · S2 — The implementation

### 1:30 – 1:50 · Repo and environment

**On screen:** the GitHub repo.

> "Everything is in one repository. Notebooks numbered by pipeline stage, `src/` for reusable code, `tests/`, and `results/` — one folder per review. Every number we quote today comes out of a file in `results/`.
>
> PyTorch and MONAI throughout. No TensorFlow anywhere."

### 1:50 – 2:20 · The data, and the split

**On screen:** notebook 01, cell 6 and cell 8 outputs.

> "MSD Task07 Pancreas: 281 labelled portal-venous CTs with pancreas and tumour masks. 11.4 GB, downloaded onto a Colab T4 in eleven minutes.
>
> We work on a 20-case subset today; the full 281 run on the college machine.
>
> The split is **patient-wise** — each MSD case is one patient, so no patient appears in both train and validation. It's saved to JSON with a fixed seed so every later stage scores the same held-out patients.
>
> One deliberate choice: we force the held-out cases to **contain tumours**, and we bias towards the small ones. A random draw could have given us four tumour-free cases, and then a tumour Dice score would mean nothing."

**Point at:** the four held-out cases —
```
pancreas_172  1.50 cm   sub-2 cm
pancreas_028  1.56 cm   sub-2 cm
pancreas_331  1.81 cm   sub-2 cm
pancreas_391  6.45 cm   >= 2 cm
```

> "Three small tumours, one large one for contrast. **Six of our twenty cases are under 2 cm.**"

### 2:20 – 3:10 · Stage 6.1 — preprocessing, on screen

**On screen:** the overlay figures in notebook 01 (or `results/review1_local/overlay_pancreas_001.png`).

> "This is Stage 6.1 output on a real volume. Four steps, all from Chapter 6 of the report:
>
> **One — resampling.** Scanners vary from 0.4 to 5 mm in-plane and up to 10 mm between slices. Everything is resampled to a uniform **1.5 × 1.5 × 2.0 mm** grid, so a voxel means the same physical size in every case.
>
> **Two — orientation.** All volumes reoriented to **RAS**, so left and right are consistent.
>
> **Three — HU windowing.** We clip to **−100 to 240 Hounsfield units** — the soft-tissue window — and normalise to 0–1. That throws away bone and air and keeps the contrast range the pancreas actually lives in.
>
> **Four — augmentation**, on training data only: random flips, 90-degree rotations, elastic deformation.
>
> Left panel is the preprocessed CT. Middle is the mask — pancreas and tumour. Right is the contour overlay: **green is pancreas, red is tumour.**"

**Point at:** the red contour. Then:

> "That's what a sub-2 cm pancreatic tumour looks like. It is almost the same grey as the tissue around it. That's the problem, in one picture."

### 3:10 – 3:30 · Scale and speed

**On screen:** `results/review1_local/full281/raw_stats.csv` or the summary JSON.

> "We also ran the full pipeline over all **281 cases** to characterise the dataset: every case carries a tumour, **110 of them — 39.1% — are under 2 cm**, median diameter 2.23 cm, range 0.92 to 11.18 cm.
>
> That run takes **74 seconds** on CPU. The pipeline is not a bottleneck; the model training will be."

---

## 3:30 – 4:00 · S2 — Stage 6.2, stated honestly

> "Stage 6.2 is the segmentation step — pancreas and duct. Our target model is an **Attention U-Net**, and training it is Weeks 1–2.
>
> Before we train anything, we're establishing a baseline with MONAI's pretrained `pancreas_ct_dints_segmentation` bundle, run on our four held-out cases and scored against ground truth.
>
> **That run is still going.** We are not going to show you a number we haven't got.
>
> Two things about that baseline, so there's no confusion: it is **not our model** — it's a published pretrained network. And its tumour Dice will be low. That's expected, and it's the point: it's the bar our Attention U-Net has to beat at Review 2."

**If the baseline finished before the review, replace the above with:**

> "The baseline scored Dice ___ on pancreas and ___ on tumour across our four held-out cases. That's the published network's performance, not ours — it's the bar our Attention U-Net has to beat."

---

## 4:00 – 4:40 · S1 — What's done, and what's next

> "So, Review 1, concretely:
>
> **Done** — environment, repository, PyTorch and MONAI on a Colab T4 with the college GPU arranged. PanTS metadata analysed, 9,901 scans. Stage 6.1 complete and demonstrated on real CT volumes, with a patient-wise split saved and reused. Baseline running.
>
> **Not done, and not claimed** — stages 6.3 through 6.6.
>
> **Next**, to Review 2: train the Attention U-Net on the full 281 cases and beat the baseline. Then the ROI classifier with sub-2 cm evaluation. Then calibrated uncertainty — MC Dropout plus conformal prediction.
>
> At Review 2 we'll have stages 6.1 to 6.4 end-to-end on MSD, checked externally on a PanTS shard."

---

## 4:40 – 5:00 · S1 — Close

> "One line on why this project isn't just another segmentation model. Of the papers we reviewed, none combines all three of: a **calibrated confidence**, an explanation that is **verified** rather than just plotted, and the ability to **abstain** — to say 'I don't know, send this to a specialist.'
>
> For a disease where a missed sub-2 cm tumour costs a patient their surgical window, a model that knows when to keep quiet is worth more than one extra point of accuracy."

---

## Questions to have ready

**"Why not train on PanTS?"**
> 300 GB and an 8 GB GPU. We use its test shard for external validation, which tests generalisation better than training on it would.

**"Why is the tumour Dice low?"**
> That's the pretrained baseline's published performance, not our model. Small isodense tumours are exactly the hard case this project exists to address — the number *is* the problem statement, in numeric form.

**"Is 30% really done?"**
> Two of six stages with running code and measured outputs, plus the dataset and infrastructure work. Stage 6.1 is complete on real data; 6.2's baseline is running.

**"Where's the MRI part?"**
> Future work. No suitable annotated MRI dataset is available to us this phase.

**"What's novel?"**
> None of the papers we reviewed combines calibrated confidence, faithfulness-verified explanation, and abstention in one pipeline. Each solves one piece.

**"Only 20 cases?"**
> 20 for today's subset run, but we characterised all 281 — that's where the 39.1% comes from. The full training set is 281 on the college GPU.

**"How do you know a tumour is under 2 cm?"**
> Equivalent spherical diameter from the mask volume: `d = (6V/π)^(1/3)`. For PanTS we parsed the long axis out of the radiology report text instead, because the metadata has no size column.

**"What if the panel asks to see it run?"**
> Notebook 00 runs in under a minute on CPU. Offer that one — never the 12 GB download.

---

## Hard rules (from the gameplan, §9)

1. Do **not** claim any of stages 6.3–6.6 exist.
2. Do **not** present the pretrained baseline as our trained model.
3. Do **not** claim PanTS full-scale training.
4. Do **not** claim MRI support.
5. Every number on a slide must trace to a file in `results/`.

---

## Number → file map

| Number | File |
|---|---|
| 37.7% sub-2 cm, 29 iso-attenuating, 29.2% coverage | `results/review1/pants_metadata_summary.json` |
| PanTS figure | `results/review1/fig_pants_metadata.png` |
| 9,901 scans, 1,077 tumour cases | `results/review1/pants_metadata_summary.json` |
| 39.1% sub-2 cm across 281, median 2.23 cm | `results/review1_local/full281/raw_stats.csv` |
| 6 of 20 sub-2 cm, the four held-out cases | `notebooks/01_preprocessing_and_baseline.ipynb`, cells 8 & 10 |
| Overlay figures | `results/review1_local/overlay_pancreas_*.png` |
| Patient-wise split | `results/review1_local/full281/split_281cases.json` |
| 74 s for 281 cases | `results/review1_local/full281/stage61_summary.json` |

---

## Timing discipline

Five minutes disappears fast. Two rules:

- **S1 does not touch the code, S2 does not re-explain the problem.** Overlap is what blows the budget.
- If you're at 3:30 and still on preprocessing, **skip the 281-case scale paragraph** and go straight to Stage 6.2. The overlay figure is the thing worth the time; the statistics table is not.

Rehearse it out loud once, timed. Reading it silently always feels shorter than it is.
