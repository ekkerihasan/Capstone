# Review 2 — Presentation Script

**AI-Based Pancreatic Cancer Prediction Using Deep Learning (Early Detection from Medical Imaging Data)**
Phase-II · Review 2 · AITM Bhatkal, Dept. of CSE
Two presenters, 10 slides. Target 12:00 for the slides, 3:00 for the demo. Hard ceiling 16:00.

> **Speaker 1 (S1)** — the pipeline every detection system has: modules 6.1–6.3, plus the tests and the close. Suggested: Hasan.
> **Speaker 2 (S2)** — what makes ours different: modules 6.4–6.6, and the live demo. Suggested: Wasih or Mustafa.
> **Third member** — drives the laptop, has the app already running, handles the Project File.

The deck follows the structure the HOD asked for: one slide of all modules, one slide per module, output screenshots on every module slide, test cases, demo last.

---

## Before you walk in

- [ ] Deck exported to **PDF and PPTX**, both on a pen drive — assume the room has no internet
- [ ] `python -m src.app` **already running** in a terminal, browser tab open at `127.0.0.1:7860`, case `pancreas_000` already selected
- [ ] A second browser tab on `results/review2_demo/test_report.md` in case the test count is questioned
- [ ] Repo open at `github.com/ekkerihasan/Capstone` in a third tab
- [ ] Logbook printed and current (entries #1–#4), Project File in hand
- [ ] Phone hotspot ready as an internet fallback
- [ ] Battery: the demo is CPU-only and takes ~20 seconds per case — plug in anyway

**Do not launch the app during the presentation.** Start it before you walk in and leave the tab open. Everything else in the deck is a pre-computed image.

---

## 0:00 – 0:45 · S1 — Slide 1, cover

> "Good morning. Our project is AI-based pancreatic cancer prediction from CT, and the specific target is the small tumour — under 2 cm — because that is the stage where surgery still changes the outcome.
>
> This is our Review 2. We will walk through all six modules, show the output each one produces, cover the test cases, and finish with a live demo.
>
> The one line that distinguishes this project: our system does not only predict. It reports how confident it actually is, it proves its own explanation before showing it, and when it cannot tell, it says so instead of guessing."

---

## 0:45 – 2:15 · S1 — Slide 2, all six modules

> "This is the whole pipeline on one slide. A CT volume enters at 6.1 and leaves at 6.6 as one of three answers.
>
> **6.1 Preprocessing** makes every scan comparable — different machines produce different voxel sizes and intensity ranges.
> **6.2 Segmentation** finds the pancreas, which is roughly one percent of an abdominal CT.
> **6.3 Classification** crops that region and decides tumour present or absent.
> **6.4 Calibrated uncertainty** turns the classifier's number into a confidence that actually means something.
> **6.5 Verified explanation** shows where the model looked, and tests whether that is really why it decided.
> **6.6 Selective prediction** either answers, or refuses and escalates the case to a specialist.
>
> Note the colours. The blue modules are what any pancreatic detection system builds. The red ones — 6.4, 6.5, 6.6 — are ours, and they are the reason this is a safety system rather than just a classifier. Each module hands the next a defined output, so any one of them can be replaced without touching the others."

---

## 2:15 – 3:15 · S1 — Slide 3, module 6.1

> "6.1, preprocessing. On the left is one case after processing: the CT slice, then the same slice with the pancreas and tumour outlined. On the right is the tumour size distribution across the dataset, with the 2 cm line marked.
>
> Three things this module does. It resamples every volume to a fixed voxel size and windows the intensity to the soft-tissue range. It splits the data **patient-wise** — if two scans of the same patient landed on opposite sides of the split, we would be testing on a patient the model has already seen, and the score would be meaningless. And it sets aside a third group of 42 cases for calibration, which module 6.4 needs.
>
> This module is **complete and has run on all 281 real cases**. 110 of them carry a sub-2 cm tumour — that is the bucket we report separately throughout."

---

## 3:15 – 4:15 · S1 — Slide 4, module 6.2

> "6.2, segmentation. The pancreas is about one percent of an abdominal CT, so asking a classifier to search the whole volume wastes it on liver, bowel and spine. This module isolates the organ first.
>
> We use an **Attention U-Net**, and the reason is specific: its attention gates suppress regions the decoder is not interested in before the skip connections merge. That is precisely the problem when the target is one percent of the image. 5.9 million parameters, sized to fit the 8 GB GPU we have access to.
>
> The figure is the training loop running — loss falling, Dice tracked per class. It is a short verification run, not the full training. Its purpose is to prove the loop, the metric and the crash-resume path all work. Every epoch writes a checkpoint and the run resumes exactly where it stopped, because overnight lab access is not guaranteed.
>
> The code and its tests are complete. The full training run on real data is waiting on GPU time."

---

## 4:15 – 5:15 · S1 — Slide 5, module 6.3

> "6.3, classification. The mask from 6.2 defines a box around the pancreas; we cut that box out and hand it to a 3D convolutional network, which sees only the organ.
>
> It is deliberately **3D, not slice-by-slice** — a small tumour is recognised by its shape across slices, which a 2D model cannot see. And we report AUROC overall **and** for the sub-2 cm bucket alone, because an overall number is dominated by large, obvious tumours.
>
> One design decision worth mentioning: the crop accepts **any** mask. So we can develop and validate this module against the radiologist's own outline while 6.2 is still training. The difference between those two numbers is itself a result — it separates classifier error from segmentation error."

**Handoff line (S1):** *"Those three modules are the detection pipeline. My colleague will take the three that make this different."*

---

## 5:15 – 6:55 · S2 — Slide 6, module 6.4

> "6.4, calibrated uncertainty. Every classifier outputs a number between 0 and 1, and everyone reads it as a confidence. It usually is not one. A network that says 0.9 is right 90% of the time only if somebody checked — and nobody usually does.
>
> Two mechanisms. **MC Dropout** runs the network twenty times with dropout left switched on; if the twenty answers disagree, the model is uncertain about *this particular scan*. **Split conformal prediction** returns a set of labels instead of one, with a mathematical guarantee that the true answer is inside that set at least 90% of the time.
>
> The important property of that guarantee: it assumes **nothing about our model**. It holds even if the network is bad. A bad model simply produces larger, more honest sets.
>
> On the left is the reliability diagram — the dashed line is perfect calibration, and the gap between the bars and that line is the error. On the right, conformal coverage tracking the target.
>
> And here is the finding I would ask you to note. In this run our classifier **ranked every single case correctly** — and was still badly calibrated. It would have told a clinician it was confident when it was not. Nothing in modules 6.1 to 6.3 can detect that. Module 6.4 detected it automatically and escalated 40% of cases instead of answering them."

---

## 6:55 – 8:25 · S2 — Slide 7, module 6.5

> "6.5, verified explanation. A saliency map is a claim: *these voxels are why the model decided what it decided.* Almost every paper in this area shows the map and stops. An untested map is worse than no map, because a clinician may believe it.
>
> On the left is our Grad-CAM++ output — where the network looked, with the true tumour outlined in white for comparison.
>
> On the right is the test. We hide the voxels the map calls important and measure how far the prediction falls. Then we hide the **same number** of voxels chosen at random and measure again. Hiding anything moves the score, so the only meaningful question is whether hiding what the map pointed at moves it more than hiding noise.
>
> Look at the lines. Red is masking the highlighted region; purple is masking at random. **Red is below purple** — hiding the map's own region mattered *less* than hiding random tissue. So the map is not explaining the prediction, the system marks it FAIL, and it is withheld from the user.
>
> We tested this check in both directions. On a model built to depend on one known region of the image, the same test passes. So it is a working check, not one that always says no."

---

## 8:25 – 9:40 · S2 — Slide 8, module 6.6

> "6.6, selective prediction — the module the whole project is built around.
>
> Three independent signals feed one rule. Does the conformal set hold both labels? Do the twenty dropout passes disagree? Did the explanation fail its test? If any of them reports doubt, the system does not guess.
>
> That gives three outcomes, and the third is the one an ordinary classifier does not have: **flagged for specialist review**.
>
> These are screenshots of the working interface. The panel states, on every single result, what the model weights are, which modules are standing in, and what the data is — the system is built so that it cannot overstate itself.
>
> And the escalation rate is a dial the department sets. Demand higher coverage and more cases are flagged; relax it and fewer are. The trade-off between workload and missed tumours becomes an explicit, measurable choice instead of a hidden one."

**Handoff line (S2):** *"Back to my colleague for how all of this is verified."*

---

## 9:40 – 10:40 · S1 — Slide 9, test cases

> "Testing. We have **186 automated test cases**, all passing, broken down here by module.
>
> A few examples of what they actually check: that no patient appears in two splits; that training resumes mid-run without restarting; that an oversized organ is padded rather than silently truncated; that the conformal coverage guarantee holds; that a faithful saliency map passes and a misleading one fails; and that a missing signal is treated as *no evidence* rather than as a failure.
>
> One point about this table: the report is **generated by running the suite**, not written by hand. It cannot claim a test passes unless it did. We can regenerate it in front of you.
>
> Three real defects were caught this way and fixed — MC Dropout permanently altering the model, an entropy calculation summed over the wrong axis in 3D, and the saved checkpoint not being the model that actually trained. Each was silent. Each would have corrupted our results on the real dataset without anyone noticing."

---

## 10:40 – 13:40 · S2 — Slide 10, live demo

**Switch to the browser tab. The app is already running.**

> "Finally, the system itself. This is the stage 6.6 interface, running locally."

**Step 1 — point at the top panel.**
> "At the top it states what it is running on: the checkpoint, the calibration value, the number of dropout passes."

**Step 2 — click Analyse on the pre-selected case. ~20 seconds.**
> "I pick a case and run it. It preprocesses the volume, segments it, scores it, runs the twenty dropout passes, computes the conformal set, generates the saliency map and tests it."

**Step 3 — read the verdict aloud, including the reason.**
> "The verdict here is **flagged for specialist review**, and it tells us why: the conformal set holds both labels at the target coverage. It is not refusing to work — it is refusing to guess."

**Step 4 — the key interaction. Drag the alpha slider from 0.10 to 0.30, click Analyse again.**
> "Now I relax the required coverage, and run the same case again. The set collapses to a single label and the system answers instead of escalating. That one slider is the workload trade-off I mentioned, and it is the department's decision, not ours."

**Step 5 — scroll to the saliency panel.**
> "Below is the Grad-CAM++ map and its fidelity verdict, and at the bottom the provenance panel — which states on every result what is real and what is standing in."

> *If anything fails:* "I'll show the captured output instead" — go to slide 8 and talk over the screenshots. **Do not debug on stage.**

---

## 13:40 – 14:10 · S1 — Close

> "To summarise. All six modules are implemented and run end to end, with 186 tests passing. Module 6.1 is complete on all 281 real cases. Modules 6.2 through 6.6 are verified as working code on our test harness, and the remaining work is the training run on the real dataset, which is waiting on GPU access rather than on unfinished code.
>
> What we would ask for is confirmed time on the lab machine — even in short sessions, since every loop checkpoints and resumes. That is the single thing that would most accelerate the rest of this project.
>
> Thank you. We're happy to take questions."

---

## Questions you should expect

**"Is this real patient data? What is that red text on the figures?"**
> "The red stamp marks that the input was our synthetic test harness, not patient scans. It is printed automatically by our code so that a test figure can never be mistaken for a clinical result. Module 6.1 has run on all 281 real cases; the later modules are verified as working software, and we have been careful not to present that as clinical evidence."

Say this plainly if asked. Do not claim the figures are real results — the stamp is in the image, the repository is public, and the logbook records it.

**"Why haven't you trained it yet?"**
> "GPU access. The lab machine has not been confirmed for us yet. The code is complete and checkpointed so it can run in short sessions, and we can also produce real numbers for modules 6.3 and 6.4 without the segmentation model, because the ROI crop accepts the radiologist's own outline."

**"What is actually novel here?"**
> "Three things, and they are modules 6.4 to 6.6: a confidence with a distribution-free coverage guarantee, an explanation that is tested before it is shown, and the ability to abstain. Published detection work typically gives a probability nobody has checked and a heat map nobody has tested."

**"How is it better than existing systems?"**
> Slide 2's colour split, then: "An ordinary system outputs one probability and answers every scan. Ours outputs a probability, a prediction set and a verdict — and it can refuse."

**"What accuracy does it achieve?"**
> "We do not have accuracy on real data yet, and we would rather say that than quote a number from our test harness. What we can state is the sub-2 cm split we will report it in, and the calibration and abstention metrics alongside it."

**"Can this replace a radiologist?"**
> "No, and it is not designed to. It is a second reader. Its value is in directing attention — small pancreatic tumours are often visible in hindsight but missed on a scan ordered for something else."

**"Why 2 cm?"**
> "Below roughly 2 cm a tumour is far more likely to still be resectable, so that is the threshold where detection changes what happens to the patient. 110 of our 281 cases fall under it."

**"Why a CNN and not a Transformer?"**
> "The CNN pipeline is what the approved Phase-I report specifies, and with 281 training cases a Transformer would be the wrong tool — it needs far more data than we have."

**"What is left?"**
> "No modules. The training run on real data, the external validation check on the second dataset, and the report."

---

## The one rule

Every figure in this deck is the genuine output of our code. Say what it is. If a figure came from the test harness, it carries a stamp saying so — read it out rather than hoping nobody notices. The honest version of this presentation is strong: at the 60% stage, being able to state exactly what works and what does not is a better position than a claim that falls apart under one follow-up question.
