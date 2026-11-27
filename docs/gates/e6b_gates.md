# E6b pass rules: one LoRA on the 7B helper, video first (written 2026-11-27, before any E6b code or run)

## Why video first

E6a (`docs/gates/e6a_gates.md`, result section) missed G6a in all four domains, but on video (d1, Perception Test)
the untrained 7B's note helped:
- the 3B student went from 42.7% to 47.0% (+4.3 [−1.3, 9.7]), with 39 answers fixed and 26 broken;
- the 7B answering alone scores 41.0%, below the student.

So the facts the 7B writes down carry something the student lacks. On still pictures the note broke more answers
than it fixed, and the 7B alone was better and faster. E6b therefore trains the helper on video only. The other
domains wait for this result.

**Plain.** The 7B's notes helped on video even though nobody taught it what this student needs. E6b teaches it.
The 7B writes several notes per training question, the student tries each, and the 7B is trained only on the
notes that made the student right. If that works, the student should be right more often than with the untrained
note, and more often than the 7B answering on its own.

## Data (no overlap with the E6a test)

- **Training pool:** the Perception Test *valid* split, with the same four tags as E6a (object permanence, occlusion,
  containment, change detection): 1,720 questions on 1,130 videos, all on disk.
- **Dev set:** 200 of those questions, picked by video (all of a video's questions go to the same side), seed 6. It
  is used only for the stop rules below.
- **Training set:** the other questions, about 1,520.
- **Test set:** the 400 E6a d1 items (100 calibration + 300 scoring) from the Perception Test *train* split. None of
  them is used for training or note selection. The test is "always call the helper", so the calibration threshold
  is not needed and all 400 count; the 300-item scoring slice is also reported, for direct comparison with E6a.
- **Leakage check (asserted in code):** the video ids of training + dev and test are disjoint.
- **Video input:** same as E6a (16 frames, long side 448, cut-frame mapping applied).

## Method (fixed now)

1. **Sample.** For each training question, the untrained 7B writes K = 4 notes with the E6a d1 prompt and the "Facts:"
   prefill. Sampling: temperature 0.8, top-p 0.95, at most 80 new tokens, seed 6.
2. **Score with the frozen student.** p0 = the student's probability of the correct option with no note (first-token
   softmax over A/B/C, as in E6a); pk = the same with note k.
3. **Select (the student's signal).** Keep the note with the highest pk if (a) the student's top option with it is
   correct, (b) pk ≥ p0 + 0.10, and (c) it passes the E6a format check (`fmt_ok`). Questions with no such note
   contribute nothing.
4. **Train LoRA_sel on the 7B.** Rank 16, alpha 32, dropout 0.05, on q/k/v/o of the language model (vision tower
   frozen). lr 1e-4, cosine decay, 1 epoch, batch 1 with 8-step accumulation, gradient checkpointing, bf16. The loss
   covers only the note tokens after the "Facts:" prefill.
5. **Control LoRA_rand.** Same number of examples from the same questions, but the note is one of the K samples
   chosen at random (format check only, no student signal). Same hyper-parameters.
6. **Test.** On the 400 test items, the 7B writes greedily, with the E6a prompt, prefill and cap (80 tokens). The
   student answers as in E6a.

## Arms on the test set

| arm | role |
|---|---|
| S (3B alone) | baseline (E6a numbers, re-run once to confirm they repeat) |
| 7B alone (untrained) | baseline |
| S + untrained 7B note | E6a's best d1 use |
| **S + LoRA_sel note** | the method |
| S + LoRA_rand note | control: training on the helper's own notes without the student's signal |
| 7B + LoRA_sel answering alone | does the adapter just make the 7B a better answerer? |
| S + LoRA_sel note, Answer line removed | do the facts help, or only the stated answer? |
| LoRA_sel note's own Answer line | accuracy of the helper's stated answer |

Time per question is measured as in E6a: CUDA-synchronised, note time plus student time.

## Pass rules

- **G6b-v (main).** S + LoRA_sel note is on the frontier on video. It beats S alone and the 7B alone (both are
  faster, so both count) with paired group-bootstrap 95% lower bounds > 0: 10,000 resamples over videos, on the 400
  test items.
  - This is the E6a rule applied to the trained helper. E6a's untrained note missed it with lower bounds of −1.3 and
    0.0 (300 items).
- **G6b-sel (mechanism).** S + LoRA_sel − S + LoRA_rand has a lower bound > 0. The gain must come from selecting by
  the student's success, not from training on video notes in general.
- **Reported, not gates:**
  - gain over the untrained note;
  - answers fixed / broken (E6a: 39 / 26 on 300);
  - the Answer-line-removed arm;
  - 7B + LoRA_sel alone;
  - note length and time;
  - accuracy per tag.
- **Reading the outcome:**
  - G6b-v pass, G6b-sel pass: the helper learned what this student lacks. Go to the one-LoRA-per-task plan on the
    next domain (charts), and E6c's token budget on video.
  - G6b-v pass, G6b-sel miss: training on video notes helps, but the student's signal adds nothing. Report it that
    way.
  - G6b-v miss: report a negative result. If 7B + LoRA_sel alone rose, the adapter taught answering, not
    note-writing.

## Stop rules

- **T0 (enough signal).** At least 300 training notes survive selection. Otherwise stop: the student's signal is too
  rare to train on. Report the count.
- **T1 (format).** LoRA_sel notes on the 200 dev questions pass `fmt_ok` ≥ 95%. One fix is allowed (half the
  learning rate) before stopping.
- **T2 (memory).** Training fits the 24 GB laptop GPU (7B bf16 about 15.6 GB, plus LoRA, plus about 1,150 video
  tokens). Otherwise stop and ask before moving to a Colab A100.
- **T3 (time).**
  - Projected total ≤ 1.5× the estimate below.
  - Median LoRA_sel note length ≤ 1.25× the untrained median (34 tokens on d1). The frontier test charges time
    anyway, but a longer note would change what is being compared. If it is exceeded, report it; it is not a stop.

## Estimate (laptop, RTX 5090 24 GB; needs your go, over 30 min)

| step | estimate |
|---|---|
| frames for 1,130 videos | about 15 min (CPU) |
| 4 sampled notes × 1,720 questions (one generate call with 4 returns per question) | about 60–80 min |
| student scoring of 5 versions per question (no note + 4 notes) | about 45 min |
| LoRA_sel and LoRA_rand training (about 300–1,000 examples each) | about 30–60 min each |
| test: 400 items × 2 LoRAs plus the extra arms | about 40 min |
| **total** | **about 4–5 h**, in steps that each save their output |

## Prior work (why this is likely to work)

- **Keep the generated reasoning that led to a right answer, then train on it:** STaR (Zelikman et al.,
  2203.14465); rejection-sampling fine-tuning (Yuan et al., 2308.01825); ReST (Gulcehre et al., 2308.08998).
  E6b changes one thing: the signal is the *student's* correctness, not the writer's own.
- **A stronger model's output is not automatically the best for a weaker one;** what suits the student matters:
  "Small Models Struggle to Learn from Strong Reasoners" (Li et al., 2502.12143; ID to re-check before the slide).
- **E1 and E6a here:** plain statements are followed by the student (E1); on video, the untrained note already helps
  (E6a). E6b tries to raise how often the note helps (39) and lower how often it misleads (26).
