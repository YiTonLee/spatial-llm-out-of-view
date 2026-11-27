# E6a gates: helper-size ladder (inference only), written 2026-11-27 BEFORE any E6a code or run

> Repository copy. The plan file named here is kept outside git (`doc/`). Nothing below has been run, and no E6a code
> exists yet. Code (to write after the laptop restart): `src/e6a.py`, stage `e6a`, outputs in `workspace/e6a/`.

Plan: `doc/2026_11_25/PLAN_evidence_pipeline_2026-11-25.md`, §4 E6a, and "Revisions proposed 2026-11-26 (evening)",
items 2 and 4 (validation slices only; one dataset per domain).

## The question

**Plain.** A small model (the student, 3B) answers questions about pictures and videos. A bigger model (the helper)
can look first and write the student a short note. Does a bigger note-writer help, and is "small model + big
note-writer, called only when the small model is unsure" ever better value than just asking the big model directly?
"Better value" means: more questions right for the same time, or the same number right in less time.

**Technical.** Frozen student Qwen2.5-VL-3B-Instruct; helpers of three sizes write an F4-style packet (plain
statements, E1's best format); the student answers with the packet. Each helper-plus-student use is placed on the
accuracy-cost plane against the student alone and every helper answering alone.

## Models (downloads: plan §4 E6a, "Model downloads")

| rung | model | precision | on disk |
|---|---|---|---|
| student, and helper "self" | Qwen/Qwen2.5-VL-3B-Instruct | bf16 | already in the Hugging Face cache (7.1 GB) |
| helper, dense | Qwen/Qwen2.5-VL-7B-Instruct | bf16 | already in the cache (16 GB) |
| helper, sparse (about 3B active of 30B) | Intel/Qwen3-VL-30B-A3B-Instruct-int4-AutoRound | int4 (group 128) | to download, 17.9 GB, plus `auto-round >= 0.9.6` |
| fallback for the sparse helper (only if S1 below fails) | Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF, Q4_K_M + mmproj F16 | 4-bit | 19.7 GB, run with llama.cpp |
| tools-only composer (domain 2 only) | E1 domain-2 tools (Grounding DINO tiny + Depth Anything V2 Small), F4 | fixed code | already on disk |

All three VLMs are Apache-2.0. Confound stated up front: the 30B helper is 4-bit, the others bf16, so "size" also
changes precision. Reported, not corrected.

## Data: one dataset per domain, validation or held-back slices only (test sets stay sealed)

Each domain: a **calibration slice of 100 items** (format check and the router threshold only, never accuracy tuning)
and a **scoring slice of 300 items**. Items drawn with seed 6, before any model runs; the lists are saved first
(`workspace/e6a/items_<domain>.json`).

| domain | dataset and slice | why this slice | answer type, scoring |
|---|---|---|---|
| 1 out-of-view (real video) | Perception Test **train**, questions tagged object permanence, occlusion, containment or change detection; held-back slice (never trained on: E3 later trains on the rest of train, minus these items) | `valid` is domain 1's final test; synthetic eval16 stays sealed | multiple choice (3 options), exact option |
| 2 spatial (photos) | VSR **dev**, measurable relations (E1 domain-2 list) | VSR test was used in E1 domain 2; dev was used only for the C0 tool check, no tuning | true/false |
| 3 charts | ChartQA **val**, human questions | ChartQA test is the final test | ChartQA relaxed accuracy (numbers within 5%, text exact, case-insensitive) |
| 4 small details | Visual CoT `gqa_cot_train`, items whose answer box covers < 5% of the image, COCO-overlap images removed; held-back slice | V\*Bench and HR-Bench are the sealed held-out tests | short answer, exact match after normalisation (lower case, articles and punctuation removed) |

Video input (domain 1): 16 frames, evenly spaced over the cut video (the cut-frame files applied), max 448 px on the
long side, the same for every model. If S1 fails on video, 8 frames for every model.

## The note each helper writes (prompts frozen before the scoring slice; written in `src/e6a.py`)

The helper sees the same picture or video and question as the student, and is asked for at most 40 words of plain
statements in the domain's F4 template, ending with its answer:
- domain 1: where the object was last seen, what moved, where it is now ("ball last seen under the left cup; the left
  cup moved to the right; the ball is under the right cup");
- domain 2: the relation as measured ("the cat is below the table; the cat is nearer the camera than the table");
- domain 3: the values read and the calculation ("2019: 41%; 2020: 45%; 2020 − 2019 = 4 points");
- domain 4: what the small region shows ("the sign above the door reads 'EXIT'; it is red").

## Arms (per domain, per helper H in {3B self, 7B, 30B-A3B})

1. student alone (S);
2. H alone (H answers the question directly, no note);
3. S + H's note, always (the helper is called on every question);
4. S + H's note, routed: the helper is called only when the student's first-pass confidence is below τ (fixed rule;
   E4's learned router replaces it later);
5. domain 1 only: S + one note per video, shared by all of that video's questions in the slice (cost divided);
6. domain 2 only: S + tools-only F4 note (E1 domain 2), and K2 (the rule alone), as reference rows;
7. domain 3 only: S + ChartQA's data table (oracle, a reference row, never a candidate for the frontier).

**Student confidence.** Probability of the chosen option (true/false and multiple choice: softmax over the option
tokens); open answers (domains 3, 4): geometric mean of the greedy answer's token probabilities.
**τ** per domain and helper: on the calibration slice, the τ with the highest accuracy among those calling the helper
on ≤ 50% of items; frozen before scoring.

## Cost (laptop, RTX 5090 Laptop 24 GB, batch 1, greedy, `agent/cost.py` timing)

- Time per question = student first pass (routed arm only) + helper reading and writing (if called) + student final
  answer; mean over the scoring slice; includes picture/video encoding.
- Memory: peak GPU memory of each model alone, and the sum (the student and the 30B helper do not fit together on
  24 GB, so the helper's notes are written first and cached; times are measured per model and added).
- Reported next to it: input tokens and note length in tokens.

## Checks before scoring (on the calibration slice; stop rules)

- **S0 format:** ≥ 95% of each helper's notes fit the template (≤ 40 words, ends with an answer). If not, the prompt
  is fixed once on the calibration slice and rechecked; still < 95%: that helper is dropped in that domain and
  reported.
- **S1 memory:** the 30B-A3B int4 model must load and answer one item per domain on the 24 GB GPU without CPU
  offload. If it does not: stop, report, and ask you whether to use the GGUF fallback (llama.cpp; its times then come
  from a different runtime and are marked so) or Colab A100.
- **S2 time:** the first 20 calibration items give a projected total. If it is more than 1.5 × the estimate below,
  stop and ask you before scoring.

## Gates

- **G6a (pass rule, the plan's).** A helper-plus-student use U (always or routed, any helper) is **on the frontier**
  in a domain if it beats, with a paired 95% lower bound > 0, every baseline in {S alone, each H alone} whose time
  per question is no larger than U's. **Pass: this holds in at least 2 of the 4 domains.**
  Prior support: Online ICD (a large teacher called on a few percent of queries lifts a small VLM);
  Self-RAG-style self help (the 3B "self" rung); sparse models answer at about their active size (Qwen3-VL-30B-A3B).
- **G6a-R (prediction from E1 domain 2, reported, not a gate).** S + H always − S alone is positive only where H alone
  is more accurate than S alone; checked in each domain × helper cell (12 cells), reported as how many agree. On VSR:
  the share of H's notes whose relation statement agrees with the label, against the about 86% break-even from E1
  domain 2.
- **G6a-C (reported).** Accuracy and time of every arm; the call rate of the routed arm; for domain 1, shared note vs
  one note per question.
- **Statistics.** Paired bootstrap over items (domain 1: over videos), 10,000 resamples, 95% intervals. With 300
  items, an interval on a difference is roughly ±4–5 points: the pilot can show large gaps only.

If G6a misses everywhere because each helper alone dominates, that is the clean negative result in the plan ("big
helps small" fails at this scale), and E6b/E6c are reconsidered before anything is built.

## Compute (needs your go: more than 30 minutes)

Per domain 400 items. Estimates, to be replaced by the S2 measurement: student arms about 10 min; 7B notes and
answers about 30–45 min; 30B-A3B int4 about 45–90 min (the speed of int4 mixture-of-experts layers in Transformers on
this GPU is not yet measured: UNVERIFIED). Domain 1 (video) about 2× the others.
**Total about 5–9 h**, run one domain at a time (each a separate step, about 1–2.5 h), domain 2 first.

## Additions written 2026-11-27 while coding, before any scoring-slice item was run

Made while writing `src/e6a.py` and smoke-testing it on 2–3 items per domain (outputs tagged `_n3`, never scored):
- **Picture size per domain**, the same for every model: VSR at most 448 × 448 pixels (as in E1 domain 2, so the
  student-alone row is comparable); ChartQA and GQA at most 1024 × 28 × 28 ≈ 0.8 MP, because reading chart values and
  small details needs resolution; video 16 frames, long side 448.
- **Note form.** The 3B answered "Yes." instead of writing a note in 2 of 3 smoke items, so the prompt now asks for two
  lines, `Facts: <at most 30 words>` and `Answer: <answer>`; S0 also requires the note to start with "Facts:".
  The student reads it after "Note from a helper:". The tools note (domain 2) and the chart table (domain 3) are given
  exactly as E1 rendered them.
- **Shared note per video (domain 1):** a question-free description, at most 60 words.
- **Helper note accuracy** (G6a-R) is read from the note's `Answer:` line.
- **Domain 4:** one question per image (so the bootstrap groups are items). The COCO-overlap filter is dropped for
  E6a: nothing is trained in E6a, so overlap cannot leak; it stays for E3, which trains.
- **Timing:** one discarded warm-up call per model before the first timed item.
- Shared-note cost is divided by the number of that video's questions in the same slice.

## S1/S2 result for the 30B-A3B int4 helper (measured 2026-11-27, before any scoring)

- **S1 memory: pass.** Loads fully on the GPU, 16.7 GB, no CPU offload.
- **S2 time: fail.** 8 new tokens take 18–20 s (about 2.2 s per token). auto-round 0.17 selected its pure-PyTorch
  backend (`auto_round:torch_zp`; it suggests `gptqmodel` for a faster one) and runs the 128 experts per layer in a
  Python loop (`linear_loop`: "Grouped MoE not used ... unsupported projection type QuantLinear"). A 40-word note
  would take about 2 minutes, about 13 h per domain. Per the stop rule, the 30B rung waits for your decision.

## Path change 2026-11-27 (your decision, before any scoring): helpers are the 3B and the 7B

The 30B-A3B rung is deferred to future work (S2 fail above; the fast GGUF route cannot be fine-tuned for E6b). G6a is
unchanged in form; its baselines are now {S alone, 7B alone}. G6a-R covers 4 domains × the 7B (the 3B "self" helper
alone is the student alone).

## S0, first calibration pass (2026-11-27), and the one allowed prompt fix

Notes in the template (calibration slice, 100 items per domain): 3B 0 / 45 / 5 / 26% (d1 / d2 / d3 / d4), 7B 95 / 97 /
99 / 98%. The 3B mostly answered the question ("Yes.", "0.96%", "cell phone") instead of writing a note. The one fix:
every helper's reply is forced to begin with "Facts:" (prefilled), the same for both helpers; calibration rerun for
both. First-pass files kept as `*_cal_v1.json`. If the 3B still fails S0 in a domain, its "self" rung is dropped there
and reported.

**S0 after the fix (calibration rerun):** 3B 94 / 100 / 96 / 100%, 7B 95 / 97 / 99 / 100% (d1 / d2 / d3 / d4). The 3B
"self" helper fails S0 on d1 (94% < 95%) and is dropped there, as the rule says. S2 projection: about 1 h for all
four domains (3B and 7B, 400 items each); full run started 2026-11-27 on your go.


## Result (full run, 2026-11-27)

**G6a: MISS** (frontier in 0 of 4 domains). Full table `pilot/e6a/E6A_table.md`; write-up `pilot/e6a/E6A_RESULTS.md`.
- d1 video: S+7B always 47.0 vs S 42.7 (+4.3 [-1.3, 9.7]) and vs 7B alone 41.0 (+6.0 [0.0, 11.9]): just short.
  Secondary (not a pre-written G6a use): shared note per video 49.0, +6.3 [0.0, 12.7] vs S, +8.0 [1.9, 14.2] vs 7B alone.
- d2 / d3 / d4: 7B alone (90.0 / 78.3 / 43.0) beats the student (87.7 / 73.3 / 40.7) and every note use, at lower time.
- G6a-R holds only on d3. S1 pass (peak 15.9 GB). S2 pass (about 65 min total).
