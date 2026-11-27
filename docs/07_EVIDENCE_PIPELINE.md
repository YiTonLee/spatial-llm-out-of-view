# 07. The evidence pipeline: what to hand a small model, in what form, at what cost

Branch `evidence-pipeline`. Started 2026-11-25. This guide grows with each experiment.

## The question

**Plain.** A small model answers questions about pictures and videos. Helper programs ("tools") can measure things for
it: where objects are, how far away. We want to know what to hand the model, written how, and when the extra help is
worth its time. Picture a student in an exam with a phone to call experts: which expert to call, how the expert should
phrase the answer, and when to hang up.

**Technical.** For a fixed student (Qwen2.5-VL-3B-Instruct, optionally with a LoRA), compare evidence *packets*,
meaning a tool's output rewritten as short text or a small image, holding the information fixed and varying the format, the
student's training and the packet's quality. Cost always includes the vision encoder and the tools, not only the
prompt length.

## Code

| Path | What it is |
|---|---|
| `agent/packets.py` | Packet formats F0-F4 for out-of-view balls (domain 1), rendered from the same tracker result |
| `agent/photo_packets.py` | Packet formats F0-F4 for photos (domain 2), rendered from the same detector + depth result; the rule alone (K2) |
| `agent/cost.py` | CUDA-synchronised stage timer, total cost per condition, accuracy-cost frontier |
| `agent/data/` | Public datasets: `manifest.py` (sources, licences, groups), `checksums.json`, `download.py` |
| `src/e0_ledger.py` | E0: time every step of answering one question; frontier and gates |
| `src/e1_d1.py` | E1 domain 1: five note formats, untrained and trained, correct / corrupted / shuffled / negated notes |
| `src/e1_d2.py` | E1 domain 2 (VSR photos): tools, stop rule, seven note formats, untrained and trained, report |
| `src/e1_d2_extra.py` | E1 domain 2, analysis written after the run: tool-right vs tool-wrong split, note-only gaps |
| `src/e6a.py` | E6a: helper notes (3B self, 7B) for the 3B student on four domains; calibration threshold, frontier, G6a |
| `docs/gates/` | Pass rules, each written before its experiment ran |
| `tests/test_agent.py` | Fast CPU checks of the above |

Run (E0 and E1 domain 1 need `python run.py data cache model_w estimator` first; E1 domain 2 needs
`python -m agent.data.download --only e1_d2`, about 21 GB):

```bash
python run.py e0        # about 3 min on one 24 GB GPU
python run.py e1_d1     # about 4 h 15 min (untrained 23 arms 1 h 44 min; four LoRAs with their tests)
python run.py e1_d2     # about 2 h 35 min (tools 50 min; untrained 15 arms 37 min; four LoRAs with their tests)
python run.py e6a       # about 70 min (needs python -m agent.data.download --only e6a; d2's tools reference arm needs e1_d2)
python run.py e0 e1_d1 --quick --work workspace_quick   # smoke test, meaningless numbers
```

Outputs go to `workspace/e0/`, `workspace/e1/`, `workspace/e1d2/` and `workspace/e6a/` (`E1_D1_table.md`, `E1_D2_table.md` are the
full tables).

## Public datasets

**Plain.** The datasets are not in this repository. Together they are about 170 GB, GitHub refuses files over 100 MB,
and several licences forbid passing copies on. Instead, the repository has the list of where each one comes from,
and one command fetches them from the original sources. A fingerprint check confirms that your copy is identical to
ours.

**Technical.** `python -m agent.data.download` reads `agent/data/manifest.py` and works as follows:
- It downloads into `data/public/<dataset>/`, which is ignored by git.
- Plain links resume from where they stopped (HTTP Range requests).
- Hugging Face items use `snapshot_download`, pinned to the exact revision we used; no login is needed.
- Archives stay zipped.
- `--check` compares file sizes with `checksums.json`; adding `--verify` also compares sha256.

```bash
python -m agent.data.download --list                 # every dataset: size, licence, which experiment, present or not
python -m agent.data.download --only e1_d2           # only what one experiment needs (groups in manifest.GROUPS)
python -m agent.data.download                        # everything (about 170 GB; checks free disk space first)
python -m agent.data.download --check --verify       # confirm your copy matches ours, download nothing
# Colab: write into Drive so it survives the session
python -m agent.data.download --only e1_d2 --data /content/drive/MyDrive/spatial_llm/data/public
```

**Three datasets need a person.** The script prints these steps and checks the files once you have saved them:
- **DocVQA and InfographicVQA:** each user registers at rrc.cvc.uab.es and accepts the terms, then downloads the
  archives into `data/public/docvqa/` and `data/public/infographicvqa/`.
- **CUB (bird photos):** the Caltech server refuses scripts, so download `CUB_200_2011.tgz` in a browser into
  `data/public/cub/`.

**What is deliberately absent:**
- Visual CoT's 143 GB image copy. We take only its question files (0.17 GB) and read the images from GQA and CUB.
- EgoTracks: it needs an institutional Ego4D licence.
- SA-V: it is 957 GB.

**Sharing within the team.** Each person downloads for themselves. A shared Drive folder is fine only for datasets
whose licence allows sharing (see `--list`). Never share DocVQA, InfographicVQA or CUB.

## Results so far

### E0, cost ledger (2026-11-25; gates `docs/gates/e0_gates.md`)

Both-correct means both videos of a twin pair are answered right. Chance is 25%; a model that only looks at the last
frame scores 0%. Costs are on one RTX 5090 laptop GPU, bf16, 20 eval8 scenes.

| condition | both-correct % | input tokens | one-shot ms | ms if frames were already processed |
|---|---|---|---|---|
| V: video only, 8 frames | 5.5 | 624 | 188 | 100 |
| D: 8 frames + state note | 67.5 | 685 | 194 | 105 |
| R: last frame + state note | 72.9 | 223 | 165 | 76 |
| K: calculator, no language model | 74.5 | 0 | 90 | 0.2 |

**Plain.** The note lifts accuracy from 5.5% to 72.9% but saves only 12% of the time. Turning frames into picture
tokens takes 86 of the 188 ms, and the tools need that step too.

**Technical.** G1 passes: the frontier is {K}. G2 misses in the one-shot setting: K costs 48% of V and R costs 88%,
against the ≤ 25% the rule required.

### E1 domain 1, same facts in five formats (2026-11-25; gates `docs/gates/e1_gates.md`)

Notes are built from estimated tracks. The trained columns use one LoRA per format.

| format | untrained | trained | trained, true tracks | trained, wrong note | trained, no note |
|---|---|---|---|---|---|
| F1 raw sightings | 1.4 | 8.3 | 8.4 | 8.1 | 7.8 |
| F2 positions at the asked time | 2.3 | 68.6 | 83.0 | 47.7 | 3.9 |
| F3 derived (offset, distances) | 10.6 | 74.5 | 98.2 | 48.8 | 4.5 |
| F4 predicates (left of, closer) | 24.3 | 74.5 | 98.3 | 48.9 | 2.6 |

**Plain.** The model can use only notes that have already done the arithmetic. Once trained, it scores exactly what
the calculator scores on its own (74.5%), and it collapses when the note is missing.

**Technical.**
- G1 passes: untrained F4 − F1 = +22.9, 95% CI [19.1, 26.8].
- After training the gap grows rather than shrinks: +66.2, 95% CI [60.5, 71.2].
- Trained F3 − F2 = +5.9, 95% CI [2.8, 9.2].

### E1 domain 2, real photos (VSR) (2026-11-26; gates `docs/gates/e1_d2_gates.md`)

VSR test, 2,195 true/false statements about COCO photos; the pass rules use the 1,170 whose relation the tools can
measure (left-right, up-down, depth, containment). Tools: Grounding DINO tiny + Depth Anything V2 Small. Stop rule on
dev: tool agreement 69.2% (≥ 60%), so the run went ahead.

| | accuracy % (measurable) |
|---|---|
| photo alone (untrained) | 81.1 |
| photo + F1 raw / F3 derived / F4 statements (untrained) | 80.8 / 79.8 / 77.4 |
| note only: none / F1 / F3 / F4 (untrained) | 46.5 / 49.5 / 57.4 / 57.4 |
| one LoRA per format, photo + its note / photo, no note | 84.3–84.5 / 84.7–85.2 |
| the tools' rule alone (K2, no model) | 70.9 |

- **G1 MISS:** max(F3, F4) − F1 = −0.9 [−2.6, 0.7] (F4: −3.4).
- **G2 MISS:** best note + photo − photo alone = −0.3 [−2.1, 1.4] (F4: −3.8).
- Trained models ignore the note: with it, without it, or with another photo's note, within 1 point.

**Why (split after the run, `src/e1_d2_extra.py`; it uses the labels, so it explains, it is not a gate).** With F4 and
the photo, the untrained model gains +3.4 [1.0, 5.9] on the 830 items where the tool was right and loses 21.2 on the
340 where it was wrong. With the photo hidden, F4 − F1 = +7.9 [5.0, 10.9].

**Plain.** The tools were less accurate than the model's own look at the photo (71% vs 81%), so their note could only
hurt on balance. The wording effect from domain 1 is still there: the model follows plain statements, which helps when
they are right and hurts a lot when they are wrong. Following a note pays only above about 86% note accuracy.

**Technical.** Evidence accuracy (K2 70.9) is below the student's prior (81.1); format then controls how much of the
evidence's error is transmitted. Lesson for the later experiments: the value of a note is its reliability relative to
the student times its usability, and the router (E4) and a reliability field in the packet (E3) are needed.

### E6a, helper size (2026-11-27; gates `docs/gates/e6a_gates.md`)

The 3B student answers alone, or after a short note written by a helper: itself (3B) or Qwen2.5-VL-7B. The note is
"Facts: at most 30 words / Answer: ...". Four domains, one dataset each, 300 scored items; the "call the helper only
when unsure" threshold is set on 100 other items. The planned 30B-A3B helper was deferred: it wrote about 2.2 s per
token on the laptop.

| % right (s per question) | 3B alone | 7B alone | 3B + 7B note | 3B + 3B note |
|---|---|---|---|---|
| d1 video (Perception Test, 3-way) | 42.7 (0.32) | 41.0 (0.43) | **47.0** (1.64) | dropped (format 94% < 95%) |
| d2 photo (VSR, yes/no) | 87.7 (0.09) | **90.0** (0.12) | 85.3 (0.82) | 79.3 (0.77) |
| d3 chart (ChartQA, open) | 73.3 (0.25) | **78.3** (0.30) | 75.3 (1.46) | 70.7 (1.39) |
| d4 small detail (Visual CoT GQA, open) | 40.7 (0.13) | **43.0** (0.17) | 34.3 (0.76) | 35.0 (0.69) |

- **G6a MISS** (0 of 4 domains; needs 2). Closest, d1: +4.3 [-1.3, 9.7] vs the 3B alone, +6.0 [0.0, 11.9] vs the 7B
  alone. One shared note per video (secondary): 49.0, +6.3 [0.0, 12.7] and +8.0 [1.9, 14.2].
- Answers fixed / broken by the 7B note: d1 39 / 26, d2 20 / 27, d3 17 / 11, d4 22 / 41.
- Calling the helper only when the student is unsure (first-token confidence) never beat calling it always.
- Peak memory: 3B 7.3 GB, 7B 15.9 GB.

**Plain.** On photos and charts, asking the bigger model directly is better and faster than having it write a note;
the small model sometimes copies a wrong fact from the note. On video the note helps even though the bigger model on
its own is worse: writing the facts down from the frames is what helps.

**Technical.** The note's own `Answer:` line is about as accurate as the 7B alone, so the losses on d2 and d4 come
from the student over-weighting wrong facts, not from a worse helper. The prompted note is not optimised for the
student; E6b trains the helper (one LoRA) on notes that make the student right, starting with video.
