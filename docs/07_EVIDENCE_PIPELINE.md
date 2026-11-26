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
| `agent/packets.py` | Packet formats F0-F4 rendered from the same tool result |
| `agent/cost.py` | CUDA-synchronised stage timer, total cost per condition, accuracy-cost frontier |
| `agent/data/` | Public datasets: `manifest.py` (sources, licences, groups), `checksums.json`, `download.py` |
| `src/e0_ledger.py` | E0: time every step of answering one question; frontier and gates |
| `src/e1_d1.py` | E1 domain 1: five note formats, untrained and trained, correct / corrupted / shuffled / negated notes |
| `docs/gates/` | Pass rules, each written before its experiment ran |
| `tests/test_agent.py` | Fast CPU checks of the above |

Run (after `python run.py data cache model_w estimator`, which E0 and E1 need):

```bash
python run.py e0        # about 3 min on one 24 GB GPU
python run.py e1_d1     # about 4 h 15 min (untrained 23 arms 1 h 44 min; four LoRAs with their tests)
python run.py e0 e1_d1 --quick --work workspace_quick   # smoke test, meaningless numbers
```

Outputs go to `workspace/e0/` and `workspace/e1/` (`E1_D1_table.md` is the full table).

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

### Next: E1 domain 2, real photos (VSR)

The pass rules were written before the run: `docs/gates/e1_d2_gates.md`.
- The tools are Grounding DINO tiny and Depth Anything V2 Small.
- The note comes in seven formats, including a depth-map picture and a crop.
- The pass rules use only the relations the tools can measure: left-right, up-down, depth and containment.
