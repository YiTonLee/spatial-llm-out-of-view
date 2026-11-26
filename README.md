# Where did the ball go? Out-of-view tracking for a small video-language model

Harvard, Fall 2026.

**The question.** A small AI model that reads video and answers in words, Qwen2.5-VL-3B, is asked where a ball is after the ball has rolled out of the camera's view. It answers wrongly. Why, and what fixes it?

**The answer so far.**
- The model answers from the last frame and ignores how the ball moved before it left.
- We keep a **written record outside the model**. A calculator extends each ball's path and handles balls that stop, are moved or turn, then hands the model one sentence to read.
- We **measure each ball more precisely with a zoomed second look**, aimed at the frames that matter most.
- On test videos where the model alone scores **5%**, the system reaches **67%** with the record and **about 75%** with zoomed looks.

This repository rebuilds everything that holds up, from generating the videos to the final accuracy tables, and compares your numbers with ours.

---

## In plain words: the test

Every result is measured on **twin videos**:
- Each twin pair is two 8-frame videos whose **last frames are identical**.
- In one video, a ball rolled straight out of view. In the other, the same ball stopped, was picked up and moved, or turned before leaving.
- We ask the same question of both, for example "Is the red ball left or right of the blue ball, 2 seconds after the last frame?". The right answer differs between the twins.
- A question counts as correct only if **both** twins are answered correctly.
- Random guessing gets 25%. A model that only looks at the last frame gets 0%, because it must give the same answer twice.

| | Twin accuracy |
|---|---|
| Model alone, sees all 8 frames | 5% |
| + written record and calculator (A), distances (B), change-aware rule (C) | 67% |
| + a zoomed look at the last moment each ball was seen | ~73–75% |
| Same system, if every position were measured perfectly | 97% |

Full tables with 95% ranges: [docs/04_RESULTS.md](docs/04_RESULTS.md).

## Technically: the pipeline

```
frames ─► Qwen2.5-VL vision tower (frozen, cached) ─► 64 tokens/frame
                                                      ├─► measuring part (2.5M) ─► (visible?, x, z) per ball per frame
                                                      │        └─► zoomed look (96 px square → 64 or 256 tokens) ─► refined (x, z)
                                                      │            at each ball's last sighting
                                                      ▼
    last frame + question + sentence  ◄── calculator (change-aware segmented least squares) ◄── written record
                 │
                 ▼
     Qwen2.5-VL decoder (LoRA) ─► answer
```

Block-by-block description, in the order it was built: [docs/02_MODEL_STRUCTURE.md](docs/02_MODEL_STRUCTURE.md).

---

## Quick start

```bash
# 1. a CUDA build of PyTorch (https://pytorch.org/get-started/locally/), then
pip install -r requirements.txt

# 2. see the plan
python run.py list

# 3. rebuild everything (about 4-5 GPU hours on a 24 GB GPU + 1.5 CPU hours; ~50 GB disk; resumable)
python run.py all

# 4. compare with our numbers
python run.py report          # -> workspace/my_results.md
```

You can also open the notebooks in `notebooks/` in order (00 → 06). They run the same stages and explain each one.

**Data is not downloaded.** Every video is regenerated from fixed seeds, bit-identical to ours, and checked by fingerprint. Only the Qwen2.5-VL-3B weights (about 7.5 GB) come from Hugging Face on first use. See [docs/01_DATA.md](docs/01_DATA.md).

## Repository map

| Path | What it is |
|---|---|
| `run.py` | **Main running file**: runs the stages in order, skips finished steps, writes logs, prints the report |
| `pipeline/` | The orchestration: `config.py` (paths, splits), `stages.py` (every stage and its exact commands), `runner.py` (runs and logs one step), `report.py` (collects and compares numbers) |
| `src/` | The experiment code, the exact scripts that produced our numbers. See [src/README.md](src/README.md) for which file does what |
| `notebooks/` | One notebook per group of stages, with plain explanations and your-vs-our tables |
| `docs/` | The guides listed below |
| `reference/` | Our numbers (`expected_results.json`) and data fingerprints (`data_fingerprints.json`) |
| `tests/` | Fast CPU checks (`python -m pytest tests`) |
| `tools/` | Authors-only helpers (rebuilding the reference file) |
| `agent/` | Evidence pipeline (branch `evidence-pipeline`): packet formats, cost accounting, public-dataset downloader |
| `data/` | Created by `python -m agent.data.download`: public datasets. Not committed |
| `workspace/` | Created by `run.py`: videos, caches, weights, results, logs. Not committed |

## Guides (read in this order)

1. [docs/01_DATA.md](docs/01_DATA.md): the videos and every generated file, step by step
2. [docs/02_MODEL_STRUCTURE.md](docs/02_MODEL_STRUCTURE.md): the model, block by block, in the order it was built
3. [docs/03_RUNNING.md](docs/03_RUNNING.md): the main running file, options, times, Colab, debugging
4. [docs/04_RESULTS.md](docs/04_RESULTS.md): the three result tables and how to read them
5. [docs/05_STATUS_AND_NEXT_STEPS.md](docs/05_STATUS_AND_NEXT_STEPS.md): current stage, progress, next steps
6. [docs/06_HISTORY.md](docs/06_HISTORY.md): what we tried that did not hold
7. [docs/07_EVIDENCE_PIPELINE.md](docs/07_EVIDENCE_PIPELINE.md): the evidence pipeline (E0, E1), public datasets

---

## Current stage and next steps

**Done.**
- Fixes A, B and C.
- The zoomed look (Versions 3 and 3c).
- The rule "with few looks, zoom on each ball's last sighting", confirmed on three fresh sets of twins.

**In progress: an agent component.**
- Choosing where to look from the system's own estimate works when looks are scarce.
- Re-looking based on what was just seen has added about +1 so far.
- A perfect "simulate and check" step would add about +6.6, mostly by telling "standing still" from "still moving" after a change. Building a real version that beats rule C is the next test.

**From the professor's feedback (not yet researched):**
- **Improve the model itself.** Everything that works so far is a tracker *outside* the model: the measuring part, the record and the calculator. We also need to make the language model itself better at keeping track of objects, not only wrap it in tools.
- **Run on real data.** All results are on simulated videos. The current system must be run on real video, for example **Beyond3D**, to see whether it holds outside simulation.

Details: [docs/05_STATUS_AND_NEXT_STEPS.md](docs/05_STATUS_AND_NEXT_STEPS.md).

## Public datasets (evidence pipeline)

The experiments from E1 domain 2 on use public image, document and video datasets. **They are not in this repository**:
together they are about 170 GB, GitHub refuses files over 100 MB, and several licences forbid passing copies on.
Instead, one command fetches them from their original sources into `data/public/`, which git ignores:

```bash
pip install huggingface_hub                         # for the Hugging Face items (no login needed)
python -m agent.data.download --list                # what exists, size, licence, which experiment uses it
python -m agent.data.download --only e1_d2          # just what one experiment needs (about 21 GB)
python -m agent.data.download                       # everything (about 170 GB; resumable: re-run after a drop)
python -m agent.data.download --check --verify      # confirm your files match ours (sha256)
```

- **Three datasets need a person.** DocVQA and InfographicVQA require each user to register at rrc.cvc.uab.es, and
  the bird photos (CUB) must be downloaded in a browser. The script prints the steps and then checks the files.
- **On Colab**, add `--data /content/drive/MyDrive/spatial_llm/data/public` so the data survives the session.
- **Do not upload or share** DocVQA, InfographicVQA or CUB copies.

Details: [docs/07_EVIDENCE_PIPELINE.md](docs/07_EVIDENCE_PIPELINE.md).

## Requirements

- Python ≥ 3.11 (tested 3.13).
- PyTorch with CUDA (tested 2.14.1+cu130).
- transformers 5.19.0, numpy, pillow.
- A GPU with 24 GB or more for the language-model and zoom stages. A Colab T4 is too small; an L4 or A100 works.
- About 50 GB of disk for the workspace.

## License

The code and documentation in this repository are released under the [MIT License](LICENSE).

The Qwen2.5-VL-3B-Instruct model weights are **not** part of this repository: they are downloaded from Hugging Face on first use and are governed by their own license (see the model page). Accept that license separately before running the pipeline.
