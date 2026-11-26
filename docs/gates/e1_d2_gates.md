# E1 gates, domain 2 (real photos, VSR), written 2026-11-26 BEFORE any domain-2 run

> Repository copy. Paths starting with `pilot/` are the authors' working folder; in this repository the code is `src/e0_ledger.py` and `src/e1_d1.py` (run with `python run.py e0` / `e1_d1`), helpers in `agent/`, and outputs in `workspace/e0/`, `workspace/e1/`. The plan file named here is kept outside git.

Plan: `Github Repo/doc/2026_11_25/PLAN_evidence_pipeline_2026-11-25.md`, E1. Domain-1 result: `pilot/e1/E1_D1_RESULTS.md`.
Code (to write): `pilot/e1_d2.py` (repo copy `src/e1_d2.py`, stage `e1_d2`). Nothing below has been run.

## The task
VSR random split (`data/public/vsr/{train,dev,test}.jsonl`: 7,680 / 1,097 / 2,195 items; COCO 2017 images read
straight from `coco2017/{train,val}2017.zip`). Each item is one photo and one sentence "The <A> is <relation> the
<B>."; the answer is true or false (test: 54% true). All captions follow that template (checked: 100% parse).

Plain: the model sees a photo and a sentence like "The cat is under the table" and says whether it is true. The tools
find the two objects and measure how far away each is, and we hand the model that result written different ways.

**Measurable relations** (the tools can decide them; the pass rules use only these; 1,170 test items):
- left-right: left of, at the left side of, right of, at the right side of (245 test items);
- up-down: above, over, on top of, below, under, beneath (426);
- depth: in front of, behind, at the back of (314);
- containment: in, inside, contains, surrounding (185).
All other relations (touching, facing, next to, near, ...) are reported separately as a control: a note should not
help there, and must not hurt.

## Tools (fixed, not trained; same output feeds every format)
- Grounding DINO tiny (`models/grounding-dino-tiny`): best box for A and for B (text prompt = the name), score.
  Below a 0.25 score the object is "not found", and every format says so.
- Depth Anything V2 Small (`models/Depth-Anything-V2-Small-hf`): relative inverse depth (larger = nearer); the median
  inside each box.

## Same information, seven formats (all from the same tool outputs; every format states ALL facts, not only the asked one)
- F0 none.
- F1 raw tool output: pixel boxes [x1, y1, x2, y2] with image size and scores; median depth values.
- F2 normalised positions: box centres and sizes as fractions of the image; nearness 0-1.
- F3 derived: centre offsets (image widths / heights, signed), nearness difference in %, share of each box inside
  the other.
- F4 predicates: "A is left of B; A is above B; A is nearer the camera than B; neither box contains the other".
- F5 depth-map picture: the depth map with both boxes drawn and labelled, as one extra image.
- F6 crop: the union of both boxes, enlarged, as one extra image.

Rules that turn F3 numbers into F4 words (fixed before the run): left/right and above/below by the sign of the centre
offset; nearer by the sign of the nearness difference; "A inside B" if >= 80% of A's box lies in B's box.

## Conditions
- Student: base Qwen2.5-VL-3B, untrained; then one LoRA per text format F1-F4, 1000 steps, seed 1, lr 1e-4, trained on
  VSR train with **tool** packets (not clean truth, the DAgger lesson), with **20% of training items given no note**
  so a missing note is a seen input (domain-1 lesson: trained students fell to 2.6-4.5% with no note).
- View: with the photo, or note only (no photo).
- Note quality: tool (correct as measured), shuffled (another photo's note, names kept), negated (F4, every fact flipped).
  No oracle: VSR has no boxes, and mapping caption names to COCO instances is ambiguous; the tool-agreement check
  below stands in for it.
- K2, the rule alone (no VLM): answer the measurable items from the F4 facts directly (reference row, like K in E0).
- Metric: accuracy on the test split, measurable subset and control subset separately; 95% CIs by bootstrap over
  images (1000 resamples); format differences by paired bootstrap. Cost per arm with `agent/cost.py`: input tokens and
  one-shot ms (image encoder + tools + prefill + one answer token).

## Check before the test run (on dev only)
- C0 tool agreement: share of measurable dev items where the F4 fact for the asked relation agrees with the label;
  and the share where both objects are found. **Stop rule:** if agreement < 60%, the notes carry too little
  information to compare formats: stop, report it as a tool finding, and do not run the test.

## Gates
- **G1 (pass rule, H1a on real photos):** untrained, with photo, tool notes, measurable test items:
  max(F3, F4) − F1 >= +5 accuracy points, lower 95% bound > 0.
  (+5, not domain 1's +10: on a true/false task chance is 50% and the base model is far above zero, so gaps are
  compressed.)
- **G2 (pass rule, does the note add to the photo):** untrained, measurable items: max(F3, F4) with photo − photo
  alone (F0) >= +3 points, lower 95% bound > 0. Prior support: P² (2604.12896) reports 15-40 point gains on BLINK
  for 4B models from rewritten tool outputs.
- G3 (reported): best student arm vs K2 (does the VLM add anything beyond the rule?); note-only vs photo-only.
- G4 (reported): trained gap max(F3, F4) − F1; trained shuffled − none (trust); trained none vs untrained none.
- G5 (reported): F5 and F6 vs F0 and vs F4, with their token and time cost (picture note vs text note).
- Control (reported): on non-measurable relations, best note − none should be within ±2 points.

## Compute (needs the user's go: > 30 min)
Tools on ~11k images ≈ 15-20 min; untrained 15 arms × 2,195 ≈ 20-30 min; 4 LoRAs × (≈10 min train + ≈5 min eval)
≈ 60 min. Total ≈ 1.5-2 h on the laptop GPU.
