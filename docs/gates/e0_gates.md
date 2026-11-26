# E0 gates (written 2026-11-25, BEFORE the timing run)

> Repository copy. Paths starting with `pilot/` are the authors' working folder; in this repository the code is `src/e0_ledger.py` and `src/e1_d1.py` (run with `python run.py e0` / `e1_d1`), helpers in `agent/`, and outputs in `workspace/e0/`, `workspace/e1/`. The plan file named here is kept outside git.

Plan: `Github Repo/doc/2026_11_25/PLAN_evidence_pipeline_2026-11-25.md`, section E0.

## What is measured

Accuracy comes from runs already done (no new accuracy runs), on eval8 (300 twin pairs), metric = twin
both-correct, plus the "differing" accuracy. Sources: `LADDER_RESULTS.md`, `res_*.jsonl`, `calc_ladder.json`.
Oracle conditions (true tracks) are reported in a separate block and never count as a tool's result.

Cost is measured now, by `e0_ledger.py`, on the RTX 5090 laptop, bf16, batch 1, on 20 real eval8 scenes,
median of repeats. The accounting boundary, once for all conditions:
1. frame decode: frames come from the cached `.npz` scenes (the synthetic renderer); timed as array load + PIL
   conversion;
2. vision encoding (ViT + merger) of every frame the condition needs;
3. tools: state estimator over frames + state cell + text rendering of the packet;
4. LLM prefill over the condition's real prompt;
5. decode: generation of the answer (max 4 new tokens).

Two settings are reported:
- **A, one-shot (primary):** the question arrives with an 8-frame clip; every frame is encoded at question time.
- **B, streaming:** frames are encoded as they arrive (paid anyway); per-question cost = only what happens after
  the question (prefill + decode, plus the packet rendering).

## Conditions (eval8, estimated tracks unless "oracle")
- V: video only, 8 frames, LoRA C0
- D: 8 frames + estimated state text (D-est)
- R: last frame + estimated state text (R-est, seeds 1-3)
- K: calculator on estimated state, no VLM
- oracle block: D-orc, R-orc, calculator on oracle tracks

## Gates (fixed now)
- **G1 (frontier, H0's first half):** in setting A, at least one packet condition (D, R or K) is on the
  accuracy-vs-total-cost Pareto frontier, and V is not on it.
- **G2 (the plan's pass rule):** in setting A, at least one packet condition P has
  acc(P) >= acc(best) - 2 points and cost(P) <= 0.25 x cost(best), where best = the most accurate non-oracle
  condition. If best is itself a packet condition, the rule is read against V instead
  (acc(P) >= acc(best) - 2 and cost(P) <= 0.25 x cost(V)), and this reading is reported as such.
- **G3 (token saving, reported, not a pass rule):** the same ratio in student input tokens, and setting B.

A miss on G2 in setting A with a pass in B is informative: it says the saving exists only when frames are
encoded anyway (streaming), and E1-E4 must account for it.
