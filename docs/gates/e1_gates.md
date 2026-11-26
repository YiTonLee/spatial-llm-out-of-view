# E1 gates, domain 1 (synthetic out-of-view), written 2026-11-25 BEFORE any E1 run

> Repository copy. Paths starting with `pilot/` are the authors' working folder; in this repository the code is `src/e0_ledger.py` and `src/e1_d1.py` (run with `python run.py e0` / `e1_d1`), helpers in `agent/`, and outputs in `workspace/e0/`, `workspace/e1/`. The plan file named here is kept outside git.

Plan: `Github Repo/doc/2026_11_25/PLAN_evidence_pipeline_2026-11-25.md`, E1. Code: `pilot/e1_d1.py`.
Domain 2 (VSR, real photos) runs after COCO 2017 and the tools are downloaded; its gates are added then, before it runs.

## Same information, five formats
Every packet is computed from the same estimated tracks (`eval8_est_tracks.json`, Stage D estimator, least-squares
velocity). The "oracle" block uses the true tracks and is reported separately.
- F0 none: no packet.
- F1 raw: every visible sighting per ball (time, x, z in last-frame axes) + camera speed and yaw rate
  (the `track_text` format). The student must fit motion, propagate and change frames itself.
- F2 camera-frame coordinates: both balls' positions at the asked time t*, in the camera axes at t* (Stage D text).
- F3 derived: at t*, the signed left-right offset between the balls and each ball's distance from the camera.
- F4 predicate: at t*, "A is to the left/right of B" and "A/B is closer to the camera".
- Map picture (F5) and crop (F6): not in this domain-1 run. Crops are meaningless for a hidden object; the map
  picture needs a new image path and is deferred to the domain-2 run, where it is a depth-map picture.

## Conditions
- Student: base Qwen2.5-VL-3B, zero-shot ("untrained"); then one LoRA per format (F1-F4), 1000 steps, seed 1,
  initialised from W (`w1k_s0.pt`), trained with ORACLE-track packets on train3+train8 (the Stage C/D recipe), and
  tested with estimated packets.
- Visual input: 8 frames ("with video") or last frame only ("last frame", the R setting).
- Packet quality: correct (estimated), oracle, corrupted (ball A's track shifted by (+0.8, +1.5) m in every
  frame), shuffled (tracks of scene (i+101) mod 600), negated (F4 only: both predicates flipped).
- Test set: eval8, all 600 scenes, all questions. Metric: twin both-correct on differing questions, pair-level
  bootstrap 95% CI (1000 resamples), plus differing accuracy. Format differences use a paired bootstrap over pairs.

## Gates
- **G1 (H1a, pass rule):** untrained student, 8 frames, correct packets: max(F3, F4) − F1 >= +10 points of twin
  both-correct, with the lower 95% bound of the paired difference > 0.
- **G2 (H1b, reported):** the same gap for the trained students. "Shrinks but does not close" = trained gap < untrained
  gap and trained gap lower bound > 0.
- **G3 (H1c, reported, no threshold):** trained student with corrupted packets vs trained student with no packet
  (F0 on the same LoRA). Corrupted < none means blind trust; this feeds E3.
- Reference rows: K (calculator, 74.5) from E0.
