"""E1, domain 1: same information, different packet formats (gates: docs/gates/e1_gates.md, written before running).
Results 2026-11-25: G1 pass (untrained F4 - F1 = +22.9 [19.1, 26.8]); see docs/07_EVIDENCE_PIPELINE.md.

Plain: the small model gets the same facts about the two balls written five ways (nothing; a raw list of sightings;
positions at the asked time; distances; a plain-words statement). We check which way it can actually use, with and
without training, and what happens when the note is wrong.

Technical: arms are "<fmt>:<quality>:<view>" with fmt in F0..F4, quality in est|orc|cor|shuf|neg, view in v8|l1.
`eval` scores eval8 (600 scenes, all questions) with cached ViT features; `train` trains one LoRA per format with
the Stage C/D recipe (oracle packets in training, from W) by wrapping sts.load; `report` computes twin both-correct
with pair-level bootstrap CIs and paired differences.
Usage:
  python run.py e1_d1                      # the whole queue through the pipeline (about 4 h on one 24 GB GPU)
  python src/e1_d1.py eval --lora "" --arms F0:est:v8,F1:est:v8 --out e1/untrained.json   (in the workspace)
  python src/e1_d1.py train --fmt F3 --save e1/lora_F3.pt
  python src/e1_d1.py report
Needs in the workspace: sts_data/{eval8,train3,train8} + feature cache, eval8_est_tracks.json, w1k_s0.pt.
"""
import argparse, json, os, sys, time
import numpy as np, torch
import sts, ladder_stage_a
from ladder_stage_a import est_velocity

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))           # repo root, for agent/
from agent import packets  # noqa: E402
OUT = os.path.join(HERE, 'e1')
ladder_stage_a.VEL['mode'] = 'lsq'
EST = json.load(open(os.path.join(HERE, 'eval8_est_tracks.json')))
N_EVAL = sts.EVAL_SCENES['eval8']


def state_at(tr, q):
    """Both balls' positions at the asked time q['t'], in the camera axes at that time (observed-only, LSQ velocity)."""
    pos = np.array(tr['pos'], float); vis = np.array(tr['vis'], bool)
    p7, vv = [], []
    for j in range(2):
        v, k, _ = est_velocity(pos, vis, j, True)
        p7.append(pos[k, j] + (7 - k) * v); vv.append(v)
    return sts.transition(torch.tensor(np.array(p7)).float(), torch.tensor(np.array(vv)).float(),
                          torch.tensor(float(tr['cam_speed'])), torch.tensor(float(tr['yaw_rate'])),
                          torch.tensor(q['t'] - 7.0)).numpy()


def packet(fmt, tr, names, q, negate=False):
    """The packet text for one question; every format is computed from the same tracks `tr` (agent/packets.py)."""
    if fmt == 'F0':
        return ''
    if fmt == 'F1':
        return sts.track_text(tr, names)
    return packets.render(fmt, names, state_at(tr, q), negate=negate)


def tracks_for(quality, i, sc):
    if quality == 'orc':
        return sc['tracks']
    if quality == 'shuf':
        return EST[(i + 101) % N_EVAL]
    tr = EST[i]
    if quality == 'cor':
        pos = np.array(tr['pos'], float); pos[:, 0] += np.array([0.8, 1.5])
        tr = dict(tr, pos=pos.tolist())
    return tr


def cmd_eval(a):
    proc, model, params, _ = sts.build(0, None)
    if a.lora:
        sts.swap_lora(params, [v.to('cuda') for v in torch.load(a.lora)])
    model.eval()
    arms = a.arms.split(',')
    rows = []; t0 = time.time()
    with torch.no_grad():
        for i in range(a.n or N_EVAL):
            if i % 50 == 0:
                print(f'{os.path.basename(a.out)} {i}/{N_EVAL} {time.time() - t0:.0f}s', flush=True)
            sc = sts.load('eval8', i)
            fv = sts.vis_feats(model, proc, 'eval8', i, sc['frames'])
            for arm in arms:
                fmt, qual, view = arm.split(':')
                tr = tracks_for(qual, i, sc)
                qs = sc['questions']
                sts.HIST['last1'] = view == 'l1'
                items = [(fv, packet(fmt, tr, sc['names'], q, negate=qual == 'neg') + q['question']) for q in qs]
                lg = torch.cat([model(**sts.encode_cached(proc, items[c:c + 4]), logits_to_keep=1).logits[:, -1].float()
                                for c in range(0, len(items), 4)])
                for qi, q in enumerate(qs):
                    o = sts.opt_ids(proc, q)
                    pred = int(lg[qi, o[1]] > lg[qi, o[0]])
                    rows.append(dict(arm=arm, scene=i, pair=sc['pair'], twin=sc['twin'], t=q['t'], qtype=q['qtype'],
                                     correct=int(pred == q['label']), differs=q['differs'], static_ok=q['static_ok']))
    sts.HIST['last1'] = False
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(lora=a.lora, arms=arms, sec=round(time.time() - t0), rows=rows), open(a.out, 'w'))
    print(json.dumps(summarize(rows), indent=1))


def cmd_train(a):
    """One LoRA per format: oracle-track packets in training (Stage C/D recipe), from W, --steps (1000), seed 1."""
    _load = sts.load

    def load(split, i):
        sc = _load(split, i)
        if split not in sts.EVAL_SCENES:            # training splits only; evaluation is done by cmd_eval
            tr = sts.scene_tracks(split, i, sc)
            for q in sc['questions']:
                q['question'] = packet(a.fmt, tr, sc['names'], q) + q['question']
        return sc
    sts.load = load
    sts.run('lora', 1, a.steps, 1, 1e-4, os.path.join(OUT, 'train_runs.jsonl'), tr_split='train3,train8', ev_split='eval8',
            init_lora=os.path.join(HERE, 'w1k_s0.pt'), save_lora=a.save)


def both_correct(rows):
    """{(pair, t, qtype): 1 if both twins right} over differing questions."""
    by = {}
    for r in rows:
        if r['differs']:
            by.setdefault((r['pair'], r['t'], r['qtype']), []).append(r['correct'])
    return {k: int(all(v)) for k, v in by.items() if len(v) == 2}


def summarize(rows, B=1000):
    rng = np.random.default_rng(0); out = {}
    for arm in dict.fromkeys(r['arm'] for r in rows):
        rs = [r for r in rows if r['arm'] == arm]
        bc = both_correct(rs)
        pairs = sorted({k[0] for k in bc}); per = {p: [v for k, v in bc.items() if k[0] == p] for p in pairs}
        bs = [100 * np.mean([v for p in rng.choice(pairs, len(pairs)) for v in per[p]]) for _ in range(B)]
        out[arm] = dict(both=round(100 * np.mean(list(bc.values())), 1),
                        ci=[round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)],
                        differ=round(100 * np.mean([r['correct'] for r in rs if r['differs']]), 1),
                        all=round(100 * np.mean([r['correct'] for r in rs]), 1), n_twin=len(bc))
    return out


def paired(rows_a, rows_b, B=1000):
    """Difference in twin both-correct (a - b), paired bootstrap over pairs."""
    ba, bb = both_correct(rows_a), both_correct(rows_b)
    keys = sorted(set(ba) & set(bb)); pairs = sorted({k[0] for k in keys})
    per = {p: [ba[k] - bb[k] for k in keys if k[0] == p] for p in pairs}
    rng = np.random.default_rng(1)
    bs = [100 * np.mean([v for p in rng.choice(pairs, len(pairs)) for v in per[p]]) for _ in range(B)]
    return round(100 * np.mean([ba[k] - bb[k] for k in keys]), 1), [round(float(np.percentile(bs, 2.5)), 1),
                                                                    round(float(np.percentile(bs, 97.5)), 1)]


def cmd_report(a):
    R = {}
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.json') and f.startswith('eval_'):
            d = json.load(open(os.path.join(OUT, f)))
            for r in d['rows']:
                R.setdefault((f[5:-5], r['arm']), []).append(r)
    L = ['| student | arm | twin both-correct % [95% CI] | differing % |', '|---|---|---|---|']
    S = {}
    for (stu, arm), rs in R.items():
        s = summarize(rs)[arm]; S[(stu, arm)] = s
        L.append(f'| {stu} | {arm} | {s["both"]} [{s["ci"][0]}, {s["ci"][1]}] | {s["differ"]} |')
    L += ['', '| comparison | difference [95% CI] |', '|---|---|']
    gates = {}
    if ('untrained', 'F1:est:v8') in R:
        for f in ('F3', 'F4', 'F2'):
            if ('untrained', f'{f}:est:v8') in R:
                d = paired(R[('untrained', f'{f}:est:v8')], R[('untrained', 'F1:est:v8')])
                gates[f'untrained {f}-F1'] = d; L.append(f'| untrained {f} − F1 (8 frames) | {d[0]} [{d[1][0]}, {d[1][1]}] |')
    for f in ('F1', 'F2', 'F3', 'F4'):
        stu = f'lora_{f}'
        if (stu, f'{f}:est:v8') in R and (stu, 'F0:est:v8') in R:
            d = paired(R[(stu, f'{f}:est:v8')], R[(stu, 'F0:est:v8')])
            L.append(f'| {stu}: {f} correct − no packet | {d[0]} [{d[1][0]}, {d[1][1]}] |')
            if (stu, f'{f}:cor:v8') in R:
                d = paired(R[(stu, f'{f}:cor:v8')], R[(stu, 'F0:est:v8')]); gates[f'{stu} cor-none'] = d
                L.append(f'| {stu}: {f} corrupted − no packet | {d[0]} [{d[1][0]}, {d[1][1]}] |')
    for f in ('F3', 'F4'):
        if (f'lora_{f}', f'{f}:est:v8') in R and ('lora_F1', 'F1:est:v8') in R:
            d = paired(R[(f'lora_{f}', f'{f}:est:v8')], R[('lora_F1', 'F1:est:v8')]); gates[f'trained {f}-F1'] = d
            L.append(f'| trained {f} − trained F1 | {d[0]} [{d[1][0]}, {d[1][1]}] |')
    open(os.path.join(OUT, 'E1_D1_table.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(dict(summary={f'{k[0]}|{k[1]}': v for k, v in S.items()}, gates={k: list(v) for k, v in gates.items()}),
              open(os.path.join(OUT, 'e1_d1_summary.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['eval', 'train', 'report'])
    ap.add_argument('--lora', default=''); ap.add_argument('--arms', default='')
    ap.add_argument('--out', default=''); ap.add_argument('--fmt', default='F3'); ap.add_argument('--save', default='')
    ap.add_argument('--steps', type=int, default=1000, help='LoRA steps (1000 in the logged run; small for smoke tests)')
    ap.add_argument('--n', type=int, default=0, help='scenes to evaluate (0 = all 600; small values for smoke tests)')
    a = ap.parse_args()
    {'eval': cmd_eval, 'train': cmd_train, 'report': cmd_report}[a.cmd](a)
