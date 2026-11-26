"""E0: evidence ledger and cost accounting (gates: docs/gates/e0_gates.md, written before this run).
Results 2026-11-25: G1 pass (frontier = {K}), G2 miss one-shot; see docs/07_EVIDENCE_PIPELINE.md.

Plain: for each way of giving the small model its evidence (raw video, video + a short state note, last frame + note,
or a calculator with no language model at all), put accuracy next to the full time it costs to answer one question,
counting every step: loading frames, the vision encoder, the tools, reading the prompt and writing the answer.

Technical: accuracy is read from earlier runs (LADDER_RESULTS.md, eval8, twin both-correct). Cost is timed here on N
real eval8 scenes (RTX 5090, bf16, batch 1, CUDA-synchronised medians), per stage:
decode -> ViT+merger (with image preprocessing) -> estimator (8 frames x 2 objects) -> state/packet text -> LLM
prefill -> decode of the answer. Two settings: A one-shot (all frames encoded at question time), B streaming (frame
work already paid; per-question cost = packet text + prefill + decode).
Usage: python run.py e0   (or, in the workspace: python src/e0_ledger.py [--n 20] [--out e0])
Needs in the workspace: sts_data/eval8, state_estimator.pt, eval8_est_tracks.json.
Writes <out>/e0_ledger.json, <out>/e0_ledger.md, <out>/run_info.json.
"""
import argparse, json, os, platform, statistics, subprocess, sys, time
import numpy as np, torch
from PIL import Image
import sts, ladder_stage_a, ladder_stage_b
from state_estimator import StateEstimator

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))           # repo root, for agent/
from agent.cost import tsync, frontier as _frontier  # noqa: E402

# Accuracy from earlier runs (eval8, 300 twin pairs): (differing %, twin both-correct %, source)
ACC = {
    'V': (51.2, 5.5, 'LADDER_RESULTS.md Stage C, C0 LoRA only'),
    'D': (83.2, 67.5, 'LADDER_RESULTS.md Stage D, D-est'),
    'R': (86.0, 72.9, 'LADDER_RESULTS.md Stage R, R-est seed 1 (seeds 2/3: 73.2 / 73.8)'),
    'K': (86.9, 74.5, 'LADDER_RESULTS.md Stage D, calculator on estimated state (lsq)'),
}
ORACLE = {
    'D-orc': (91.0, 82.4, 'Stage D'), 'R-orc': (93.8, 87.8, 'Stage R'), 'K-orc': (98.6, 97.2, 'Stage A, B4 executor'),
}
LABEL = {'V': 'video only (8 frames)', 'D': '8 frames + state text', 'R': 'last frame + state text',
         'K': 'calculator on estimated state (no VLM)'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default=os.path.join(HERE, 'e0'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    proc, model, _, _ = sts.build(0, None); model.eval()
    est = StateEstimator().cuda(); est.load_state_dict(torch.load(os.path.join(HERE, 'state_estimator.pt'))); est.eval()
    est_tracks = json.load(open(os.path.join(HERE, 'eval8_est_tracks.json')))
    ladder_stage_a.VEL['mode'] = 'lsq'
    rows = {k: [] for k in ('decode', 'vit8', 'vit1', 'est', 'text', 'pre_V', 'pre_D', 'pre_R', 'dec1',
                            'tok_V', 'tok_D', 'tok_R')}
    gen_ok = None
    for i in range(a.n):
        path = os.path.join(sts.DATA, 'eval8', f'{i:05d}.npz')
        sc = {}

        def decode():
            sc.update(sts.load('eval8', i)); sc['imgs'] = [Image.fromarray(f) for f in sc['frames']]
        rows['decode'].append(tsync(decode, reps=3, warm=1))
        q = sc['questions'][-1]

        def vit(imgs):
            enc = proc.image_processor(images=imgs, return_tensors='pt')
            out = model.model.get_image_features(enc['pixel_values'].cuda(), enc['image_grid_thw'].cuda(), return_dict=True)
            return torch.cat(out.pooler_output, 0)
        with torch.no_grad():
            rows['vit8'].append(tsync(lambda: vit(sc['imgs'])))
            rows['vit1'].append(tsync(lambda: vit(sc['imgs'][-1:])))
            feats = vit(sc['imgs'])                                   # (8*64, 2048)
            tok = feats.view(8, 64, -1).repeat_interleave(2, 0)       # 8 frames x 2 objects
            col = torch.tensor(sc.get('colour_ids', [0, 1]) * 8).cuda()[:16]
            rows['est'].append(tsync(lambda: est(tok, col)))
        tr = dict(est_tracks[i]); state = {}

        def text():
            state['t'] = ladder_stage_b.propagated_text(tr, sc['names'], q)
        rows['text'].append(tsync(text, reps=20, warm=2) * 1.0)
        with torch.no_grad():
            for arm, last1, pre in (('V', False, ''), ('D', False, state['t']), ('R', True, state['t'])):
                sts.HIST['last1'] = last1
                enc = sts.encode_cached(proc, [(feats, pre + q['question'])], n_frames=8)
                rows['tok_' + arm].append(int(enc['input_ids'].shape[1]))
                rows['pre_' + arm].append(tsync(lambda: model(**enc, logits_to_keep=1)))
            sts.HIST['last1'] = False
            if i == 0:  # per-token decode cost, measured once on a text-only prompt of the same length as D
                ids = proc.tokenizer('word ' * (rows['tok_D'][-1] - 1), return_tensors='pt').input_ids.cuda()
                g = lambda k: model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=k,
                                             min_new_tokens=k, do_sample=False)
                try:
                    rows['dec1'].append((tsync(lambda: g(5)) - tsync(lambda: g(1))) / 4); gen_ok = True
                except Exception as e:  # noqa: BLE001 -- report, do not hide
                    gen_ok = repr(e)[:200]
        print(i, {k: round(v[-1], 1) for k, v in rows.items() if v}, flush=True)

    med = {k: statistics.median(v) for k, v in rows.items() if v}
    dec = med.get('dec1', 0.0)
    frame = med['decode'] + med['vit8'] + med['est']
    costA = {'V': med['decode'] + med['vit8'] + med['pre_V'] + dec,
             'D': frame + med['text'] + med['pre_D'] + dec,
             'R': frame + med['text'] + med['pre_R'] + dec,
             'K': frame + med['text']}
    costB = {'V': med['pre_V'] + dec, 'D': med['text'] + med['pre_D'] + dec,
             'R': med['text'] + med['pre_R'] + dec, 'K': med['text']}
    toks = {'V': med['tok_V'], 'D': med['tok_D'], 'R': med['tok_R'], 'K': 0}

    def frontier(cost):
        return _frontier(cost, {k: ACC[k][1] for k in cost})
    fA = frontier(costA)
    g1 = any(k in fA for k in 'DRK') and 'V' not in fA
    best = max(ACC, key=lambda k: ACC[k][1])
    ref = 'V' if best in 'DRK' else best
    def g2(cost):
        return [p for p in 'DRK' if p != best and ACC[p][1] >= ACC[best][1] - 2 and cost[p] <= 0.25 * cost[ref]] + \
               ([best] if best in 'DRK' and cost[best] <= 0.25 * cost[ref] else [])
    res = dict(median_ms=med, cost_A_ms=costA, cost_B_ms=costB, tokens=toks, acc=ACC, oracle=ORACLE,
               frontier_A=fA, frontier_B=frontier(costB), G1=g1, best=best, G2_reference=ref,
               G2_pass_A=g2(costA), G2_pass_B=g2(costB),
               token_ratio_vs_V={k: toks[k] / toks['V'] for k in toks}, decode_step=gen_ok, n=a.n)
    json.dump(res, open(os.path.join(a.out, 'e0_ledger.json'), 'w'), indent=1)
    L = ['| condition | differing % | both-correct % | input tokens | A one-shot ms | B streaming ms | A cost / V |',
         '|---|---|---|---|---|---|---|']
    for k in 'VDRK':
        L.append(f'| {k}: {LABEL[k]} | {ACC[k][0]} | {ACC[k][1]} | {toks[k]:.0f} | {costA[k]:.0f} | {costB[k]:.0f} | '
                 f'{costA[k] / costA["V"]:.2f} |')
    L += ['', 'Stage medians (ms): ' + ', '.join(f'{k} {v:.1f}' for k, v in med.items()),
          f'G1 (frontier A) = {g1}; frontier A = {fA}; frontier B = {res["frontier_B"]}',
          f'G2: best = {best}, reference = {ref}; passing A = {res["G2_pass_A"]}, passing B = {res["G2_pass_B"]}']
    open(os.path.join(a.out, 'e0_ledger.md'), 'w').write('\n'.join(L) + '\n')
    try:
        commit = subprocess.run(['git', '-C', os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    json.dump(dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), repo_commit=commit, gpu=torch.cuda.get_device_name(0),
                   torch=torch.__version__, python=platform.python_version(), n_scenes=a.n, split='eval8',
                   script='src/e0_ledger.py', gates='docs/gates/e0_gates.md'),
              open(os.path.join(a.out, 'run_info.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
