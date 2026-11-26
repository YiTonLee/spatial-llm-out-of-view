"""The pipeline, stage by stage, in the order the project was built. Each stage is a list of Steps (runner.py).

Stage            what it produces                                         slides / docs
1 data           every video split, regenerated from fixed seeds          docs/01_DATA.md
2 cache          the image encoder's view of every frame (feature cache)  docs/01_DATA.md
3 model_w        model W: Qwen2.5-VL-3B + LoRA trained on ordinary videos docs/02_MODEL_STRUCTURE.md, block 1
4 estimator      the measuring part + estimated tracks for every split    block 2
5 record_lora    the LoRA that reads the written record (fix A)           block 3
6 rules          the calculator's motion rules, tuned (fix C)             block 4
7 ladder         model + calculator accuracy, A -> A+B -> A+B+C (eval9)   docs/04_RESULTS.md, table 1
8 versions       Versions 1-3 (extra looks), 3c (cheaper look)            block 5, table 2
9 look_choice    which frames to look at: last sighting of each ball      block 6, table 3
10 report        your numbers next to ours                                docs/04_RESULTS.md

Evidence-pipeline stages (branch evidence-pipeline; not part of `run.py all`, run them by name):
e0             cost of every answering step, accuracy-cost frontier     docs/07_EVIDENCE_PIPELINE.md, E0
e1_d1          same facts in five note formats, untrained and trained   docs/07_EVIDENCE_PIPELINE.md, E1

quick=True keeps every stage but shortens training and evaluation (minutes instead of hours); the numbers are then
meaningless, it only checks that everything runs. The data and cache stages are the same in both modes.
"""
import os
from . import config
from .config import SPLITS, last_video
from .runner import Step, copy_into

MODES = [s for s, v in SPLITS.items() if v[0] == 'modes']
ALL = list(SPLITS)


def _gen_code(split):
    n = SPLITS[split][1]
    return ['-c', f"import os, sts; sts.gen('{split}', {n}, workers=os.cpu_count())"]


def data(quick):
    steps = [Step(f'gen_{s}', f'generate {SPLITS[s][2]} videos of {s} ({SPLITS[s][0]})', _gen_code(s),
                  [last_video(s)], minutes='1-25 (CPU)')
             for s in ('train3', 'eval3', 'train8', 'eval8')]
    steps.append(Step('oracle_tracks_3', 'exact per-frame positions of the natural splits', ['gen_tracks.py', '3000'],
                      ['sts_data/train3_tracks.json', 'sts_data/eval3_tracks.json'], minutes='3 (CPU)'))
    steps += [Step(f'gen_{s}', f'generate {SPLITS[s][1]} motion-mode twin pairs of {s}', ['scenes_modes.py', s, str(SPLITS[s][1])],
                   [last_video(s)], minutes='2-5 (CPU)') for s in MODES]
    steps.append(Step('fingerprint', 'check the first videos of every split are bit-identical to ours', check_fingerprints))
    return steps


def check_fingerprints():
    import hashlib, json
    import numpy as np
    ref = json.load(open(config.REFERENCE / 'data_fingerprints.json'))
    bad = []
    for sp, want in ref.items():
        h = hashlib.sha256()
        for i in range(4):
            z = np.load(config.WORK / 'sts_data' / sp / f'{i:05d}.npz')
            h.update(z['frames'].tobytes()); h.update(str(z['meta']).encode())
        got = h.hexdigest()[:16]
        print(f'    {sp:7s} {got}  {"identical" if got == want else "DIFFERENT (expected " + want + ")"}')
        if got != want:
            bad.append(sp)
    if bad:
        print('    WARNING: regenerated videos differ (numpy version?). Results will differ slightly; see docs/03_RUNNING.md.')


def cache(quick):
    args = [f'{s}:{SPLITS[s][2]}' for s in ALL]
    return [Step('precache', 'run the frozen image encoder once on every frame of every split', ['precache.py'] + args,
                 [f'sts_data/vcache/{s}/{SPLITS[s][2] - 1:05d}.pt' for s in ALL], gpu=True, minutes='25 (~20 GB disk)')]


def model_w(quick):
    n = '40' if quick else '1000'
    return [Step('train_w', f'train model W: LoRA, {n} steps on ordinary videos (train3)',
                 ['sts.py', 'run', '--arm', 'lora', '--seed', '0', '--train_split', 'train3', '--n_train', n,
                  '--save_lora', 'w1k_s0.pt', '--tag', '_w1k_local', '--n_eval', '1', '--eval_split', 'eval8',
                  '--out', 'res_w1k_local.jsonl'],
                 ['w1k_s0.pt'], gpu=True, minutes='9', fresh=['res_w1k_local.jsonl'])]


def estimator(quick):
    steps = '100' if quick else '3000'
    rest = ['train8:1200', 'train9:600', 'eval9:600', 'eval9s:600', 'eval10:600', 'eval11:600', 'eval12:600', 'eval13:600']
    return [Step('train_estimator', f'train the measuring part ({steps} steps on train3 + train8); writes eval8/eval3 tracks',
                 ['state_estimator.py', '--steps', steps],
                 ['state_estimator.pt', 'eval8_est_tracks.json', 'eval3_est_tracks.json'], gpu=True, minutes='8'),
            Step('est_tracks', 'run the measuring part on every other split -> <split>_est_tracks.json',
                 ['est_infer.py'] + rest, [f'{s.split(":")[0]}_est_tracks.json' for s in rest], gpu=True, minutes='3')]


def record_lora(quick):
    n_train, n_eval = ('40', '20') if quick else ('1000', '600')
    return [Step('train_record_lora', 'train the LoRA that reads the written record (from W; train3 + train8)',
                 ['sts.py', 'run', '--seed', '1', '--arm', 'lora', '--n_eval', n_eval, '--eval_split', 'eval8,eval3',
                  '--text_state', '--vel_fit', 'lsq', '--hist', 'last1', '--init_lora', 'w1k_s0.pt', '--n_train', n_train,
                  '--train_split', 'train3,train8', '--save_lora', 'r1_lsq_s1.pt', '--est_tracks', 'eval8,eval3',
                  '--out', 'res_r_est.jsonl'],
                 ['r1_lsq_s1.pt'], gpu=True, minutes='8', fresh=['res_r_est.jsonl'])]


def rules(quick):
    return [Step('tune_rules', 'tune every motion rule of the calculator on train8 + train9 (grid search)',
                 ['state_rules.py'], ['state_rules_results.json'], minutes='20 (CPU)')]


# the model runs of the ladder: all on eval9, evaluation only, LoRA r1 (record reader) unless stated
LADDER_RUNS = {
    'm_w':             ('model only: W sees all 8 frames, no record', ['--init_lora', 'w1k_s0.pt']),
    'm_r_lsq':         ('A: record with a straight-line calculator', ['--est_tracks', 'eval9', '--state_rule', 'lsq']),
    'flow_lsq_est_dist': ('A + B: + distances in the sentence', ['--est_tracks', 'eval9', '--state_rule', 'lsq', '--dist_text', 'append']),
    'm_r_ca':          ('A + C: change-aware rule, no distances', ['--est_tracks', 'eval9', '--state_rule', 'ca']),
    'm_r_ca_dist':     ('A + B + C: the current system', ['--est_tracks', 'eval9', '--state_rule', 'ca', '--dist_text', 'append']),
    'm_r_ca_orc_dist': ('side test: A + B + C with exact positions', ['--state_rule', 'ca', '--dist_text', 'append']),
}


def ladder(quick):
    n_eval = '20' if quick else '600'
    base = ['sts.py', 'run', '--arm', 'lora', '--seed', '1', '--n_eval', n_eval, '--eval_split', 'eval9', '--n_train', '0']
    rec = ['--init_lora', 'r1_lsq_s1.pt', '--text_state', '--hist', 'last1']
    steps = []
    for name, (what, extra) in LADDER_RUNS.items():
        args = extra if name == 'm_w' else rec + extra
        steps.append(Step(name, what, base + args + ['--out', f'res_{name}.jsonl'], [f'res_{name}.jsonl'], gpu=True,
                          minutes='2', fresh=[f'res_{name}.jsonl']))
    steps.append(Step('calculator', 'the calculator answering directly (no model): straight line / IMM / change-aware',
                      ['calc_ladder.py'], ['calc_ladder.json'], minutes='1 (CPU)'))
    return steps


def versions(quick):
    q = ['--quick'] if quick else []
    sfx = '_quick' if quick else ''
    r5, r6, r7 = f'crop_runs_r5{sfx}', f'crop_runs_r6{sfx}', f'crop_runs_r7{sfx}'
    return [
        Step('round5', 'Versions 1, 2, 3: encode extra looks, train 3 heads x 3 seeds, score on eval10',
             ['crop_ablation_r5.py', 'all'] + q, [f'crop_ablation_r5{sfx}.json'], gpu=True, minutes='70'),
        Step('handoff_r5_r6', 'give round 6 the round-5 heads and features', lambda: (
            copy_into(r5, r6, 'head_*_s*.pt'), copy_into(r5, r6, '*_feats.pt'))),
        Step('round6', 'Versions 1h, 2h (same head start as Version 3): train, score on eval10',
             ['crop_ablation_r6.py', 'all'] + q, [f'crop_ablation_r6{sfx}.json'], gpu=True, minutes='30'),
        Step('handoff_r6_r7', 'give round 7 the heads of Versions 1h, 2h (round 6) and 3 (round 5)', lambda: (
            copy_into(r6, r7, 'head_base_box_s*.pt'), copy_into(r6, r7, 'head_global_box_s*.pt'),
            copy_into(r5, r7, 'head_crop_s*.pt'))),
        Step('round7', 'Version 3c (same square, 4x fewer tokens): encode, train, score on eval11 + eval10',
             ['crop_ablation_r7.py', 'all'] + q, [f'crop_ablation_r7{sfx}.json'], gpu=True, minutes='70'),
    ]


def look_choice(quick):
    q = ['--quick'] if quick else []
    sfx = '_quick' if quick else ''
    r5 = f'crop_runs_r5{sfx}'
    steps = [
        Step('handoff_r5_r8', "give round 8 Version 3's heads", lambda: copy_into(r5, f'look_runs_r8{sfx}', 'head_crop_s*.pt')),
        Step('round8', 'zoom on every sighting (train9, eval10, eval12); score the look-choosing rules on eval12',
             ['look_choice_r8.py', 'all'] + q, [f'look_choice_r8{sfx}.json'], gpu=True, minutes='20'),
    ]
    if not quick:   # round 9 has no quick mode (it reuses round 8's readings and adds eval13)
        steps += [
            Step('round9_predict', 'zoom on every sighting of eval13', ['look_choice_r9.py', 'predict'],
                 ['look_runs_r9/preds_eval13.json'], gpu=True, minutes='7'),
            Step('round9_score', 'score the look-choosing rules on eval13', ['look_choice_r9.py', 'score'],
                 ['look_choice_r9.json'], minutes='5 (CPU)'),
        ]
    return steps


def e0(quick):
    n = '2' if quick else '20'
    return [Step('e0_ledger', f'time every answering step on {n} eval8 scenes; frontier and gates (docs/gates/e0_gates.md)',
                 ['e0_ledger.py', '--n', n, '--out', 'e0'], ['e0/e0_ledger.json'], gpu=True, minutes='3')]


E1_UNTRAINED = ','.join([f'F{i}:est:v8' for i in range(5)] + [f'F{i}:est:l1' for i in range(5)] +
                        [f'F{i}:{q}:v8' for q in ('orc', 'cor', 'shuf') for i in range(1, 5)] + ['F4:neg:v8'])


def e1_d1(quick):
    n = ['--n', '4'] if quick else []
    steps_ = ['--steps', '20'] if quick else []
    steps = [Step('e1_untrained', 'untrained model, 23 note variants on eval8 (docs/gates/e1_gates.md)',
                  ['e1_d1.py', 'eval', '--arms', E1_UNTRAINED, '--out', 'e1/eval_untrained.json'] + n,
                  ['e1/eval_untrained.json'], gpu=True, minutes='105')]
    for f in ('F1', 'F2', 'F3', 'F4'):
        steps += [Step(f'e1_train_{f}', f'LoRA trained with {f} notes (from W, true-track notes)',
                       ['e1_d1.py', 'train', '--fmt', f, '--save', f'e1/lora_{f}.pt'] + steps_, [f'e1/lora_{f}.pt'],
                       gpu=True, minutes='22' if f == 'F1' else '8'),
                  Step(f'e1_eval_{f}', f'the {f} LoRA on 5 note variants',
                       ['e1_d1.py', 'eval', '--lora', f'e1/lora_{f}.pt', '--arms',
                        f'{f}:est:v8,{f}:orc:v8,{f}:cor:v8,{f}:shuf:v8,F0:est:v8', '--out', f'e1/eval_lora_{f}.json'] + n,
                       [f'e1/eval_lora_{f}.json'], gpu=True, minutes='20')]
    steps.append(Step('e1_report', 'table, paired differences, gates', ['e1_d1.py', 'report'], ['e1/E1_D1_table.md'],
                      minutes='1 (CPU)', fresh=['e1/E1_D1_table.md']))
    return steps


def report(quick):
    from . import report as R
    return [Step('report', 'collect every result and compare with ours', lambda: R.main(quick))]


STAGES = {'data': data, 'cache': cache, 'model_w': model_w, 'estimator': estimator, 'record_lora': record_lora,
          'rules': rules, 'ladder': ladder, 'versions': versions, 'look_choice': look_choice, 'report': report,
          'e0': e0, 'e1_d1': e1_d1}
CORE = ['data', 'cache', 'model_w', 'estimator', 'record_lora', 'rules', 'ladder', 'versions', 'look_choice', 'report']
EVIDENCE = ['e0', 'e1_d1']   # evidence-pipeline experiments: run by name, after model_w and estimator
