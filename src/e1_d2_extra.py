"""E1 domain 2, exploratory analysis written AFTER the run (not a gate): note-only format gaps, and the
untrained student split by whether the tool's fact for the asked relation was right (K2 correct) or wrong.
Writes e1d2/e1_d2_extra.json (python run.py e1_d2 runs it after the report)."""
import json, os
import numpy as np
from e1_d2 import items, k2, paired, acc_ci, OUT, REL  # src/e1_d2.py

d = json.load(open(os.path.join(OUT, 'eval_untrained.json')))
rows = {}
for r in d['rows']:
    rows.setdefault(r['arm'], []).append(r)
tl = json.load(open(os.path.join(OUT, 'tools_test.json')))
its = items('test')[:len(tl)]
tool_ok = {}
for i, it in enumerate(its):
    if it['relation'] not in REL:
        continue
    p = k2(it, tl[i])
    tool_ok[i] = (p == it['label'])
M = {a: [r for r in rs if r['measurable']] for a, rs in rows.items()}
out = {}
for a, b in [('F3:tool:none', 'F1:tool:none'), ('F4:tool:none', 'F1:tool:none'), ('F4:tool:none', 'F0:tool:none'),
             ('F3:tool:none', 'F0:tool:none')]:
    out[f'note only: {a} - {b}'] = paired(M[a], M[b])
for name, ok in [('tool right', True), ('tool wrong', False)]:
    sub = {a: [r for r in rs if tool_ok[r['i']] == ok] for a, rs in M.items()}
    out[f'{name}: n'] = len(sub['F0:tool:img'])
    for a in ['F0:tool:img', 'F1:tool:img', 'F3:tool:img', 'F4:tool:img', 'F4:tool:none']:
        out[f'{name}: {a}'] = acc_ci(sub[a])
    for a in ['F1:tool:img', 'F3:tool:img', 'F4:tool:img']:
        out[f'{name}: {a} - F0:tool:img'] = paired(sub[a], sub['F0:tool:img'])
json.dump(out, open(os.path.join(OUT, 'e1_d2_extra.json'), 'w'), indent=1)
for k, v in out.items():
    print(k, v)
