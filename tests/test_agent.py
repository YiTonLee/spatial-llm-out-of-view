"""Fast CPU checks for agent/ (about 1 s): packets, the cost frontier, and the dataset manifest."""
import json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from agent import cost, packets  # noqa: E402
from agent.data import download, manifest  # noqa: E402


def test_packets_same_facts_every_format():
    r = np.array([[-1.0, 3.0], [2.0, 5.0]], dtype=np.float32)   # red left of green and nearer
    names = ('red', 'green')
    assert packets.render('F0', names, r) == ''
    assert '(-1.00, +3.00)' in packets.render('F2', names, r)
    f3 = packets.render('F3', names, r)
    assert '3.00 m to the left of the green ball' in f3 and 'the red ball 3.16 m' in f3
    f4 = packets.render('F4', names, r)
    assert 'red ball is to the left of the green ball' in f4 and 'the red ball is closer' in f4
    neg = packets.render('F4', names, r, negate=True)
    assert 'to the right of' in neg and 'the green ball is closer' in neg


def test_frontier_matches_e0():
    # E0 one-shot costs (ms) and both-correct %: only the calculator K is on the frontier
    c = {'V': 188, 'D': 194, 'R': 165, 'K': 90}
    a = {'V': 5.5, 'D': 67.5, 'R': 72.9, 'K': 74.5}
    assert cost.frontier(c, a) == ['K']
    assert cost.total(['decode', 'vit8', 'pre'], {'decode': 2, 'vit8': 86, 'pre': 66}, 'B') == 66


def test_manifest_is_consistent():
    sums = json.load(open(REPO / 'agent' / 'data' / 'checksums.json'))
    for name, d in manifest.DATASETS.items():
        assert d['kind'] in ('url', 'gdrive', 'hf', 'hf_model', 'manual'), name
        assert d['licence'] and d['used_by'], name
        if d['kind'] == 'manual':
            assert d['how'] and d['files'], name
        if d['kind'] in ('hf', 'hf_model'):
            assert len(d.get('rev', '')) == 40, f'{name}: pin the Hugging Face revision'
        if d['kind'] in ('url', 'gdrive', 'manual'):
            assert download.expected(name), f'{name}: no checksums recorded'
    for g, names in manifest.GROUPS.items():
        assert set(names) <= set(manifest.DATASETS), g
    assert all('/' in k and len(v) == 2 for k, v in sums.items())
    assert download.select('e1_d2') == manifest.GROUPS['e1_d2']
