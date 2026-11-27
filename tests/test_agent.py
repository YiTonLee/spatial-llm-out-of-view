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


def test_photo_packets_same_facts_every_format():
    from agent import photo_packets as pp
    it = dict(A='cup', B='plate', relation='left of')
    tl = dict(W=640, H=480, A=dict(box=[100, 200, 200, 300], score=0.6, depth=5.0, near=0.8),
              B=dict(box=[400, 250, 600, 400], score=0.5, depth=3.0, near=0.3))
    assert pp.packet('F0', it, tl) == ''
    assert '[100, 200, 200, 300]' in pp.packet('F1', it, tl)
    assert '(0.23, 0.52)' in pp.packet('F2', it, tl)
    assert '0.55 image widths to the left' in pp.packet('F3', it, tl)
    f4 = pp.packet('F4', it, tl)
    assert 'the cup is to the left of the plate' in f4 and 'nearer to the camera' in f4
    assert 'to the right of' in pp.packet('F4', it, tl, negate=True)
    assert pp.k2(it, tl) == 1                                   # the rule says "true"
    miss = dict(tl, B=dict(box=None, score=0.1, depth=None, near=None))
    assert 'did not find the plate' in pp.packet('F3', it, miss) and pp.k2(it, miss) == 1

def test_e6a_scoring_rules():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import e6a
    assert e6a.fmt_ok('Facts: the cup is left of the plate.\nAnswer: Yes')
    assert not e6a.fmt_ok('Yes, the cup is left of the plate.')                 # no Facts/Answer form
    assert not e6a.fmt_ok('Facts: ' + 'word ' * 45 + '\nAnswer: Yes')           # over 40 words
    yn, mc, op = dict(kind='yn', label=1), dict(kind='mc', label=2), dict(kind='open', label='12.5')
    assert e6a.note_answer('d2', yn, 'Facts: x\nAnswer: Yes.') == 1
    assert e6a.correct('d3', op, '12.9') == 1 and e6a.correct('d3', op, '14') == 0   # ChartQA: within 5%
    assert e6a.correct('d1', mc, 2) == 1
    m, ci = e6a.boot({'a': [1, 0], 'b': [1], 'c': [0, 0, 1]}, B=200)
    assert m == 50.0 and ci[0] <= m <= ci[1]
    assert download.select('e6a') == manifest.GROUPS['e6a']
