"""Packet formats for photos (domain 2): the same detector + depth result written different ways (E1 domain 2).

Plain: two fixed tools look at a photo. A detector draws a box around each of the two objects named in the question,
and a depth model says how near each box is. This file turns that one result into the note the small model reads:
nothing (F0), the raw tool output (F1), positions as fractions of the image (F2), worked-out differences (F3), or
plain-words statements (F4: "the cup is to the left of the plate; the cup is nearer to the camera than the plate").
F5 and F6 are picture notes (a depth map, a crop) and carry no text here. The domain-1 version for out-of-view balls
is agent/packets.py; both follow the same rule: one tool result, every format states every fact, only wording changes.

Technical:
  it  dict with the question's object names 'A', 'B' and the VSR 'relation'
  tl  tool record: {'W', 'H', 'A': {box [x1, y1, x2, y2] px, score, depth (median inverse depth), near (0-1)}, 'B': ...}
  An object counts as found when its Grounding DINO score >= THR; otherwise every text format says it was not found.
  REL maps the measurable VSR relations to (fact, value that makes the caption true); k2 is the rule alone (no VLM).
Moved unchanged from pilot/e1_d2.py on 2026-11-26 (checked: identical notes on all 2,195 test items, all formats).
"""
THR = 0.25

# measurable relations -> (fact, value that makes the caption true)
REL = {}
for r in ('left of', 'at the left side of'):
    REL[r] = ('lr', 'left')
for r in ('right of', 'at the right side of'):
    REL[r] = ('lr', 'right')
for r in ('above', 'over', 'on top of'):
    REL[r] = ('ud', 'above')
for r in ('below', 'under', 'beneath'):
    REL[r] = ('ud', 'below')
REL['in front of'] = ('depth', 'nearer')
for r in ('behind', 'at the back of'):
    REL[r] = ('depth', 'farther')
for r in ('in', 'inside'):
    REL[r] = ('cont', 'A in B')
for r in ('contains', 'surrounding'):
    REL[r] = ('cont', 'B in A')


def facts(it, tl):
    """None if an object was not found; else dict of the four facts and the numbers behind them."""
    A, B = tl['A'], tl['B']
    if A['score'] < THR or B['score'] < THR:
        return None
    W, H = tl['W'], tl['H']
    ca = ((A['box'][0] + A['box'][2]) / 2 / W, (A['box'][1] + A['box'][3]) / 2 / H)
    cb = ((B['box'][0] + B['box'][2]) / 2 / W, (B['box'][1] + B['box'][3]) / 2 / H)

    def inside(p, q):   # share of p's box area inside q's box
        ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
        return ix * iy / max((p[2] - p[0]) * (p[3] - p[1]), 1e-6)
    ab, ba = inside(A['box'], B['box']), inside(B['box'], A['box'])
    cont = 'A in B' if ab >= 0.8 and ab >= ba else ('B in A' if ba >= 0.8 else 'neither')
    return dict(dx=ca[0] - cb[0], dy=ca[1] - cb[1], dn=A['near'] - B['near'], ab=ab, ba=ba, ca=ca, cb=cb,
                lr='right' if ca[0] > cb[0] else 'left', ud='below' if ca[1] > cb[1] else 'above',
                depth='nearer' if A['near'] > B['near'] else 'farther', cont=cont)


def packet(fmt, it, tl, negate=False):
    a, b = it['A'], it['B']
    if fmt in ('F0', 'F5', 'F6'):
        return ''
    miss = [it[k] for k in ('A', 'B') if tl[k]['score'] < THR]
    if miss:
        return 'Tool result: the object detector did not find the ' + ' or the '.join(miss) + ' in the image. '
    A, B = tl['A'], tl['B']; f = facts(it, tl)
    if fmt == 'F1':
        bx = lambda o: '[' + ', '.join(f'{v:.0f}' for v in o['box']) + ']'
        return (f"Tool output. Object detector (image {tl['W']}x{tl['H']} pixels, boxes [x1, y1, x2, y2]): the {a} {bx(A)} "
                f"score {A['score']:.2f}; the {b} {bx(B)} score {B['score']:.2f}. Depth model (relative inverse depth, "
                f"larger = nearer): the {a} median {A['depth']:.2f}; the {b} median {B['depth']:.2f}. ")
    if fmt == 'F2':
        sz = lambda o: ((o['box'][2] - o['box'][0]) / tl['W'], (o['box'][3] - o['box'][1]) / tl['H'])
        sa, sb = sz(A), sz(B)
        return (f"Tool output, as fractions of the image width and height from the top-left corner: the {a} centre "
                f"({f['ca'][0]:.2f}, {f['ca'][1]:.2f}), size {sa[0]:.2f} x {sa[1]:.2f}; the {b} centre ({f['cb'][0]:.2f}, "
                f"{f['cb'][1]:.2f}), size {sb[0]:.2f} x {sb[1]:.2f}. Nearness to the camera (0 = farthest in the image, "
                f"1 = nearest): the {a} {A['near']:.2f}, the {b} {B['near']:.2f}. ")
    if fmt == 'F3':
        return (f"Measured relations: the {a}'s centre is {abs(f['dx']):.2f} image widths to the "
                f"{'right' if f['dx'] > 0 else 'left'} of the {b}'s and {abs(f['dy']):.2f} image heights "
                f"{'below' if f['dy'] > 0 else 'above'} it; the {a} is {abs(f['dn']):.2f} {'nearer to' if f['dn'] > 0 else 'farther from'} "
                f"the camera than the {b} (0-1 nearness scale); {100 * f['ab']:.0f}% of the {a}'s box lies inside the "
                f"{b}'s box, and {100 * f['ba']:.0f}% of the {b}'s box lies inside the {a}'s. ")
    # F4
    lr, ud, dp, ct = f['lr'], f['ud'], f['depth'], f['cont']
    if negate:
        lr = {'left': 'right', 'right': 'left'}[lr]; ud = {'above': 'below', 'below': 'above'}[ud]
        dp = {'nearer': 'farther', 'farther': 'nearer'}[dp]; ct = {'A in B': 'B in A', 'B in A': 'A in B', 'neither': 'neither'}[ct]
    cs = {'A in B': f"the {a}'s box is inside the {b}'s box", 'B in A': f"the {b}'s box is inside the {a}'s box",
          'neither': 'neither box is inside the other'}[ct]
    return (f"Measured relations: the {a} is to the {lr} of the {b}; the {a} is {ud} the {b}; the {a} is "
            f"{'nearer to' if dp == 'nearer' else 'farther from'} the camera than the {b}; {cs}. ")


def k2(it, tl):
    """The rule alone: answer a measurable item from the F4 facts (not found -> 'true', the train majority)."""
    f = facts(it, tl)
    if f is None:
        return 1
    key, want = REL[it['relation']]
    return int(f[key] == want)
