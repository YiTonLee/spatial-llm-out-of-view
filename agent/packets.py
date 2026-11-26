"""Packet formats: the same facts written different ways (E1).

Plain: a tool has worked out where two objects are. This file writes that result as a short note for the small model,
in one of several styles: nothing at all, the positions as numbers, the left-right gap and distances already worked
out, or a plain-words statement ("red is left of green, green is closer"). E1 found that the small model can use only
the worked-out styles (F3, F4); a raw list of sightings (F1) stays useless even after training.

Technical: every renderer takes the same input, so only the format changes:
  names  (A, B): object names as they appear in the question
  r      array-like (2, 2): [[xA, zA], [xB, zB]] in the camera's own axes at the asked time
         (x = metres to the right, z = metres forward; the camera is at the origin)
F1 (raw sightings) needs the whole track and is written by sts.track_text; it is listed here for completeness.
`negate=True` flips both facts of F4 (the "negated note" control); distance is Euclidean in the x-z plane. Pass r as a
numpy array to keep its precision (E1 used float32; a float64 copy changes the last printed digit in ~0.1% of notes).
"""
import numpy as np

FORMATS = {
    'F0': 'no packet',
    'F1': 'raw sightings: every visible (t, x, z) per object + camera motion (sts.track_text)',
    'F2': 'positions at the asked time, camera axes',
    'F3': 'derived: signed left-right offset + each distance to the camera',
    'F4': 'predicates: left/right and closer/farther',
}


def f2_positions(names, r):
    A, B = names
    return ("Computed positions at the asked time, in the camera's own axes at that time (x = metres to the right, "
            f'z = metres forward): the {A} ball at ({r[0][0]:+.2f}, {r[0][1]:+.2f}), the {B} ball at '
            f'({r[1][0]:+.2f}, {r[1][1]:+.2f}). ')


def _facts(r):
    # numpy keeps the caller's precision (E1 passed float32 arrays), so the printed digits match the logged run
    r = np.asarray(r)
    dx = r[0, 0] - r[1, 0]
    return dx, np.linalg.norm(r[0]), np.linalg.norm(r[1])


def f3_derived(names, r):
    A, B = names
    dx, dA, dB = _facts(r)
    return (f'At the asked time, seen from the camera: the {A} ball is {abs(dx):.2f} m to the '
            f'{"right" if dx > 0 else "left"} of the {B} ball; distance from the camera: the {A} ball {dA:.2f} m, '
            f'the {B} ball {dB:.2f} m. ')


def f4_predicates(names, r, negate=False):
    A, B = names
    dx, dA, dB = _facts(r)
    right, a_closer = dx > 0, dA < dB
    if negate:
        right, a_closer = not right, not a_closer
    near, far = (A, B) if a_closer else (B, A)
    return (f'At the asked time, seen from the camera: the {A} ball is to the {"right" if right else "left"} of the '
            f'{B} ball, and the {near} ball is closer to the camera than the {far} ball. ')


def render(fmt, names, r=None, raw_text='', negate=False):
    """The packet text for format fmt. raw_text is the F1 text (from sts.track_text); r is needed for F2-F4."""
    if fmt == 'F0':
        return ''
    if fmt == 'F1':
        return raw_text
    if fmt == 'F2':
        return f2_positions(names, r)
    if fmt == 'F3':
        return f3_derived(names, r)
    if fmt == 'F4':
        return f4_predicates(names, r, negate)
    raise ValueError(f'unknown packet format {fmt}; choose from {list(FORMATS)}')
