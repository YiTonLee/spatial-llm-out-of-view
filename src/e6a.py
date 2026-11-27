"""E6a: helper-size ladder, inference only (gates: docs/gates/e6a_gates.md, written before any run).

Plain: a small model (the student, 3B) answers questions about videos, photos, charts and small details in photos. A
helper model (the student itself or a 7B) looks first and writes a short note. (A 30B with about 3B active was
planned; it is deferred to future work because it wrote about 2.2 s per token on the laptop, see the gates.) We measure
how many questions the student gets right with each helper's note, how long it all takes, and compare that with the
student alone and with each helper answering on its own.

Technical:
  domains d1 Perception Test train held-back slice (permanence/occlusion/containment/change detection, 3-way MC),
          d2 VSR dev measurable (yes/no), d3 ChartQA val human (open, relaxed accuracy), d4 Visual CoT GQA held-back
          slice with answer box < 5% of the image (open, exact match). 100 calibration + 300 scoring items, seed 6.
  models  s3b Qwen2.5-VL-3B (student and helper "self"), h7b Qwen2.5-VL-7B bf16; h30b Qwen3-VL-30B-A3B int4
          AutoRound is still loadable (needs auto-round >= 0.9.6 and the model in <data>/models/) but not in the run
  scoring closed answers by the first-token logits of the options (Yes/No, A/B/C); open answers by greedy decoding
          (<= 16 tokens). Confidence: softmax probability of the chosen option, or exp(mean log-prob) of the answer.
  cost    CUDA-synchronised wall time of each model call (processor + forward/generate), media loading excluded.
Commands (outputs in <work>/e6a/<domain>/):
  python e6a.py items                                  -> items_<d>.json for all domains (d1 also extracts frames)
  python e6a.py notes   --domain d2 --model h7b [--n 0]  -> notes_h7b.json (note, helper alone, times, peak memory)
  python e6a.py student --domain d2 [--n 0]             -> student.json (alone + with every notes file present)
  python e6a.py check   --domain d2                     -> S0 format and S2 time on the calibration slice
  python e6a.py report                                  -> E6A_table.md, e6a_summary.json (tau from calibration)
"""
import argparse, io, json, os, re, string, sys, time, zipfile
import numpy as np, torch
from PIL import Image

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.environ.get('SPATIAL_LLM_DATA') or os.path.join(ROOT, 'data', 'public')
OUT = os.path.join(HERE, 'e6a')
MODELS = {'s3b': 'Qwen/Qwen2.5-VL-3B-Instruct', 'h7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
          'h30b': os.path.join(DATA, 'models', 'Qwen3-VL-30B-A3B-Instruct-int4-AutoRound')}
HELPERS = ['s3b', 'h7b', 'h30b']
DOMAINS = ['d1', 'd2', 'd3', 'd4']
N_CAL, N_SCORE, SEED = 100, 300, 6
PIX = {'d1': None, 'd2': 448 * 448, 'd3': 1024 * 28 * 28, 'd4': 1024 * 28 * 28}   # max pixels per image
N_FRAMES, FRAME_LONG = 16, 448
MEDIA = {'d1': 'video', 'd2': 'photo', 'd3': 'chart', 'd4': 'photo'}
HINT = {'d1': 'where each object was last seen, what moved or changed, and where the objects are now',
        'd2': 'the relation between the two objects as you see it (left or right, above or below, nearer or farther, inside or not)',
        'd3': 'the values you read from the chart and any calculation needed',
        'd4': 'what the small region that answers the question shows (text, colour, object)'}
TAGS = {'object permanence', 'occlusion', 'containment', 'change detection'}


def ddir(d):
    p = os.path.join(OUT, d); os.makedirs(p, exist_ok=True); return p


# ------------------------------------------------------------------------------------------------ items
def items_d1(rng):
    d = json.loads(zipfile.ZipFile(os.path.join(DATA, 'perception_test', 'mc_question_train_annotations.zip')).read('mc_question_train.json'))
    qs = [dict(id=f"{vid}:{q['id']}", group=vid, kind='mc', question=q['question'], options=q['options'], label=int(q['answer_id']),
               video=vid, n_frames=d[vid]['metadata']['num_frames'])
          for vid in sorted(d) for q in d[vid]['mc_question'] if TAGS & set(q['tag'])]
    return [qs[i] for i in rng.permutation(len(qs))[:N_CAL + N_SCORE]]


def items_d2(rng):
    import e1_d2
    its = e1_d2.items('dev')
    idx = [i for i, it in enumerate(its) if it['measurable']]
    out = []
    for i in rng.permutation(idx)[:N_CAL + N_SCORE]:
        it = its[int(i)]
        out.append(dict(id=f'vsr_dev:{int(i)}', group=it['image'], kind='yn', question=it['caption'], label=it['label'],
                        dev_index=int(i), image=it['image'], folder=it['folder'], relation=it['relation'], A=it['A'], B=it['B']))
    return out


def items_d3(rng):
    import pandas as pd
    df = pd.read_parquet(os.path.join(DATA, 'chartqa', 'data', 'val-00000-of-00001.parquet'), columns=['imgname', 'query', 'label', 'type'])
    df = df[df['type'] == 'human'].reset_index(drop=True)
    return [dict(id=f'chartqa_val:{int(i)}', group=df.imgname[i], kind='open', question=df['query'][i], label=str(df.label[i]),
                 imgname=df.imgname[i], row=int(i)) for i in rng.permutation(len(df))[:N_CAL + N_SCORE]]


def items_d4(rng):
    rows = []
    for line in open(os.path.join(DATA, 'visual_cot', 'metadata', 'gqa_cot_train.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        x1, y1, x2, y2 = r['bboxs'][0]
        if (x2 - x1) * (y2 - y1) < 0.05 * r['width'] * r['height']:
            rows.append(r)
    out, seen = [], set()
    for i in rng.permutation(len(rows)):
        r = rows[int(i)]
        if r['image'] in seen:   # one question per image, so bootstrap groups are items
            continue
        seen.add(r['image'])
        out.append(dict(id=f"gqa_cot:{r['image']}:{int(i)}", group=r['image'], kind='open', question=r['question'], label=r['answer'],
                        image=r['image'], box=r['bboxs'][0]))
        if len(out) == N_CAL + N_SCORE:
            break
    return out


def frames_d1(vid, n_total):
    """16 frames evenly spaced over the video, cut at the cut frame (cup games), long side 448, cached as npz."""
    p = os.path.join(ddir('d1'), 'frames', vid + '.npz')
    if os.path.exists(p):
        return list(np.load(p)['f'])
    import av
    cut = json.load(open(os.path.join(DATA, 'perception_test', 'cut_frame_mapping_train.json'))).get(vid, -1)
    last = (cut if cut and cut > 0 else n_total) - 1
    want = set(np.linspace(0, last, N_FRAMES).round().astype(int).tolist())
    z = _zip(os.path.join(DATA, 'perception_test', 'train_videos.zip'))
    cont = av.open(io.BytesIO(z.read(f'videos/{vid}.mp4'))); cont.streams.video[0].thread_type = 'AUTO'
    fr = []
    for k, f in enumerate(cont.decode(video=0)):
        if k in want:
            im = f.to_image(); s = FRAME_LONG / max(im.size)
            fr.append(np.asarray(im.resize((round(im.width * s / 28) * 28, round(im.height * s / 28) * 28))))
        if k >= last:
            break
    cont.close()
    while len(fr) < N_FRAMES:   # decoder reported fewer frames than the metadata: repeat the last
        fr.append(fr[-1])
    os.makedirs(os.path.dirname(p), exist_ok=True); np.savez(p, f=np.stack(fr[:N_FRAMES]))
    return fr[:N_FRAMES]


def cmd_items(a):
    for d, fn in (('d1', items_d1), ('d2', items_d2), ('d3', items_d3), ('d4', items_d4)):
        if a.domain and d != a.domain:
            continue
        its = fn(np.random.default_rng(SEED))
        for k, it in enumerate(its):
            it['part'] = 'cal' if k < N_CAL else 'score'
        json.dump(its, open(os.path.join(ddir(d), f'items_{d}.json'), 'w'), indent=0)
        print(d, len(its), 'items', flush=True)
        if d == 'd1':
            t0 = time.time()
            for k, vid in enumerate(dict.fromkeys(it['video'] for it in its)):
                frames_d1(vid, next(it['n_frames'] for it in its if it['video'] == vid))
                if k % 50 == 0:
                    print(f'frames {k} {time.time() - t0:.0f}s', flush=True)


def load_items(d, n=0, part='all'):
    its = json.load(open(os.path.join(OUT, d, f'items_{d}.json')))
    if part == 'cal':   # stop checks S0 and S2 run on the calibration slice only
        its = [it for it in its if it['part'] == 'cal']
    if n:   # quick run: the first n calibration and first n scoring items
        its = [it for it in its if it['part'] == 'cal'][:n] + [it for it in its if it['part'] == 'score'][:n]
    return its


# ------------------------------------------------------------------------------------------------ media
_Z = {}


def _zip(p):
    if p not in _Z:
        _Z[p] = zipfile.ZipFile(p)
    return _Z[p]


_CQ = None


def media(d, it):
    """(images list, video frames or None)."""
    global _CQ
    if d == 'd1':
        return [], frames_d1(it['video'], it['n_frames'])
    if d == 'd2':
        z = _zip(os.path.join(DATA, 'coco2017', it['folder'] + '.zip'))
        im = Image.open(io.BytesIO(z.read(f"{it['folder']}/{it['image']}"))).convert('RGB')
    elif d == 'd3':
        if _CQ is None:
            import pandas as pd
            df = pd.read_parquet(os.path.join(DATA, 'chartqa', 'data', 'val-00000-of-00001.parquet'))
            df = df[df['type'] == 'human'].reset_index(drop=True); _CQ = df
        v = _CQ.image[it['row']]; im = Image.open(io.BytesIO(v['bytes'] if isinstance(v, dict) else v)).convert('RGB')
    else:
        im = Image.open(io.BytesIO(_zip(os.path.join(DATA, 'gqa', 'images.zip')).read('images/' + it['image']))).convert('RGB')
    m = PIX[d]
    if im.width * im.height > m:
        s = (m / (im.width * im.height)) ** 0.5; im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))))
    return [im], None


def chart_table(it):
    z = _zip(os.path.join(DATA, 'chartqa', 'ChartQA Dataset.zip'))
    return z.read(f"ChartQA Dataset/val/tables/{os.path.splitext(it['imgname'])[0]}.csv").decode('utf-8', 'replace').strip()


# ------------------------------------------------------------------------------------------------ prompts
LET = 'ABC'


def question_text(d, it):
    if it['kind'] == 'yn':
        return f"Statement: {it['question']}\nIs the statement true for the photo? Answer Yes or No."
    if it['kind'] == 'mc':
        opts = '\n'.join(f'({LET[k]}) {o}' for k, o in enumerate(it['options']))
        return f"Question: {it['question']}\nOptions:\n{opts}\nAnswer with the letter of the correct option."
    return f"Question: {it['question']}\nAnswer with a single word or number."


def note_prompt(d, it):
    return (f"You are helping a smaller model answer a question about this {MEDIA[d]}. Write a note for it in exactly "
            f"this form, two lines:\nFacts: <at most 30 words of plain statements: {HINT[d]}>\nAnswer: <your answer>\n\n"
            f"The question the smaller model must answer:\n{question_text(d, it)}")


PREFILL = 'Facts:'   # S0 fix, applied to every helper alike
SHARED_PROMPT = ("Describe this video for someone who must later answer questions about it without seeing it. In at most 60 "
                 "words of plain statements, say where each object is hidden or moved, what changes, and where the objects "
                 "are at the end.")


def with_note(note, d, it):
    return ('' if not note else f"Note from a helper: {note.strip()}\n") + question_text(d, it)


def tools_note(it):
    import e1_d2
    from agent.photo_packets import packet
    global _TL
    if '_TL' not in globals():
        _TL = json.load(open(os.path.join(HERE, 'e1d2', 'tools_dev.json')))
    return packet('F4', e1_d2.items('dev')[it['dev_index']], _TL[it['dev_index']])


# ------------------------------------------------------------------------------------------------ model
class VLM:
    def __init__(self, key):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.key = key; path = MODELS[key]
        torch.cuda.reset_peak_memory_stats()
        kw = dict(dtype='auto') if key == 'h30b' else dict(dtype=torch.bfloat16)
        self.model = AutoModelForImageTextToText.from_pretrained(path, device_map='cuda', **kw).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        tok = self.proc.tokenizer
        self.ids = {w: tok.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'No', 'A', 'B', 'C')}
        self.load_mem = torch.cuda.max_memory_allocated() / 2**30

    def enc(self, ims, video, text, prefill=''):
        content = [{'type': 'image'} for _ in ims] + ([{'type': 'video'}] if video is not None else []) + [{'type': 'text', 'text': text}]
        t = self.proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False) + prefill
        kw = dict(text=[t], return_tensors='pt')
        if ims:
            kw['images'] = ims
        if video is not None:
            kw['videos'] = [np.stack(video)]
            kw['videos_kwargs'] = dict(do_sample_frames=False)
        return {k: v.to('cuda') for k, v in self.proc(**kw).items()}

    @torch.no_grad()
    def answer(self, kind, ims, video, text):
        """(prediction, confidence, ms). yn -> 0/1, mc -> option index, open -> string."""
        torch.cuda.synchronize(); t0 = time.perf_counter()
        e = self.enc(ims, video, text)
        if kind in ('yn', 'mc'):
            lg = self.model(**e, logits_to_keep=1).logits[0, -1].float()
            keys = ['No', 'Yes'] if kind == 'yn' else ['A', 'B', 'C']
            p = torch.softmax(lg[[self.ids[k] for k in keys]], 0)
            pred, conf = int(p.argmax()), float(p.max())
        else:
            g = self.model.generate(**e, max_new_tokens=16, do_sample=False, output_scores=True, return_dict_in_generate=True)
            new = g.sequences[0, e['input_ids'].shape[1]:]
            lp = [float(torch.log_softmax(s[0].float(), -1)[t]) for s, t in zip(g.scores, new)]
            pred = self.proc.tokenizer.decode(new, skip_special_tokens=True).strip()
            conf = float(np.exp(np.mean(lp))) if lp else 0.0
        torch.cuda.synchronize()
        return pred, conf, 1000 * (time.perf_counter() - t0)

    @torch.no_grad()
    def write(self, ims, video, text, max_new=80, prefill=''):
        """prefill: text the reply is forced to start with (S0 fix 2026-11-27: 'Facts:' for every helper's note)."""
        torch.cuda.synchronize(); t0 = time.perf_counter()
        e = self.enc(ims, video, text, prefill)
        g = self.model.generate(**e, max_new_tokens=max_new, do_sample=False)
        new = g[0, e['input_ids'].shape[1]:]
        out = (prefill + self.proc.tokenizer.decode(new, skip_special_tokens=True)).strip()
        torch.cuda.synchronize()
        return out, 1000 * (time.perf_counter() - t0), int(len(new)), int(e['input_ids'].shape[1])


# ------------------------------------------------------------------------------------------------ runs
def warm(m, d, it):
    """One discarded call of each kind, so the first timed item does not pay CUDA start-up."""
    ims, vid = media(d, it); m.write(ims, vid, note_prompt(d, it), max_new=8); m.answer(it['kind'], ims, vid, question_text(d, it))


def cmd_notes(a):
    d = a.domain; its = load_items(d, a.n, a.part); m = VLM(a.model); rows = []; shared = {}
    warm(m, d, its[0]); t0 = time.time()
    for k, it in enumerate(its):
        ims, vid = media(d, it)
        note, nms, ntok, intok = m.write(ims, vid, note_prompt(d, it), prefill=PREFILL)
        row = dict(id=it['id'], note=note, note_ms=round(nms, 1), note_tokens=ntok, in_tokens=intok)
        if a.model != 's3b':   # the student's own "alone" answer is in student.json
            p, c, ams = m.answer(it['kind'], ims, vid, question_text(d, it))
            row.update(alone=p, alone_conf=round(c, 4), alone_ms=round(ams, 1))
        if d == 'd1' and it['video'] not in shared:
            s, sms, stok, _ = m.write(ims, vid, SHARED_PROMPT, max_new=110)
            shared[it['video']] = dict(note=s, ms=round(sms, 1), tokens=stok)
        rows.append(row)
        if k % 20 == 0:
            print(f'notes {d} {a.model} {k}/{len(its)} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)
    res = dict(model=a.model, domain=d, n=len(rows), sec=round(time.time() - t0), load_mem_gb=round(m.load_mem, 2),
               peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), rows=rows, shared=shared)
    json.dump(res, open(os.path.join(ddir(d), f'notes_{a.model}{a.tag}.json'), 'w'), indent=0)
    print('done', d, a.model, res['sec'], 's peak', res['peak_mem_gb'], 'GB', flush=True)


def cmd_student(a):
    d = a.domain; its = load_items(d, a.n, a.part); m = VLM('s3b'); warm(m, d, its[0]); t0 = time.time()
    notes = {h: json.load(open(os.path.join(OUT, d, f'notes_{h}{a.tag}.json'))) for h in HELPERS
             if os.path.exists(os.path.join(OUT, d, f'notes_{h}{a.tag}.json'))}
    nmap = {h: {r['id']: r for r in v['rows']} for h, v in notes.items()}
    rows = []
    for k, it in enumerate(its):
        ims, vid = media(d, it)
        arms = {'S': ''}
        for h in notes:
            if it['id'] in nmap[h]:
                arms[f'S+{h}'] = nmap[h][it['id']]['note']
                if d == 'd1':
                    arms[f'S+{h}:shared'] = notes[h]['shared'][it['video']]['note']
        if d == 'd2' and os.path.exists(os.path.join(HERE, 'e1d2', 'tools_dev.json')):   # reference arm: needs run.py e1_d2
            arms['S+tools'] = tools_note(it).strip()
        if d == 'd3':
            arms['S+table'] = 'Data table of the chart:\n' + chart_table(it)
        row = dict(id=it['id'])
        for arm, note in arms.items():
            txt = question_text(d, it) if arm == 'S' else (note + '\n' + question_text(d, it) if arm in ('S+tools', 'S+table')
                                                         else with_note(note, d, it))
            p, c, ms = m.answer(it['kind'], ims, vid, txt)
            row[arm] = dict(pred=p, conf=round(c, 4), ms=round(ms, 1))
        rows.append(row)
        if k % 20 == 0:
            print(f'student {d} {k}/{len(its)} {time.time() - t0:.0f}s', flush=True)
    res = dict(domain=d, n=len(rows), sec=round(time.time() - t0), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), rows=rows)
    json.dump(res, open(os.path.join(ddir(d), f'student{a.tag}.json'), 'w'), indent=0)
    print('done student', d, res['sec'], 's', flush=True)


# ------------------------------------------------------------------------------------------------ scoring
def _norm(s):
    s = str(s).lower().strip()
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in ('a', 'an', 'the'))


def _num(s):
    try:
        return float(str(s).replace(',', '').replace('%', '').strip())
    except ValueError:
        return None


def correct(d, it, pred):
    if it['kind'] in ('yn', 'mc'):
        return int(pred == it['label'])
    if d == 'd3':   # ChartQA relaxed accuracy
        p, g = _num(pred), _num(it['label'])
        if p is not None and g is not None:
            return int(abs(p - g) <= 0.05 * abs(g)) if g != 0 else int(p == 0)
        return int(_norm(pred) == _norm(it['label']))
    return int(_norm(pred) == _norm(it['label']))


def note_answer(d, it, note):
    """The answer written at the end of a helper's note, parsed like a model answer."""
    m = re.search(r'answer\s*:\s*(.+)$', note, re.I | re.S)
    if not m:
        return None
    s = m.group(1).strip().strip('.').strip()
    if it['kind'] == 'yn':
        return 1 if s.lower().startswith(('yes', 'true')) else 0 if s.lower().startswith(('no', 'false')) else None
    if it['kind'] == 'mc':
        mm = re.match(r'\(?([ABC])\)?', s)
        if mm:
            return LET.index(mm.group(1))
        for k, o in enumerate(it['options']):
            if _norm(o) and _norm(o) == _norm(s):
                return k
        return None
    return s


def fmt_ok(note):
    body = re.split(r'answer\s*:', note, flags=re.I)
    return (len(note.split()) <= 40 and note.lower().lstrip().startswith('facts:') and len(body) >= 2
            and body[-1].strip() != '')


def cmd_check(a):
    d = a.domain; its = {it['id']: it for it in load_items(d, a.n)}
    out = {}
    for h in HELPERS:
        p = os.path.join(OUT, d, f'notes_{h}{a.tag}.json')
        if not os.path.exists(p):
            continue
        v = json.load(open(p)); cal = [r for r in v['rows'] if its[r['id']]['part'] == 'cal']
        ok = float(np.mean([fmt_ok(r['note']) for r in cal])) if cal else None
        per = float(np.mean([r['note_ms'] + r.get('alone_ms', 0) for r in v['rows']])) / 1000
        out[h] = dict(S0_format_ok=None if ok is None else round(100 * ok, 1), S0_pass=bool(ok is not None and ok >= 0.95),
                      sec_per_item=round(per, 2), projected_min_400=round(per * (N_CAL + N_SCORE) / 60, 1),
                      peak_mem_gb=v['peak_mem_gb'], mean_note_words=round(float(np.mean([len(r['note'].split()) for r in v['rows']])), 1))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ddir(d), f'check{a.tag}.json'), 'w'), indent=1)


def boot(diff_by_group, B=10000, seed=1):
    keys = sorted(diff_by_group); rng = np.random.default_rng(seed)
    vals = [diff_by_group[k] for k in keys]
    sums = np.array([sum(v) for v in vals]); cnts = np.array([len(v) for v in vals])
    m = 100 * sums.sum() / cnts.sum()
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    bs = 100 * sums[idx].sum(1) / cnts[idx].sum(1)
    return round(float(m), 1), [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


def domain_table(d, tag=''):
    """Per-item correctness and time per arm on the scoring slice, tau chosen on the calibration slice."""
    its = {it['id']: it for it in load_items(d)}
    st = json.load(open(os.path.join(OUT, d, f'student{tag}.json')))
    notes = {h: json.load(open(os.path.join(OUT, d, f'notes_{h}{tag}.json'))) for h in HELPERS
             if os.path.exists(os.path.join(OUT, d, f'notes_{h}{tag}.json'))}
    ck = os.path.join(OUT, d, 'check_cal.json')   # S0: a helper failing the format check in this domain is dropped
    if not os.path.exists(ck):   # run.py e6a: check on the full notes files (it reads their calibration rows only)
        ck = os.path.join(OUT, d, f'check{tag}.json')
    dropped = [h for h, v in (json.load(open(ck)).items() if os.path.exists(ck) else []) if not v['S0_pass']]
    notes = {h: v for h, v in notes.items() if h not in dropped}
    nmap = {h: {r['id']: r for r in v['rows']} for h, v in notes.items()}
    srows = {r['id']: r for r in st['rows']}
    q_per_video = {}
    for r in st['rows']:
        if d == 'd1':
            v = its[r['id']]['video']; q_per_video[(its[r['id']]['part'], v)] = q_per_video.get((its[r['id']]['part'], v), 0) + 1

    def arm_rows(part, tau=None):
        A = {}
        ids = [i for i in srows if its[i]['part'] == part]
        for i in ids:
            it, r = its[i], srows[i]
            A.setdefault('S', []).append((i, correct(d, it, r['S']['pred']), r['S']['ms']))
            for h in notes:
                n = nmap[h][i]; key = f'S+{h}'
                cost_note = n['note_ms'] + r[key]['ms']
                A.setdefault(f'{h} alone', []).append((i, correct(d, it, n['alone']) if h != 's3b' else correct(d, it, r['S']['pred']),
                                                       n['alone_ms'] if h != 's3b' else r['S']['ms']))
                A.setdefault(f'S+{h} always', []).append((i, correct(d, it, r[key]['pred']), cost_note))
                if tau is not None:
                    call = r['S']['conf'] < tau[h]
                    A.setdefault(f'S+{h} routed', []).append((i, correct(d, it, r[key]['pred'] if call else r['S']['pred']),
                                                              r['S']['ms'] + (cost_note if call else 0), call))
                if d == 'd1':
                    sh = notes[h]['shared'][it['video']]
                    A.setdefault(f'S+{h} shared/video', []).append((i, correct(d, it, r[f'{key}:shared']['pred']),
                                                                    sh['ms'] / q_per_video[(part, it['video'])] + r[f'{key}:shared']['ms']))
                na = note_answer(d, it, n['note'])
                A.setdefault(f'{h} note answer', []).append((i, int(na is not None and correct(d, it, na)), None))
            for ref in ('S+tools', 'S+table'):
                if ref in r:
                    A.setdefault(ref + ' (reference)', []).append((i, correct(d, it, r[ref]['pred']), r[ref]['ms']))
        A = {k: v for k, v in A.items() if not (k == 's3b alone')}
        return A

    cal = arm_rows('cal')
    tau = {}
    for h in notes:   # tau: best calibration accuracy with the helper called on <= 50% of items
        S = {i: c for i, c, _ in cal['S']}; H = {i: c for i, c, _ in cal[f'S+{h} always']}
        conf = {i: srows[i]['S']['conf'] for i in S}
        best = (-1, 0.0)
        for t in sorted(set(conf.values())) + [1.01]:
            called = [i for i in S if conf[i] < t]
            if len(called) > 0.5 * len(S):
                continue
            acc = np.mean([H[i] if conf[i] < t else S[i] for i in S])
            if acc > best[0] + 1e-12:
                best = (acc, t)
        tau[h] = best[1]
    return arm_rows('score', tau), tau, its, notes, st, dropped


def cmd_report(a):
    L = ['# E6a results (scoring slices; tau from the calibration slices)', '']
    summ = {}; frontier_domains = []
    for d in DOMAINS:
        if not os.path.exists(os.path.join(OUT, d, f'student{a.tag}.json')):
            continue
        A, tau, its, notes, st, dropped = domain_table(d, a.tag)
        grp = lambda i: its[i]['group']
        acc = {k: 100 * np.mean([x[1] for x in v]) for k, v in A.items()}
        cost = {k: (np.mean([x[2] for x in v]) / 1000 if v[0][2] is not None else None) for k, v in A.items()}
        L += [f'## {d} ({MEDIA[d]}), n = {len(A["S"])}; tau = {tau}; dropped by S0: {dropped or "none"}', '',
              '| arm | accuracy % | s per question | call rate % |', '|---|---|---|---|']
        for k in A:
            cr = f"{100 * np.mean([x[3] for x in A[k]]):.0f}" if k.endswith('routed') else ''
            L.append(f"| {k} | {acc[k]:.1f} | {'' if cost[k] is None else f'{cost[k]:.2f}'} | {cr} |")
        base = ['S'] + [f'{h} alone' for h in notes if h != 's3b']
        uses = [k for k in A if k.endswith(('always', 'routed'))]
        L += ['', '| use U | baseline B (no more time than U) | U - B [95% CI] |', '|---|---|---|']
        on = []
        for u in uses:
            ok = True
            for b in base:
                if cost[b] <= cost[u]:
                    diff = {}
                    cb = {x[0]: x[1] for x in A[b]}
                    for x in A[u]:
                        diff.setdefault(grp(x[0]), []).append(x[1] - cb[x[0]])
                    m, ci = boot(diff)
                    L.append(f'| {u} | {b} | {m} [{ci[0]}, {ci[1]}] |')
                    ok = ok and ci[0] > 0
            if ok:
                on.append(u)
        L += ['', f'On the frontier (G6a): {on or "none"}', '']
        if on:
            frontier_domains.append(d)
        pred = {}
        for h in notes:   # G6a-R: does the note help only where the helper alone beats the student alone?
            if h == 's3b':
                continue
            gain = acc[f'S+{h} always'] - acc['S']; edge = acc[f'{h} alone'] - acc['S']
            pred[h] = dict(gain=round(gain, 1), helper_minus_student=round(edge, 1), agree=bool((gain > 0) == (edge > 0)))
        summ[d] = dict(acc={k: round(v, 1) for k, v in acc.items()}, sec={k: (None if v is None else round(v, 3)) for k, v in cost.items()},
                       tau=tau, frontier=on, G6a_R=pred, dropped_S0=dropped,
                       peak_mem_gb={h: v['peak_mem_gb'] for h, v in notes.items()} | {'student': st['peak_mem_gb']})
    summ['G6a_pass'] = len(frontier_domains) >= 2; summ['frontier_domains'] = frontier_domains
    L += [f'**G6a: {"PASS" if summ["G6a_pass"] else "MISS"}** (frontier in {frontier_domains}; needs >= 2 of 4 domains)']
    open(os.path.join(OUT, f'E6A_table{a.tag}.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump(summ, open(os.path.join(OUT, f'e6a_summary{a.tag}.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['items', 'notes', 'student', 'check', 'report'])
    ap.add_argument('--domain', default=''); ap.add_argument('--model', default='h7b', choices=list(MODELS))
    ap.add_argument('--n', type=int, default=0); ap.add_argument('--tag', default='')
    ap.add_argument('--part', default='all', choices=['all', 'cal'])
    a = ap.parse_args()
    if a.n and not a.tag:
        a.tag = f'_n{a.n}'   # quick runs never overwrite the real outputs
    if a.part == 'cal' and not a.tag:
        a.tag = '_cal'
    {'items': cmd_items, 'notes': cmd_notes, 'student': cmd_student, 'check': cmd_check, 'report': cmd_report}[a.cmd](a)
