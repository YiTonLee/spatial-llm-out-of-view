"""E1, domain 2: real photos (VSR). Same tool facts, seven formats (gates: docs/gates/e1_d2_gates.md, written before running).
Results 2026-11-26: G1 and G2 MISS (the tools, 70.9%, are less accurate than the photo alone, 81.1%); see
docs/07_EVIDENCE_PIPELINE.md. Note formats: agent/photo_packets.py.

Plain: the small model sees a photo and a sentence ("The cat is under the table") and says whether it is true. Two
fixed tools find the two objects and measure how near each is. We hand the model that result written in different
ways, with and without the photo, before and after a short training run, and with deliberately wrong notes.

Technical:
  tools   Grounding DINO tiny (best box per name, score >= 0.25 = found) and Depth Anything V2 Small (relative
          inverse depth; median inside each box; nearness = (median - p5) / (p95 - p5) of the image's depth map)
  formats F0 none, F1 raw tool output, F2 normalised positions, F3 derived, F4 predicates, F5 depth-map picture
          (second image), F6 crop of the union box (second image). F1-F4 state ALL facts, not only the asked one.
  arms    "<fmt>:<quality>:<view>", quality tool|shuf|neg, view img|none
  student Qwen2.5-VL-3B (sts.build, LoRA r16 on attention), photos at <= 448x448 pixels (<= 256 tokens);
          answer = logit("Yes") > logit("No") at the first answer position
Commands (python run.py e1_d2 runs the whole queue; data: python -m agent.data.download --only e1_d2):
  python e1_d2.py tools  [--split dev,test,train] [--n 0]     -> e1d2/tools_<split>.json (+ e1d2/f5/<split>_<i>.png)
  python e1_d2.py check                                     -> C0 tool agreement on dev (stop rule)
  python e1_d2.py eval --arms F0:tool:img,F4:tool:img [--lora e1d2/lora_F4.pt] --out e1d2/eval_x.json [--n 0]
  python e1_d2.py train --fmt F4 --save e1d2/lora_F4.pt [--steps 1000]
  python e1_d2.py report
"""
import argparse, io, json, os, sys, time, zipfile
import numpy as np, torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = os.environ.get('SPATIAL_LLM_WORK') or os.path.dirname(os.path.abspath(__file__))   # repo: outputs go to the workspace
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                         # repo root, for agent/ and data/
sys.path.insert(0, ROOT)
DATA = os.environ.get('SPATIAL_LLM_DATA') or os.path.join(ROOT, 'data', 'public')         # python -m agent.data.download --only e1_d2
from agent.photo_packets import THR, REL, facts, packet, k2  # noqa: E402
OUT = os.path.join(HERE, 'e1d2')
GDINO = os.path.join(DATA, 'models', 'grounding-dino-tiny')
DEPTH = os.path.join(DATA, 'models', 'Depth-Anything-V2-Small-hf')
MAX_PIX = 448 * 448


# ------------------------------------------------------------------------------------------------ data
def items(split):
    out = []
    for line in open(os.path.join(DATA, 'vsr', f'{split}.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        cap, rel = r['caption'], r['relation']
        body = cap[len('The '):].rstrip('.')
        a, b = body.split(' ' + rel + ' ', 1)
        b = b[4:] if b.startswith('the ') else b
        a = a.strip()
        for v in (' is', ' are'):
            if a.endswith(v):
                a = a[:-len(v)]
        out.append(dict(image=r['image'], folder=r['image_link'].split('/')[-2], caption=cap, relation=rel,
                        A=a.strip(), B=b.strip(), label=int(r['label']), measurable=rel in REL))
    return out


_ZIPS = {}


def photo(it):
    z = _ZIPS.get(it['folder'])
    if z is None:
        z = _ZIPS[it['folder']] = zipfile.ZipFile(os.path.join(DATA, 'coco2017', it['folder'] + '.zip'))
    return Image.open(io.BytesIO(z.read(f"{it['folder']}/{it['image']}"))).convert('RGB')


# ------------------------------------------------------------------------------------------------ tools
def cmd_tools(a):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForDepthEstimation
    gp = AutoProcessor.from_pretrained(GDINO); gm = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).cuda().eval()
    dp = AutoProcessor.from_pretrained(DEPTH); dm = AutoModelForDepthEstimation.from_pretrained(DEPTH).cuda().eval()
    os.makedirs(os.path.join(OUT, 'f5'), exist_ok=True)
    for split in a.split.split(','):
        its = items(split)[:a.n or None]; res = []; t0 = time.time()
        for i, it in enumerate(its):
            if i % 200 == 0:
                print(f'tools {split} {i}/{len(its)} {time.time() - t0:.0f}s', flush=True)
            im = photo(it); W, H = im.size
            with torch.no_grad():
                dep = dm(**{k: v.cuda() for k, v in dp(images=im, return_tensors='pt').items()}).predicted_depth
                dep = F.interpolate(dep[:, None].float(), size=(H, W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                p5, p95 = np.percentile(dep, 5), np.percentile(dep, 95)
                objs = {}
                for key in ('A', 'B'):
                    enc = gp(images=im, text=f"{it[key]}.", return_tensors='pt').to('cuda')
                    out = gm(**enc)
                    r = gp.post_process_grounded_object_detection(out, enc['input_ids'], threshold=0.05, text_threshold=0.05,
                                                                  target_sizes=[(H, W)])[0]
                    if len(r['scores']):
                        j = int(r['scores'].argmax()); box = [float(v) for v in r['boxes'][j]]
                        x1, y1, x2, y2 = [int(round(v)) for v in box]
                        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, max(x2, x1 + 1)), min(H, max(y2, y1 + 1))
                        med = float(np.median(dep[y1:y2, x1:x2]))
                        objs[key] = dict(box=[round(v, 1) for v in box], score=round(float(r['scores'][j]), 3), depth=round(med, 3),
                                         near=round(float(np.clip((med - p5) / max(p95 - p5, 1e-6), 0, 1)), 3))
                    else:
                        objs[key] = dict(box=None, score=0.0, depth=None, near=None)
            rec = dict(W=W, H=H, **objs)
            if split == 'test':   # F5 picture: depth map (bright = near) with both found boxes drawn and labelled
                g = np.clip((dep - p5) / max(p95 - p5, 1e-6), 0, 1)
                pic = Image.fromarray((g * 255).astype(np.uint8)).convert('RGB'); d = ImageDraw.Draw(pic)
                for key, col in (('A', (255, 60, 60)), ('B', (60, 160, 255))):
                    o = objs[key]
                    if o['score'] >= THR:
                        d.rectangle(o['box'], outline=col, width=max(2, W // 150)); d.text((o['box'][0] + 4, o['box'][1] + 4), key, fill=col)
                pic.thumbnail((448, 448)); pic.save(os.path.join(OUT, 'f5', f'{split}_{i}.png'))
            res.append(rec)
        json.dump(res, open(os.path.join(OUT, f'tools_{split}.json'), 'w'))
        print(f'tools {split} done {len(res)} items {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------ facts and packets
def second_image(fmt, split, i, it, tl, im):
    if fmt == 'F5':
        return Image.open(os.path.join(OUT, 'f5', f'{split}_{i}.png')).convert('RGB')
    if fmt == 'F6':
        bs = [tl[k]['box'] for k in ('A', 'B') if tl[k]['score'] >= THR]
        if not bs:
            return im
        x1 = min(b[0] for b in bs); y1 = min(b[1] for b in bs); x2 = max(b[2] for b in bs); y2 = max(b[3] for b in bs)
        pad = 0.1 * max(x2 - x1, y2 - y1)
        c = im.crop((max(0, x1 - pad), max(0, y1 - pad), min(im.width, x2 + pad), min(im.height, y2 + pad)))
        return c.resize((448, int(448 * c.height / max(c.width, 1))) if c.width >= c.height else (int(448 * c.width / max(c.height, 1)), 448))
    return None


EXTRA = {'F5': 'The second image is a depth map of the photo (brighter = nearer to the camera); box A marks the {A}, box B marks the {B}. ',
         'F6': 'The second image is an enlarged crop of the photo around the {A} and the {B}. '}


def prompt(text_packet, it):
    return (text_packet + f"Statement: {it['caption']}\nIs the statement true for the photo? Answer Yes or No.")


# ------------------------------------------------------------------------------------------------ student
def build():
    import sts, mve_nowrope as M
    from transformers import AutoProcessor
    _, model, params, _ = sts.build(0, None)
    proc = AutoProcessor.from_pretrained(M.MODEL, min_pixels=224 * 224, max_pixels=MAX_PIX)
    proc.tokenizer.padding_side = 'left'
    yes, no = (proc.tokenizer.encode(w, add_special_tokens=False)[0] for w in ('Yes', 'No'))
    return proc, model, params, yes, no


def encode(proc, batch):
    """batch: list of (images list (0-2), text)."""
    texts, imgs = [], []
    for ims, txt in batch:
        content = [{'type': 'image'} for _ in ims] + [{'type': 'text', 'text': txt}]
        texts.append(proc.apply_chat_template([{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False))
        imgs += ims
    enc = proc(text=texts, images=imgs or None, return_tensors='pt', padding=True)
    return {k: v.to('cuda') for k, v in enc.items()}


def make_input(arm, split, i, it, tl, tools_all, im):
    fmt, qual, view = arm.split(':')
    if qual == 'shuf':
        j = (i + 101) % len(tools_all); src = dict(it, A=it['A'], B=it['B']); tsrc = tools_all[j]
        txt = packet(fmt, src, tsrc)
    else:
        txt = packet(fmt, it, tl, negate=qual == 'neg')
    ims = [im] if view == 'img' else []
    if fmt in EXTRA:
        ims = ims + [second_image(fmt, split, i, it, tl, im)]; txt = EXTRA[fmt].format(A=it['A'], B=it['B'])
    return ims, prompt(txt, it)


def cmd_eval(a):
    proc, model, params, yes, no = build()
    import sts
    if a.lora:
        sts.swap_lora(params, [v.to('cuda') for v in torch.load(a.lora)])
    model.eval()
    split = a.split; its = items(split); tools_all = json.load(open(os.path.join(OUT, f'tools_{split}.json')))
    n = a.n or len(tools_all); arms = a.arms.split(','); rows = []; t0 = time.time(); BS = 8
    with torch.no_grad():
        for s in range(0, n, BS):
            if s % 200 < BS:
                print(f'{os.path.basename(a.out)} {s}/{n} {time.time() - t0:.0f}s', flush=True)
            idx = list(range(s, min(n, s + BS))); ims = {i: photo(its[i]) for i in idx}
            for arm in arms:
                batch = [make_input(arm, split, i, its[i], tools_all[i], tools_all, ims[i]) for i in idx]
                lg = model(**encode(proc, batch), logits_to_keep=1).logits[:, -1].float()
                for k, i in enumerate(idx):
                    it = its[i]; pred = int(lg[k, yes] > lg[k, no])
                    rows.append(dict(arm=arm, i=i, image=it['image'], relation=it['relation'], measurable=it['measurable'],
                                     label=it['label'], pred=pred, margin=round(float(lg[k, yes] - lg[k, no]), 3),
                                     found=facts(it, tools_all[i]) is not None))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(lora=a.lora, split=split, arms=arms, n=n, sec=round(time.time() - t0), rows=rows), open(a.out, 'w'))
    print(json.dumps(summarize(rows), indent=1))


def cmd_train(a):
    proc, model, params, yes, no = build()
    tl = json.load(open(os.path.join(OUT, 'tools_train.json'))); its = items('train')[:len(tl)]
    rng = np.random.default_rng(1); torch.manual_seed(1)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    model.train(); BS = 4; t0 = time.time(); la = []
    order = rng.permutation(len(its))
    for step in range(a.steps):
        idx = [int(order[(step * BS + k) % len(its)]) for k in range(BS)]
        batch = []
        for i in idx:
            txt = '' if rng.random() < a.drop else packet(a.fmt, its[i], tl[i])   # 20% no note: a seen input
            batch.append(([photo(its[i])], prompt(txt, its[i])))
        tgt = torch.tensor([yes if its[i]['label'] else no for i in idx], device='cuda')
        loss = F.cross_entropy(model(**encode(proc, batch), logits_to_keep=1).logits[:, -1].float(), tgt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        la.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f'train {a.fmt} {step + 1}/{a.steps} loss {np.mean(la[-100:]):.3f} {time.time() - t0:.0f}s', flush=True)
    torch.save([p.detach().cpu() for p in params], a.save)
    print('saved', a.save, f'{time.time() - t0:.0f}s')


# ------------------------------------------------------------------------------------------------ scoring
def acc_ci(rows, B=1000, seed=0):
    by = {}
    for r in rows:
        by.setdefault(r['image'], []).append(int(r['pred'] == r['label']))
    keys = sorted(by); rng = np.random.default_rng(seed)
    m = 100 * np.mean([v for k in keys for v in by[k]])
    bs = [100 * np.mean([v for k in rng.choice(keys, len(keys)) for v in by[k]]) for _ in range(B)]
    return round(float(m), 1), [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


def paired(ra, rb, B=1000):
    """accuracy(a) - accuracy(b) on the same items, bootstrap over images."""
    ka = {r['i']: r for r in ra}; kb = {r['i']: r for r in rb}; common = sorted(set(ka) & set(kb))
    by = {}
    for i in common:
        by.setdefault(ka[i]['image'], []).append(int(ka[i]['pred'] == ka[i]['label']) - int(kb[i]['pred'] == kb[i]['label']))
    keys = sorted(by); rng = np.random.default_rng(1)
    m = 100 * np.mean([v for k in keys for v in by[k]])
    bs = [100 * np.mean([v for k in rng.choice(keys, len(keys)) for v in by[k]]) for _ in range(B)]
    return round(float(m), 1), [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]


def summarize(rows):
    out = {}
    for arm in dict.fromkeys(r['arm'] for r in rows):
        rs = [r for r in rows if r['arm'] == arm]
        m = [r for r in rs if r['measurable']]; c = [r for r in rs if not r['measurable']]
        out[arm] = dict(measurable=acc_ci(m) if m else None, control=acc_ci(c) if c else None, n=len(rs))
    return out


def cmd_check(a):
    its = items('dev'); tl = json.load(open(os.path.join(OUT, 'tools_dev.json')))
    m = [(it, t) for it, t in zip(its, tl) if it['measurable']]
    found = np.mean([facts(it, t) is not None for it, t in m])
    agree = np.mean([facts(it, t) is not None and k2(it, t) == it['label'] for it, t in m])
    k2acc = np.mean([k2(it, t) == it['label'] for it, t in m])
    res = dict(n_measurable=len(m), both_found=round(100 * found, 1), agreement=round(100 * agree, 1),
               k2_accuracy=round(100 * k2acc, 1), stop=bool(agree < 0.60))
    by = {}
    for it, t in m:
        by.setdefault(REL[it['relation']][0], []).append(facts(it, t) is not None and k2(it, t) == it['label'])
    res['agreement_by_fact'] = {k: round(100 * float(np.mean(v)), 1) for k, v in by.items()}
    json.dump(res, open(os.path.join(OUT, 'c0_dev.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
    if res['stop']:
        sys.exit('C0 stop rule: tool agreement below 60% on dev; do not run the test (docs/gates/e1_d2_gates.md)')


def cmd_report(a):
    R = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith('eval_') and f.endswith('.json'):
            d = json.load(open(os.path.join(OUT, f)))
            for r in d['rows']:
                R.setdefault((f[5:-5], r['arm']), []).append(r)
    tl = json.load(open(os.path.join(OUT, 'tools_test.json'))); its = items('test')[:len(tl)]
    k2rows = [dict(i=i, image=it['image'], label=it['label'], pred=k2(it, tl[i]), measurable=True)
              for i, it in enumerate(its) if it['measurable']]
    L = ['| student | arm | measurable acc % [95% CI] | control acc % [95% CI] |', '|---|---|---|---|']
    L.append(f'| rule alone (K2) | F4 facts | {acc_ci(k2rows)[0]} {acc_ci(k2rows)[1]} | - |')
    for (stu, arm), rs in R.items():
        s = summarize(rs)[arm]
        L.append(f"| {stu} | {arm} | {s['measurable'][0]} {s['measurable'][1]} | {s['control'][0]} {s['control'][1]} |")
    M = lambda key: [r for r in R.get(key, []) if r['measurable']]
    C = lambda key: [r for r in R.get(key, []) if not r['measurable']]
    L += ['', '| comparison (measurable unless stated) | difference [95% CI] |', '|---|---|']
    g = {}

    def add(name, ka, kb, sel=M):
        if R.get(ka) and R.get(kb):
            d = paired(sel(ka), sel(kb)); g[name] = d; L.append(f'| {name} | {d[0]} [{d[1][0]}, {d[1][1]}] |')
    U = 'untrained'
    for f in ('F3', 'F4', 'F2'):
        add(f'G1 untrained {f} - F1 (photo)', (U, f'{f}:tool:img'), (U, 'F1:tool:img'))
    for f in ('F1', 'F2', 'F3', 'F4', 'F5', 'F6'):
        add(f'G2 untrained {f} + photo - photo alone', (U, f'{f}:tool:img'), (U, 'F0:tool:img'))
    for f in ('F3', 'F4'):
        add(f'untrained {f} note only - photo alone', (U, f'{f}:tool:none'), (U, 'F0:tool:img'))
        add(f'control: untrained {f} - photo alone', (U, f'{f}:tool:img'), (U, 'F0:tool:img'), C)
        add(f'untrained {f} shuffled - photo alone', (U, f'{f}:shuf:img'), (U, 'F0:tool:img'))
    add('untrained F4 negated - photo alone', (U, 'F4:neg:img'), (U, 'F0:tool:img'))
    add('F5 depth picture - F4 text', (U, 'F5:tool:img'), (U, 'F4:tool:img'))
    add('F6 crop - F4 text', (U, 'F6:tool:img'), (U, 'F4:tool:img'))
    for f in ('F1', 'F2', 'F3', 'F4'):
        s = f'lora_{f}'
        add(f'{s}: {f} - no note', (s, f'{f}:tool:img'), (s, 'F0:tool:img'))
        add(f'{s}: {f} shuffled - no note', (s, f'{f}:shuf:img'), (s, 'F0:tool:img'))
        add(f'{s}: no note - untrained photo alone', (s, 'F0:tool:img'), (U, 'F0:tool:img'))
    for f in ('F3', 'F4'):
        add(f'G4 trained {f} - trained F1', (f'lora_{f}', f'{f}:tool:img'), ('lora_F1', 'F1:tool:img'))
    open(os.path.join(OUT, 'E1_D2_table.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    json.dump(dict(gates=g, k2=acc_ci(k2rows)), open(os.path.join(OUT, 'e1_d2_summary.json'), 'w'), indent=1)
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['tools', 'check', 'eval', 'train', 'report'])
    ap.add_argument('--split', default='test'); ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--arms', default=''); ap.add_argument('--lora', default=''); ap.add_argument('--out', default='')
    ap.add_argument('--fmt', default='F4'); ap.add_argument('--save', default='')
    ap.add_argument('--steps', type=int, default=1000); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--drop', type=float, default=0.2)
    a = ap.parse_args()
    {'tools': cmd_tools, 'check': cmd_check, 'eval': cmd_eval, 'train': cmd_train, 'report': cmd_report}[a.cmd](a)
