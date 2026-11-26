"""Download the public datasets the evidence-pipeline experiments use (never commit them: data/ is in .gitignore).

Plain: one command fetches every dataset from its original source into data/public/<dataset>/, keeps the archives
zipped, and picks up where it stopped if the connection drops. Three datasets need a person: DocVQA and
InfographicVQA (each user registers and accepts their terms) and the bird photos, CUB (a browser download); the script
prints the steps and checks the files once you have saved them. `--check` confirms every file has the size we got;
`--verify` also compares fingerprints (sha256), which takes a few minutes per 100 GB.

Technical: sources, licences and groups are in manifest.py; expected sizes and sha256 in checksums.json. 'url' and
'gdrive' items use urllib with HTTP Range resume (5 tries, 30 s back-off); 'hf' and 'hf_model' items use
huggingface_hub.snapshot_download (no login: all repos are public and ungated). Standard library only, plus
huggingface_hub for the Hugging Face items.

  python -m agent.data.download --list                      # what exists, sizes, licences, what is present
  python -m agent.data.download                             # everything (about 170 GB)
  python -m agent.data.download --only e1_d2                # one experiment's needs (group names in manifest.GROUPS)
  python -m agent.data.download --only vsr,coco2017 --data /content/drive/MyDrive/spatial_llm/data/public   # Colab
  python -m agent.data.download --check [--verify]          # check what is on disk, download nothing
"""
import argparse, hashlib, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

from .manifest import DATASETS, GD, GROUPS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SUMS = json.load(open(HERE / 'checksums.json')) if (HERE / 'checksums.json').exists() else {}


def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)


def expected(name):
    """{relative path inside the dataset folder: (bytes, sha256)} from checksums.json"""
    pre = name + '/'
    return {k[len(pre):]: tuple(v) for k, v in SUMS.items() if k.startswith(pre)}


def fetch(url, out: Path, size=None, tries=5):
    out.parent.mkdir(parents=True, exist_ok=True)
    if size and out.exists() and out.stat().st_size == size:
        log(f'skip {out.name} (complete)'); return True
    for t in range(tries):
        try:
            have = out.stat().st_size if out.exists() else 0
            req = urllib.request.Request(url, headers={'Range': f'bytes={have}-'} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = 'ab' if (have and r.status == 206) else 'wb'
                log(f'get  {out.name}' + (f' (resuming at {have / 1e9:.2f} GB)' if mode == 'ab' else ''))
                with open(out, mode) as f:
                    for b in iter(lambda: r.read(8 << 20), b''):
                        f.write(b)
            if size and out.stat().st_size != size:
                raise IOError(f'size {out.stat().st_size} != expected {size}')
            log(f'done {out.name} {out.stat().st_size / 1e9:.2f} GB'); return True
        except urllib.error.HTTPError as e:
            if e.code == 416:                       # nothing left to fetch
                log(f'skip {out.name} (complete)'); return True
            log(f'retry {t + 1}/{tries} {out.name}: {e}'); time.sleep(30)
        except Exception as e:                      # noqa: BLE001 -- network errors: retry, then report
            log(f'retry {t + 1}/{tries} {out.name}: {e}'); time.sleep(30)
    log(f'FAILED {out.name}'); return False


def download(name, root: Path):
    d = DATASETS[name]; folder = root / name; exp = expected(name)
    if d['kind'] in ('url', 'gdrive'):
        ok = True
        for fname, src in d['items']:
            url = GD + src if d['kind'] == 'gdrive' else src
            ok &= fetch(url, folder / fname, exp.get(fname, (None,))[0])
        return ok
    if d['kind'] in ('hf', 'hf_model'):
        from huggingface_hub import snapshot_download
        kw = dict(repo_type='dataset') if d['kind'] == 'hf' else {}
        snapshot_download(d['repo'], revision=d.get('rev'), local_dir=str(folder), allow_patterns=d.get('patterns'), **kw)
        log(f'done {name} ({d["repo"]})'); return True
    # manual
    missing = [f for f in d['files'] if not (folder / f).exists()]
    if missing:
        log(f'MANUAL {name}: missing {missing}\n    {d["how"].replace("<data>", str(root))}')
        return False
    log(f'ok   {name} (manual files present)'); return True


def check(name, root: Path, verify=False):
    """returns (status, detail). status: ok | missing | partial | bad"""
    exp = expected(name); folder = root / name
    if not exp:
        return ('ok' if folder.exists() and any(folder.iterdir()) else 'missing'), 'no checksums recorded'
    bad, miss = [], []
    for rel, (size, sha) in exp.items():
        p = folder / rel
        if not p.exists():
            miss.append(rel); continue
        if p.stat().st_size != size:
            bad.append(f'{rel} size {p.stat().st_size} != {size}'); continue
        if verify:
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(16 << 20), b''):
                    h.update(b)
            if h.hexdigest() != sha:
                bad.append(f'{rel} sha256 differs')
    if bad:
        return 'bad', '; '.join(bad[:3])
    if miss:
        return ('missing' if len(miss) == len(exp) else 'partial'), f'{len(miss)} of {len(exp)} files missing'
    return 'ok', f'{len(exp)} files' + (' verified' if verify else ' (sizes match)')


def select(only):
    if not only:
        return list(DATASETS)
    names = []
    for x in only.split(','):
        x = x.strip()
        names += GROUPS.get(x, [x])
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f'unknown dataset(s) {unknown}; choose from {list(DATASETS)} or groups {list(GROUPS)}')
    return list(dict.fromkeys(names))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=str(REPO / 'data' / 'public'), help='target folder (default <repo>/data/public)')
    ap.add_argument('--only', default='', help='comma-separated dataset names or group names')
    ap.add_argument('--list', action='store_true', help='show the datasets and what is on disk; download nothing')
    ap.add_argument('--check', action='store_true', help='check sizes on disk; download nothing')
    ap.add_argument('--verify', action='store_true', help='with --check: also compare sha256 (slow)')
    a = ap.parse_args(argv)
    root = Path(a.data); names = select(a.only)
    if a.list or a.check:
        tot = 0.0
        for n in names:
            d = DATASETS[n]; st, det = check(n, root, a.verify and a.check); tot += d['gb']
            print(f'{n:36s} {d["gb"]:7.2f} GB  {d["kind"]:8s} {st:8s} {det:28s}  {", ".join(d["used_by"]):12s}  {d["licence"]}')
        print(f'total {tot:.1f} GB in {root}')
        return 0
    need = sum(DATASETS[n]['gb'] for n in names)
    free = __import__('shutil').disk_usage(root if root.exists() else root.anchor or '.').free / 1e9
    log(f'{len(names)} item(s), about {need:.1f} GB; {free:.0f} GB free at {root}')
    if need > free:
        sys.exit('not enough disk space')
    failed = [n for n in names if not download(n, root)]
    for n in names:
        st, det = check(n, root)
        log(f'check {n}: {st} ({det})')
    if failed:
        log(f'not complete: {failed} (manual steps above, or re-run the same command to resume)')
        return 1
    log('all done')
    return 0


if __name__ == '__main__':
    sys.exit(main())
