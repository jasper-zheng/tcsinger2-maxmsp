"""Extract the GTSinger language zips into one clean tree.

- re-decodes file names stored without the zip UTF-8 flag (cp437 -> utf-8) and NFC-normalises them
- skips macOS resource forks (__MACOSX, ._*, .DS_Store)
- renames TextGrids stored as NNNN_TextGrid to NNNN.TextGrid
- idempotent: files already present with the right size are skipped

Usage: python preprocess/gtsinger_unzip.py --zip_dir /notebooks/GTSinger-Dataset --out_dir /notebooks/data/GTSinger
"""
import argparse
import collections
import glob
import os
import re
import shutil
import unicodedata
import zipfile
from multiprocessing import Pool


def fix_name(info):
    name = info.filename
    if not info.flag_bits & 0x800:  # no UTF-8 flag: zipfile decoded the raw bytes as cp437
        try:
            name = name.encode('cp437').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    name = unicodedata.normalize('NFC', name)
    d, b = os.path.split(name)
    m = re.fullmatch(r'(\d+)_TextGrid', b)
    if m:
        b = m.group(1) + '.TextGrid'
    return os.path.join(d, b)


def wanted(info, name):
    if info.is_dir():
        return False
    parts = name.split('/')
    if parts[0] == '__MACOSX' or parts[-1].startswith('._') or parts[-1] == '.DS_Store':
        return False
    if os.path.isabs(name) or '..' in parts:
        raise ValueError(f'unsafe path in archive: {name!r}')
    return True


def extract_shard(args):
    zip_path, out_dir, shard, n_shards = args
    counts = collections.Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for i, info in enumerate(zf.infolist()):
            if i % n_shards != shard:
                continue
            name = fix_name(info)
            if not wanted(info, name):
                continue
            dst = os.path.join(out_dir, name)
            ext = os.path.splitext(name)[1].lower()
            counts[ext] += 1
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                counts['skipped'] += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + '.part'
            with zf.open(info) as src, open(tmp, 'wb') as f:
                shutil.copyfileobj(src, f, 1 << 20)
            os.replace(tmp, dst)
    return os.path.basename(zip_path), counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--shards', type=int, default=4, help='parallel shards per zip')
    args = parser.parse_args()

    zips = sorted(glob.glob(os.path.join(args.zip_dir, '*.zip')))
    tasks = [(z, args.out_dir, s, args.shards) for s in range(args.shards) for z in zips]
    totals = collections.defaultdict(collections.Counter)
    with Pool(args.workers) as pool:
        for zname, counts in pool.imap_unordered(extract_shard, tasks):
            totals[zname].update(counts)
    for zname in sorted(totals):
        print(zname, dict(sorted(totals[zname].items())))
