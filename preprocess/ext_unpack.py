"""Unpack the extra corpora (WP9) into one tree of 48 kHz mono wavs.

  <out_root>/M4Singer/M4-Alto-1/<song>/NNNN.wav            + M4Singer/meta.json
  <out_root>/Opencpop/OCP-1/<song id>/NNNN.wav             + Opencpop/segments.json, Opencpop/textgrids/
  <out_root>/OpenSinger/OS-Woman-12/<song>/NNNN.wav        + OpenSinger/labels.json

- every wav is resampled to --sample_rate and written atomically; wavs already present are skipped
- M4Singer: clips as in the zip (44.1, 48 and 96 kHz by singer)
- Opencpop: the zips are password-protected (environment variable OPENCPOP_ZIP_PASSWORD); the
  whole-song wavs are cut at the sentence tier of the TextGrid, NNNN is the index of the sentence
  interval in that tier
- OpenSinger: clips as in the tar; the phoneme (.lab) and text (.txt) files go into labels.json

The archives are untrusted: run with `python -I`; member names are validated and only wav / label
content is read.

Usage:
  python -I preprocess/ext_unpack.py --dataset m4singer --archive /notebooks/M4Singer/m4singer.zip --out_root /notebooks/data/ext
  OPENCPOP_ZIP_PASSWORD=... python -I preprocess/ext_unpack.py --dataset opencpop --archive /notebooks/OpenCPop --out_root /notebooks/data/ext
  python -I preprocess/ext_unpack.py --dataset opensinger --archive /notebooks/OpenSinger/OpenSinger.tar.gz --out_root /notebooks/data/ext
"""
import argparse
import collections
import io
import json
import os
import re
import subprocess
import tarfile
import unicodedata
import zipfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf
import soxr


def safe_parts(name):
    parts = unicodedata.normalize('NFC', name).split('/')
    if os.path.isabs(name) or '..' in parts or any('\x00' in p or '\t' in p or '\n' in p for p in parts):
        raise ValueError(f'unsafe path in archive: {name!r}')
    return parts


def write_wav(x, sr, dst, subtype):
    """Resample a float mono signal to ARGS.sample_rate and write it atomically."""
    if sr != ARGS.sample_rate:
        x = soxr.resample(x, sr, ARGS.sample_rate, quality='VHQ')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    sf.write(tmp, np.clip(x, -1.0, 1.0), ARGS.sample_rate, subtype=subtype, format='WAV')
    os.replace(tmp, dst)


def convert_bytes(job):
    data, dst = job
    x, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    write_wav(x.mean(axis=1), sr, dst, 'PCM_16')
    return sr


# ---------------------------------------------------------------- M4Singer

def m4singer_shard(shard):
    counts = collections.Counter()
    with zipfile.ZipFile(ARGS.archive) as zf:
        for i, info in enumerate(zf.infolist()):
            if i % ARGS.workers != shard or info.is_dir() or not info.filename.endswith('.wav'):
                continue
            parts = safe_parts(info.filename)  # m4singer/<Singer>#<Song>/NNNN.wav
            singer, song = parts[1].split('#')
            dst = os.path.join(ARGS.out_root, 'M4Singer', 'M4-' + singer, song, parts[2])
            if os.path.exists(dst):
                counts['skipped'] += 1
                continue
            counts[convert_bytes((zf.read(info), dst))] += 1
    return counts


def unpack_m4singer():
    out = os.path.join(ARGS.out_root, 'M4Singer')
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(ARGS.archive) as zf:
        meta = json.loads(zf.read('m4singer/meta.json'))
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    totals = collections.Counter()
    with Pool(ARGS.workers) as pool:
        for counts in pool.imap_unordered(m4singer_shard, range(ARGS.workers)):
            totals.update(counts)
    print('M4Singer: clips by source sample rate (skipped = already there):', dict(totals))


# ---------------------------------------------------------------- Opencpop

def read_textgrid(path):
    """{tier name: [(start, end, text), ...]} of a long-format Praat TextGrid."""
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'utf-16'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    tiers = {}
    for block in re.split(r'item \[\d+\]:', text)[1:]:
        name = re.search(r'name = "(.*?)"', block).group(1)
        tiers[name] = [(float(a), float(b), c) for a, b, c in
                       re.findall(r'xmin = (\S+)\s+xmax = (\S+)\s+text = "(.*?)"', block)]
    return tiers


def opencpop_song(song):
    raw = os.path.join(ARGS.out_root + '_raw', 'Opencpop')
    sentences = list(read_textgrid(os.path.join(raw, 'textgrids', song + '.TextGrid')).values())[0]
    todo = []
    for i, (start, end, text) in enumerate(sentences):
        if text.strip() in ('', 'silence', 'SP', 'AP'):
            continue
        dst = os.path.join(ARGS.out_root, 'Opencpop', 'OCP-1', song, f'{i:04d}.wav')
        todo.append((i, start, end, text, dst))
    if any(not os.path.exists(t[-1]) for t in todo):
        x, sr = sf.read(os.path.join(raw, 'wavs', song + '.wav'), dtype='float32', always_2d=True)
        x = x.mean(axis=1)
        for i, start, end, text, dst in todo:
            if not os.path.exists(dst):
                write_wav(x[int(round(start * sr)):int(round(end * sr))], sr, dst, 'PCM_24')
    return [dict(song=song, index=i, start=start, end=end, text=text) for i, start, end, text, _ in todo]


def unpack_opencpop():
    raw = os.path.join(ARGS.out_root + '_raw', 'Opencpop')
    password = os.environ.get('OPENCPOP_ZIP_PASSWORD')
    for name in ('textgrids', 'midis', 'wavs'):
        dst = os.path.join(raw, name)
        with zipfile.ZipFile(os.path.join(ARGS.archive, name + '.zip')) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
        for info in infos:
            if len(safe_parts(info.filename)) != 1:
                raise ValueError(f'unexpected path in {name}.zip: {info.filename!r}')
        if all(os.path.exists(os.path.join(dst, i.filename)) and os.path.getsize(os.path.join(dst, i.filename)) == i.file_size
               for i in infos):
            continue
        if not password:
            raise SystemExit('set OPENCPOP_ZIP_PASSWORD to extract the Opencpop zips')
        os.makedirs(dst, exist_ok=True)
        subprocess.run(['unzip', '-o', '-q', '-P', password, os.path.join(ARGS.archive, name + '.zip'), '-d', dst], check=True)
    songs = sorted(f[:-9] for f in os.listdir(os.path.join(raw, 'textgrids')) if f.endswith('.TextGrid'))
    with Pool(ARGS.workers) as pool:
        segments = [s for song in pool.imap(opencpop_song, songs) for s in song]
    out = os.path.join(ARGS.out_root, 'Opencpop')
    with open(os.path.join(out, 'segments.json'), 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False)
    print(f'Opencpop: {len(songs)} songs, {len(segments)} sentences, '
          f'{sum(s["end"] - s["start"] for s in segments) / 3600:.2f} h')


# ---------------------------------------------------------------- OpenSinger

def opensinger_target(parts):
    """OpenSinger/ManRaw/1_<song>/1_<song>_32.wav -> (OS-Man-1, <song>, 32)"""
    group, folder, base = parts[-3], parts[-2], parts[-1]
    if group not in ('ManRaw', 'WomanRaw'):
        raise ValueError(f'unexpected OpenSinger path: {"/".join(parts)!r}')
    number, song = folder.split('_', 1)
    return f'OS-{group[:-3]}-{int(number)}', song, int(os.path.splitext(base)[0].rsplit('_', 1)[1])


def unpack_opensinger():
    out = os.path.join(ARGS.out_root, 'OpenSinger')
    labels = collections.defaultdict(dict)
    totals = collections.Counter()
    batch = []

    def flush(pool):
        for sr in pool.imap_unordered(convert_bytes, batch, chunksize=8):
            totals[sr] += 1
        batch.clear()

    with Pool(ARGS.workers) as pool, tarfile.open(ARGS.archive, 'r|gz') as tar:
        for member in tar:
            base = member.name.split('/')[-1]
            ext = os.path.splitext(base)[1]
            if not member.isfile() or base.startswith('._') or ext not in ('.wav', '.lab', '.txt'):
                continue
            singer, song, number = opensinger_target(safe_parts(member.name))
            rel = os.path.join(singer, song, f'{number:04d}')
            if ext == '.wav':
                dst = os.path.join(out, rel + '.wav')
                if os.path.exists(dst):
                    totals['skipped'] += 1
                    continue
                batch.append((tar.extractfile(member).read(), dst))
                if len(batch) >= 256:
                    flush(pool)
            else:
                labels[rel][ext[1:]] = tar.extractfile(member).read().decode('utf-8', 'replace').strip()
        flush(pool)
    with open(os.path.join(out, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False)
    print(f'OpenSinger: clips by source sample rate (skipped = already there): {dict(totals)}; {len(labels)} label entries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['m4singer', 'opencpop', 'opensinger'])
    parser.add_argument('--archive', required=True, help='the zip / tar.gz, or the directory of the three Opencpop zips')
    parser.add_argument('--out_root', default='/notebooks/data/ext')
    parser.add_argument('--sample_rate', type=int, default=48000)
    parser.add_argument('--workers', type=int, default=8)
    ARGS = parser.parse_args()
    ARGS.out_root = os.path.abspath(ARGS.out_root)
    dict(m4singer=unpack_m4singer, opencpop=unpack_opencpop, opensinger=unpack_opensinger)[ARGS.dataset]()
