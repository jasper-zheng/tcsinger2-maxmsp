"""Assign train / valid / test_seen / test_unseen splits to the extra corpora (WP9), merge them into
one metadata.json and write a data report. The GTSinger splits are not touched.

The corpora are handled in a fixed order (M4Singer, Opencpop, OpenSinger), each with its own random
stream, so adding a corpus later never changes the splits of an earlier one:

- test_unseen: every clip of the corpus's held-out singers (drawn half female, half male). Their
  songs are NOT held out for the other singers: the corpora share many titles with each other and
  with GTSinger's fixed training split, so that guarantee cannot be had, and it would cost hours of
  training data. The report counts the test_unseen clips whose song is not in training at all.
- a song title that is in valid or test_seen in GTSinger (Chinese) or in an earlier corpus is in
  the same split here, for every singer
- further whole titles are drawn until valid and test_seen reach the corpus's target hours; titles
  that are training songs in GTSinger or an earlier corpus are not drawn
- everything else is train

Opencpop has song numbers, not titles, so it cannot be matched against the other corpora.

Usage:
  python preprocess/ext_split_report.py --meta_dir data/meta_ext --gtsinger_meta data/meta/metadata.json
"""
import argparse
import collections
import json
import os
import random
import re
import unicodedata

SPLITS = ['train', 'valid', 'test_seen', 'test_unseen']
# corpus -> (file, held-out singers, target hours for valid and for test_seen)
CORPORA = {'M4Singer': ('m4singer.json', 2, 0.5), 'Opencpop': ('opencpop.json', 0, 0.15), 'OpenSinger': ('opensinger.json', 6, 0.5)}


def title_key(item):
    dataset, _, song = item['item_name'].split('#')[:3]
    if dataset == 'Opencpop':
        return 'opencpop:' + song
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFC', song).casefold())


def gtsinger_titles(path):
    """{title: split} of the Chinese GTSinger songs."""
    with open(path, encoding='utf-8') as f:
        items = json.load(f)
    out = {}
    for it in items:
        if it['language'] == 'Chinese':
            song = it['item_name'].split('#')[3]
            out[re.sub(r'[\W_]+', '', unicodedata.normalize('NFC', song).casefold())] = it['split']
    return out


def draw_unseen(items, n, rng, override):
    singers = sorted({it['singer'] for it in items})
    if override is not None:
        return sorted(s for s in singers if s in override)
    gender = {it['singer']: it['gender'] for it in items}
    chosen = []
    for g in ('female', 'male'):
        chosen += rng.sample([s for s in singers if gender[s] == g], n // 2)
    return sorted(chosen)


def assign(items, n_unseen, target_h, known, rng, override):
    """Sets it['split'] for the items of one corpus. known: {title: split} of GTSinger and earlier corpora."""
    unseen = set(draw_unseen(items, n_unseen, rng, override))
    title_split = {}
    for it in items:
        k = title_key(it)
        if known.get(k) in ('valid', 'test_seen'):
            title_split[k] = known[k]
    seen_items = [it for it in items if it['singer'] not in unseen]
    hours = collections.Counter()
    by_title = collections.defaultdict(float)
    for it in seen_items:
        by_title[title_key(it)] += it['duration'] / 3600
    for k, s in title_split.items():
        hours[s] += by_title.get(k, 0.0)
    free = sorted(k for k in by_title if k not in title_split and k not in known)
    rng.shuffle(free)
    for split in ('valid', 'test_seen'):
        while hours[split] < target_h and free:
            k = free.pop()
            title_split[k] = split
            hours[split] += by_title[k]
    for it in items:
        it['split'] = 'test_unseen' if it['singer'] in unseen else title_split.get(title_key(it), 'train')
    return sorted(unseen)


def table(rows, header):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def stats(items):
    return [len(items), f"{sum(it['duration'] for it in items) / 3600:.2f}", len({it['singer'] for it in items}),
            len({title_key(it) for it in items})]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_dir', default='data/meta_ext')
    parser.add_argument('--gtsinger_meta', default='data/meta/metadata.json')
    parser.add_argument('--unseen_singers', nargs='*', default=None, help='held-out singer IDs instead of the seeded draw')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    gts = gtsinger_titles(args.gtsinger_meta)
    known = dict(gts)  # title -> split, grows corpus by corpus
    merged, unseen_all, report = [], {}, []
    for name, (fn, n_unseen, target_h) in CORPORA.items():
        path = os.path.join(args.meta_dir, fn)
        if not os.path.exists(path):
            print(f'{name}: {path} not built yet, skipped')
            continue
        with open(path, encoding='utf-8') as f:
            items = json.load(f)
        rng = random.Random(f'{args.seed}/{name}')
        unseen_all[name] = assign(items, n_unseen, target_h, known, rng, args.unseen_singers)
        for it in items:  # a title keeps the first split it got; held-out singers do not define one
            if it['split'] != 'test_unseen':
                known.setdefault(title_key(it), it['split'])
        # no song is shared between train and valid / test_seen, here or with GTSinger and earlier corpora;
        # no held-out singer is in another split
        train_titles = {title_key(it) for it in items if it['split'] == 'train'}
        assert not train_titles & {title_key(it) for it in items if it['split'] in ('valid', 'test_seen')}, name
        assert all(known[title_key(it)] == it['split'] for it in items if it['split'] != 'test_unseen'), name
        assert not {it['singer'] for it in items if it['split'] == 'test_unseen'} & {it['singer'] for it in items if it['split'] != 'test_unseen'}, name
        report.append((name, items))
        merged += items

    with open(os.path.join(args.meta_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False)

    md = ['# Extra data report (WP9)', '',
          f'Generated by `preprocess/ext_split_report.py` from `{args.meta_dir}`. Seed {args.seed}.', '',
          '## Splits', '',
          table([[name, s] + stats([it for it in items if it['split'] == s]) for name, items in report for s in SPLITS]
                + [['all', s] + stats([it for it in merged if it['split'] == s]) for s in SPLITS]
                + [['all', 'total'] + stats(merged)],
                ['corpus', 'split', 'clips', 'hours', 'singers', 'songs']), '',
          '## Held-out singers', '']
    train_known = {k for k, s in known.items() if s == 'train'}
    for name, items in report:
        mine = [it for it in items if it['split'] == 'test_unseen']
        new = [it for it in mine if title_key(it) not in train_known]
        md.append(f"- {name}: {', '.join(unseen_all[name]) or 'none'}."
                  + (f" {len(new)} of their {len(mine)} clips ({sum(it['duration'] for it in new) / 3600:.2f} h) are of songs "
                     f'that no training singer sings.' if mine else ''))
    by_singer = collections.defaultdict(list)
    for it in merged:
        by_singer[it['singer']].append(it)
    rows = []
    for singer, its in sorted(by_singer.items(), key=lambda kv: (kv[1][0]['dataset'], kv[0])):
        h = collections.Counter()
        for it in its:
            h[it['split']] += it['duration'] / 3600
        rows.append([singer, its[0]['gender'], its[0]['vocal_range'] or '', len(its), len({title_key(it) for it in its})]
                    + [f'{h[s]:.2f}' for s in SPLITS])
    md += ['', '## Singers', '', table(rows, ['singer', 'gender', 'range', 'clips', 'songs'] + [s + ' h' for s in SPLITS]), '']
    with open(os.path.join(args.meta_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('\n'.join(md[:md.index('## Singers')]))
