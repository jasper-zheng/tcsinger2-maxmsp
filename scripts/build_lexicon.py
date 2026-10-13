"""Pronunciation lexicon of the training data, for the inference front-end (WP7, ldm/infer/g2p.py).

Per language: word -> the phoneme sequence it has most often in the training split. A phoneme that the
metadata repeats because the word is sung over several notes is counted once. Reads the metadata only.

Usage (from TCSinger2/, PYTHONPATH=.):
  python scripts/build_lexicon.py            # writes data/meta/lexicon.json
"""
import argparse
import collections
import json

from ldm.infer.g2p import norm_word

REST = ('<AP>', '<SP>')


def word_phonemes(item):
    """(word, phonemes) of the sung words of a metadata item, note-split repeats collapsed."""
    by_word = collections.defaultdict(list)
    for ph, w in zip(item['ph'], item['ph2words']):
        if not by_word[w] or by_word[w][-1] != ph:
            by_word[w].append(ph)
    return [(item['txt'][w], phs) for w, phs in sorted(by_word.items()) if phs[0] not in REST]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='data/meta/metadata.json,data/meta_ext/metadata.json')
    parser.add_argument('--out', default='data/meta/lexicon.json')
    args = parser.parse_args()

    counts = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for path in args.metadata.split(','):
        with open(path, encoding='utf-8') as f:
            items = json.load(f)
        for item in items:
            if item['split'] != 'train':
                continue
            for word, phs in word_phonemes(item):
                counts[item['language']][norm_word(word)][tuple(phs)] += 1
    lexicon = {lang: {w: list(c.most_common(1)[0][0]) for w, c in sorted(words.items())} for lang, words in sorted(counts.items())}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(lexicon, f, ensure_ascii=False, indent=0)
    for lang, words in lexicon.items():
        tokens = sum(sum(c.values()) for c in counts[lang].values())
        top = sum(c.most_common(1)[0][1] for c in counts[lang].values())
        print(f'{lang:9s} {len(words):6d} words, {tokens:7d} tokens, {top / tokens:.1%} of them sung as the entry says')
    print('wrote', args.out)
