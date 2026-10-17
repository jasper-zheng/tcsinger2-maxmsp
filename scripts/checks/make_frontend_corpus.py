"""Scores and what the Python front-end makes of them, for the parity test of the JS front-end in Max.

Writes export/artifacts/frontend_corpus.jsonl, one case per line:
  {"id", "name", "score", "style", "caption", "expected"}
expected is the dict of scripts/score_to_max.py, or {"error": {"kind", "where"}} with the exception class
(ScoreError, G2PError, ValueError) and the "words[i]" it names, if any. The cases (deterministic seed):
  * the 17 examples/;
  * per language, scores of lexicon words (every one at least once) and of dictionary words, some in
    capitals or with punctuation, with apostrophes (French, Italian), without accents (French, Italian,
    Spanish), with "ph", with "ph_durs" across notes and with "notes_per_syllable"; rests of both types;
    tempo and seconds; a style or a caption given from outside;
  * Chinese: lines of pypinyin phrases, the song titles of data/meta/splits, typed pinyin (tones, ü, u:),
    rare characters;
  * scores that must fail: bad pitch, unknown word or phoneme, bad technique or caption, over 20 s, over 256
    pieces, bad notes_per_syllable, bad f0.
Then: node export/max/js/test/parity.js export/artifacts/frontend_corpus.jsonl

Usage (from TCSinger2/): python scripts/checks/make_frontend_corpus.py [--per_language 120] [--seed 0]
"""
import argparse
import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.getcwd())

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
import score_to_max  # noqa: E402
from ldm.data.phone_set import language_set  # noqa: E402
from ldm.infer import g2p  # noqa: E402
from ldm.infer.score import ScoreError, parse_score  # noqa: E402
from ldm.infer.timing import load_duration_model, syllabify  # noqa: E402

OUT = 'export/artifacts/frontend_corpus.jsonl'
PUNCT = ['', '', '', ',', '!', '?', '.', '…', '«', '"']
STYLES = dict(singer=['female', 'male', 'female alto', 'male tenor', 'female soprano', 'male bass'], emotion=['happy', 'sad'],
              singing_method=['pop', 'bel canto'], technique=['none', 'mix', 'breathy', 'falsetto and breathy', 'mix then falsetto'])


def expected(data, name, predictor, style=None, caption=None):
    try:
        d, _ = score_to_max.convert(parse_score(data, name), predictor, caption, style)
    except (ScoreError, g2p.G2PError, ValueError) as e:
        m = re.match(r'words\[(\d+)\]', str(e))
        return {'error': {'kind': type(e).__name__, 'where': int(m.group(1)) if m else None, 'message': str(e)}}
    d['meta']['warnings'] = [w for w in d['meta']['warnings'] if not w.startswith('conditions check')]
    return d


class Maker:
    def __init__(self, rng):
        self.rng = rng

    def notes(self, k, tempo):
        r = self.rng
        lens = [0.25, 0.5, 0.5, 0.75, 1, 1, 1.5, 2] if tempo else [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0, 1.2]
        out = []
        for _ in range(k):
            pitch = r.randint(48, 76)
            length = r.choice(lens) if r.random() < 0.8 else round(r.uniform(0.15, 1.2), r.choice([2, 3, 5]))
            out.append([r.choice([pitch, pitch, g2p_name(pitch)]), length])
        return out

    def score(self, language, texts, given=None, max_seconds=19.5):
        """A score of the words `texts` (some with ph etc. from `given`: text -> phonemes with suffix)."""
        r = self.rng
        tempo = r.choice([None, 90, 112, 120])
        beat = 60 / tempo if tempo else 1.0
        words, total = [], 0.0
        if r.random() < 0.7:
            words.append({'rest': r.choice([0.5, 1])})
            total += words[-1]['rest'] * beat
        for text in texts:
            k = r.choice([1, 1, 1, 2, 2, 3])
            w = {'text': text, 'notes': self.notes(k, tempo)}
            phs = (given or {}).get(text)
            u = r.random()
            if phs and u < 0.15:
                w['ph'] = [p if r.random() < 0.3 else p[:-3] for p in phs]
            elif phs and u < 0.25:
                w['ph'] = [p[:-3] for p in phs]
                w['ph_durs'] = [round(r.uniform(0.03, 0.4), 3) for _ in phs]
            elif phs and u < 0.32:
                n_syl = len(syllabify(phs))
                if n_syl <= k:
                    split = sorted(r.sample(range(1, k), n_syl - 1)) if n_syl > 1 else []
                    bounds = [0] + split + [k]
                    w['notes_per_syllable'] = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
            sec = sum(n[1] for n in w['notes']) * beat
            if total + sec > max_seconds:
                break
            words.append(w)
            total += sec
            if r.random() < 0.12 and total + 0.5 * beat <= max_seconds:
                rest = {'rest': 0.5}
                if r.random() < 0.5:
                    rest['type'] = r.choice(['AP', 'SP'])
                words.append(rest)
                total += 0.5 * beat
        data = {'language': language, 'words': words}
        if tempo:
            data['tempo'] = tempo
        if r.random() < 0.5:
            data['style'] = {k: r.choice(v) for k, v in STYLES.items() if r.random() < 0.6}
        return data


def g2p_name(midi):
    names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    return f'{names[midi % 12]}{midi // 12 - 1}'


def styled(word, r, language):
    u = r.random()
    if u < 0.1:
        word = word.capitalize()
    elif u < 0.15:
        word = word.upper()
    if language in g2p.ACCENT_LANGS and r.random() < 0.15:
        word = g2p._no_accents(word)
    return word + r.choice(PUNCT) if r.random() < 0.3 else word


def chinese_lines(r, n):
    from pypinyin.constants import PHRASES_DICT
    phrases = sorted(PHRASES_DICT)
    lines = []
    for _ in range(n):
        line = ''.join(r.choice(phrases) for _ in range(r.randint(2, 6)))
        lines.append(line[:r.randint(4, 24)])
    titles = set()
    for path in glob.glob('data/meta/splits/*.txt'):
        with open(path, encoding='utf-8') as f:
            for row in f:
                parts = row.strip().split('#')
                if len(parts) > 3 and parts[0] == 'Chinese':
                    t = ''.join(c for c in parts[3] if '一' <= c <= '鿿')
                    if t:
                        titles.add(t)
    return lines, sorted(titles)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=OUT)
    p.add_argument('--per_language', type=int, default=120, help='synthetic scores per language')
    p.add_argument('--dict_words', type=int, default=2000, help='dictionary words per language')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    r = random.Random(args.seed)
    mk = Maker(r)
    predictor = load_duration_model('configs/tcsinger2.yaml', 'cpu')
    cases = []

    def add(kind, data, name='score', style=None, caption=None):
        cases.append(dict(id=f'{kind}/{len(cases)}', name=name, score=data, style=style, caption=caption,
                          expected=expected(data, name, predictor, style, caption)))

    for path in sorted(glob.glob('examples/*.json')):
        with open(path, encoding='utf-8') as f:
            add('example', json.load(f), os.path.splitext(os.path.basename(path))[0])

    for language in language_set:
        lex = g2p.lexicon(language)
        pool = sorted(lex)
        if language != 'Chinese':
            dict_words = sorted(g2p.mfa_dict(language))
            pool += r.sample(dict_words, min(args.dict_words, len(dict_words)))
        r.shuffle(pool)
        given = {w: v for w, v in lex.items()}
        if language == 'Chinese':
            pool = [w for w in pool if len(w) == 1] + [w for w in pool if len(w) != 1]
        n_scores, i = max(args.per_language, len(pool) // 12 + 1), 0
        for s in range(n_scores):
            words = []
            for _ in range(r.randint(6, 16)):
                words.append(pool[i % len(pool)])
                i += 1
            if language in ('French', 'Italian') and r.random() < 0.3:
                words.append(r.choice(["l'", "d'", "dell'", "c'", "qu'", "un'"]) + r.choice(pool))
            texts = [styled(w, r, language) for w in words]
            g = {t: given[w] for t, w in zip(texts, words) if w in given}
            data = mk.score(language, texts, g)
            if s % 10 == 3:
                add(language, data, f'{language}_{s}', style={k: r.choice(v) for k, v in STYLES.items() if r.random() < 0.5})
            elif s % 10 == 7:
                add(language, data, f'{language}_{s}', caption=r.choice(['', 'Singer: male bass, Emotion: sad', 'Technique: mix']))
            else:
                add(language, data, f'{language}_{s}')

    lines, titles = chinese_lines(r, 150)
    for line in lines + titles:
        add('zh_line', mk.score('Chinese', list(line)), 'zh')
    from ldm.infer.score import F0_FPS  # noqa: F401
    syllables = ['zhuang', 'nv', 'lv', 'lu:e', 'nü', 'yuan', 'wo', 'er', 'ri', 'shi', 'ci', 'jiong', 'xue', 'ang', 'ou', 'm', 'n',
                 'hm', 'ng', 'yo', 'lo', 'ueng', 'qiu', 'gui', 'dun', 'ê', 'zz', 'abc', 'Hao', 'MA', 'li3', 'nv3', 'lü4']
    for k in range(40):
        texts = [r.choice(syllables) + r.choice(['', '', '1', '2', '3', '4', '5']) for _ in range(r.randint(3, 8))]
        add('zh_pinyin', mk.score('Chinese', texts), 'zh_pinyin')
    # every toneless reading pypinyin knows and random strings, one per score (a token without phonemes fails)
    from pypinyin.constants import PINYIN_DICT
    from pypinyin.contrib.tone_convert import to_normal
    readings = sorted({to_normal(x) for v in PINYIN_DICT.values() for x in v.split(',')})
    letters = 'abcdefghijklmnopqrstuvwxyz'
    junk = [''.join(r.choice(letters + 'ü:') for _ in range(r.randint(1, 6))) for _ in range(400)]
    for token in readings + junk:
        token = token + r.choice(['', '', '3'])
        add('zh_typed', {'language': 'Chinese', 'words': [{'text': token, 'notes': [[60, 1]]}]}, 'zh_typed')
    rare = [chr(c) for c in range(0x3400, 0x4DC0, 37)] + [chr(c) for c in range(0x9F00, 0xA000, 7)]
    for k in range(30):
        add('zh_rare', mk.score('Chinese', [r.choice(rare) for _ in range(r.randint(1, 4))]), 'zh_rare')

    # scores that must fail, and a few edge cases
    base = lambda: {'language': 'English', 'words': [{'text': 'love', 'notes': [['C4', 1]]}]}
    bad = [
        {'language': 'Klingon', 'words': [{'text': 'qa', 'notes': [[60, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [['H4', 1]]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[99, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[60.5, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[60, -1]]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[60, 1]]}, {'text': 'qzxv', 'notes': [[62, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'ph': ['L', 'XX9'], 'notes': [[60, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'ph': ['L', 'AH1', 'V'], 'ph_durs': [0.1, 0.2], 'notes': [[60, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[60, 1], [62, 1]], 'notes_per_syllable': [1, 1]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[60, 1]], 'notes_per_syllable': [2]}]},
        {**base(), 'words': [{'text': 'love', 'notes': [[60, 25]]}]},
        {**base(), 'words': [{'rest': 1}]},
        {**base(), 'colour': 'blue'},
        {**base(), 'tempo': 0},
        {**base(), 'style': {'voice': 'nice'}},
        {**base(), 'style': {'singer': 'female baritone'}},
        {**base(), 'style': {'technique': 'breathy and mix'}},
        {**base(), 'f0': [0, 220, -3]},
        {**base(), 'f0': [0, 220, 221], 'f0_fps': 100},
        {**base(), 'words': [{'text': 'love', 'notes': [[60, 1]]}, {'rest': 1, 'type': 'XX'}]},
        {**base(), 'words': [{'text': 'constructor', 'notes': [[60, 1]]}, {'text': 'toString', 'notes': [[62, 1]]}]},
        {**base(), 'words': [{'text': '', 'ph': ['L', 'AH1', 'V'], 'notes': [[60, 1]]}]},
        {**base(), 'words': [{'text': 'love', 'ph': ['a_zh', 'o_zh'], 'notes': [[60, 1]]}]},
    ]
    many = {'language': 'English', 'words': [{'text': 'strengths', 'notes': [[60, 0.07]]} for _ in range(280)]}
    for data in bad + [many]:
        add('bad', data, 'bad')
    add('bad', base(), 'bad', caption='Singer female')
    add('bad', base(), 'bad', style={'emotion': 'angry'})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    errors = sum('error' in c['expected'] for c in cases)
    print(f'{len(cases)} cases ({errors} that fail in Python) -> {args.out}')


if __name__ == '__main__':
    main()
