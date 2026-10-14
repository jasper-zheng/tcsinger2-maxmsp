"""Lyrics to the phonemes of the training phone set (WP7).

The phonemes of a sung word come from, in this order:
  given    the word's own `ph` in the score
  pinyin   Chinese: pypinyin initials and finals (strict mode, read from the whole line so that characters
           with several readings are resolved in context), or a pinyin syllable typed as the text
  lexicon  data/meta/lexicon.json: how the word is sung in the training data (scripts/build_lexicon.py)
  dict     GTSinger's MFA pronunciation dictionary: the most probable variant whose phonemes are all in
           the phone set. Japanese and Korean dictionary entries often differ from the training labels
           (71% and 40% agreement on the training words), so these get a warning. A word with apostrophes
           is also tried part by part, and in French, Italian and Spanish a word typed without its accents
           is read as the dictionary word with them.
A word that none of them knows raises G2PError, which names the word and asks for `ph`.
"""
import json
import os
import re
import unicodedata

from ldm.data.phone_set import PHONE_SET_PATH, ph2id

LANG_SUFFIX = {'Chinese': '_zh', 'English': '_en', 'French': '_fr', 'German': '_de', 'Italian': '_it',
               'Japanese': '_ja', 'Korean': '_ko', 'Russian': '_ru', 'Spanish': '_es'}
REST = ('<AP>', '<SP>')
LEXICON_PATH = os.path.join(os.path.dirname(PHONE_SET_PATH), 'lexicon.json')
# data/mfa_dict where scripts/setup_env_mac.sh downloads them, else the GTSinger checkout on /notebooks
_LOCAL_MFA_DICT = os.path.join(os.path.dirname(os.path.dirname(PHONE_SET_PATH)), 'mfa_dict')
MFA_DICT_DIR = os.environ.get('TCSINGER2_MFA_DICT', _LOCAL_MFA_DICT if os.path.isdir(_LOCAL_MFA_DICT)
                              else '/notebooks/GTSinger/Data-Process/utils/mfa_dict')
WEAK_DICT = ('Japanese', 'Korean')
ACCENT_LANGS = ('French', 'Italian', 'Spanish')  # lyrics are often typed without their accents
# as preprocess/gtsinger_build_metadata.py cleans the labels
IPA_LANGS = {'French', 'German', 'Italian', 'Japanese', 'Korean', 'Russian', 'Spanish'}
PH_FIXES = {
    'Japanese': {'t͡ɕ': 'tɕ', 't͡s': 'ts', 'ʨ': 'tɕ'},
    'Korean': {'t͡ɕ': 'tɕ', 't͡ɕʰ': 'tɕʰ', 't͡ɕ͈': 'tɕ͈', 't͈ɕ': 'tɕ͈', 'd͡ʑ': 'dʑ'},
}
# pinyin finals that the phone set does not have, written with the nearest one it has
ZH_FINAL_FIXES = {'ueng': 'ong', 'io': 'iou', 'ê': 'e'}

_lexicon = None
_dicts = {}


class G2PError(ValueError):
    pass


def norm_word(word):
    word = unicodedata.normalize('NFC', word).replace('\u2019', "'").lower()
    return word.strip(' \t.,;:!?¡¿"()[]…«»')


def _no_accents(word):
    return ''.join(c for c in unicodedata.normalize('NFD', word) if not unicodedata.combining(c))


def lexicon(language):
    global _lexicon
    if _lexicon is None:
        if not os.path.exists(LEXICON_PATH):
            raise FileNotFoundError(f'{LEXICON_PATH} not found: build it with scripts/build_lexicon.py')
        with open(LEXICON_PATH, encoding='utf-8') as f:
            _lexicon = json.load(f)
    return _lexicon.get(language, {})


def _clean(p, language):
    p = unicodedata.normalize('NFC', p)
    if p == 'g' and language in IPA_LANGS:
        p = 'ɡ'
    return PH_FIXES.get(language, {}).get(p, p)


def mfa_dict(language):
    """word -> [(probability, phonemes with the language suffix)], only variants inside the phone set."""
    if language not in _dicts:
        path = os.path.join(MFA_DICT_DIR, f'{language.lower()}_mfa.dict')
        entries = {}
        if os.path.exists(path):
            suffix = LANG_SUFFIX[language]
            with open(path, encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 2:
                        parts = line.split(None, 1)
                        if len(parts) < 2:
                            continue
                    prob = float(parts[1]) if len(parts) > 2 and re.fullmatch(r'[\d.]+', parts[1]) else 1.0
                    phs = [_clean(p, language) + suffix for p in parts[-1].split()]
                    if phs and all(p in ph2id for p in phs):
                        entries.setdefault(norm_word(parts[0]), []).append((prob, phs))
        _dicts[language] = entries
    return _dicts[language]


def _unaccented(language):
    """Dictionary words by their spelling without accents (the most probable one where several share it)."""
    key = language + '/unaccented'
    if key not in _dicts:
        index = {}
        for word, variants in mfa_dict(language).items():
            plain, prob = _no_accents(word), max(v[0] for v in variants)
            if plain != word and (plain not in index or prob > index[plain][0]):
                index[plain] = (prob, word)
        _dicts[key] = {plain: word for plain, (_, word) in index.items()}
    return _dicts[key]


def lookup(key, language, lex):
    """(phonemes, source, note) of a normalised word from lexicon and dictionary, or None.

    Tried in turn: the word itself; a word with apostrophes part by part ("dell'alba" = "dell'" + "alba");
    in French, Italian and Spanish the dictionary word that is spelt the same apart from accents."""
    def direct(k):
        if k in lex and all(p in ph2id for p in lex[k]):
            return list(lex[k]), 'lexicon'
        variants = mfa_dict(language).get(k)
        return (list(max(variants, key=lambda v: v[0])[1]), 'dict') if variants else None

    hit = direct(key)
    if hit:
        return hit + (None,)
    if "'" in key.strip("'"):
        parts = [direct(part) or direct(part.strip("'")) for part in re.split(r"(?<=')", key) if part.strip("'")]
        if parts and all(parts):
            return [p for ph, _ in parts for p in ph], 'dict', None
    if language in ACCENT_LANGS and key in _unaccented(language):
        word = _unaccented(language)[key]
        return direct(word) + (f'read as {word!r}',)
    return None


def with_suffix(ph, language):
    """Phonemes as the user writes them ('IH1' or 'IH1_en') -> phone-set entries; raises G2PError."""
    out = []
    for p in ph:
        p = unicodedata.normalize('NFC', p.strip())
        cand = p if p in ph2id else _clean(p, language) + LANG_SUFFIX[language]
        if cand not in ph2id or cand in REST:
            known = sorted(k[:-3] for k in ph2id if k.endswith(LANG_SUFFIX[language]))
            raise G2PError(f'unknown {language} phoneme {p!r}. The phonemes of this language: {" ".join(known)}')
        out.append(cand)
    return out


def _is_han(ch):
    return '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿'


def _zh_syllable(initial, final):
    final = ZH_FINAL_FIXES.get(final, final)
    phs = [p + '_zh' for p in (initial, final) if p]
    return phs if phs and all(p in ph2id for p in phs) else None


def _chinese(texts):
    """Phonemes per word of a Chinese line, or None where pypinyin gives nothing usable."""
    from pypinyin import Style, lazy_pinyin
    from pypinyin.contrib.tone_convert import to_finals, to_initials
    out = [None] * len(texts)
    han = [i for i, t in enumerate(texts) if t is not None and len(t) == 1 and _is_han(t)]
    if han:
        line = ''.join(texts[i] for i in han)
        initials = lazy_pinyin(line, style=Style.INITIALS, strict=True)
        finals = lazy_pinyin(line, style=Style.FINALS, strict=True)
        if len(initials) == len(han) == len(finals):
            for i, a, b in zip(han, initials, finals):
                out[i] = _zh_syllable(a, b)
    for i, t in enumerate(texts):
        if t is not None and out[i] is None and re.fullmatch(r'[a-zü:]+\d?', t.lower()):  # a pinyin syllable
            syl = re.sub(r'\d$', '', t.lower()).replace('u:', 'v').replace('ü', 'v')
            try:
                out[i] = _zh_syllable(to_initials(syl, strict=True), to_finals(syl, strict=True))
            except Exception:  # noqa: BLE001
                out[i] = None
    return out


def phonemize(score, use_lexicon=True, strict=True):
    """Phonemes of every word of a Score.

    Returns (list of phoneme lists, list of sources, warnings); a rest gives its own symbol and 'rest'.
    use_lexicon=False leaves the training lexicon out (to check the dictionaries and pypinyin alone).
    strict=False: a word without a pronunciation gets None instead of raising G2PError."""
    lang = score.language
    phs, src, warnings = [None] * len(score.words), [None] * len(score.words), []
    for i, w in enumerate(score.words):
        if w.is_rest:
            phs[i], src[i] = [w.text], 'rest'
        elif w.ph is not None:
            try:
                phs[i], src[i] = with_suffix(w.ph, lang), 'given'
            except G2PError as e:
                raise G2PError(f'words[{i}] ({w.text!r}): {e}') from None
    if lang == 'Chinese':
        zh = _chinese([w.text if phs[i] is None else None for i, w in enumerate(score.words)])
        for i, p in enumerate(zh):
            if p is not None:
                phs[i], src[i] = p, 'pinyin'
    lex = lexicon(lang) if use_lexicon else {}
    for i, w in enumerate(score.words):
        if phs[i] is not None:
            continue
        hit = lookup(norm_word(w.text), lang, lex) if lang != 'Chinese' or use_lexicon else None
        if hit:
            phs[i], src[i], note = hit
            if note:
                warnings.append(f'words[{i}] ({w.text!r}): {note}')
            if src[i] == 'dict' and lang in WEAK_DICT:
                warnings.append(f'words[{i}] ({w.text!r}): pronunciation {" ".join(p[:-3] for p in phs[i])} taken from the {lang} '
                                f'dictionary, which often differs from the training labels; check it or give "ph"')
            continue
        if not strict:
            continue
        hint = 'one character or one pinyin syllable per word' if lang == 'Chinese' else 'it is not in the training lexicon or the dictionary'
        raise G2PError(f'words[{i}]: no pronunciation for {w.text!r} ({lang}: {hint}). Give its phonemes in the score, '
                       f'e.g. {{"text": {json.dumps(w.text, ensure_ascii=False)}, "ph": [...], "notes": ...}}; '
                       f'the phonemes of the language are listed by: python scripts/sing.py --phonemes {lang}')
    return phs, src, warnings
