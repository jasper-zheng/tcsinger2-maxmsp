"""Style captions: the single place that knows the caption format.

    Singer: female alto, Emotion: happy, Singing Method: pop, Technique: mix then falsetto

  field            values
  Singer           <gender> <vocal range>: {female, male} x {soprano, alto, tenor, bass},
                   or <gender> alone when the vocal range is not known (OpenSinger, Opencpop)
  Emotion          happy | sad
  Singing Method   pop | bel canto
  Technique        none | X | X and Y | A then B, over mix, falsetto, breathy, pharyngeal

Fields come in this order, joined by ', '; an unspecified field is left out. The empty string
is the unconditional caption (classifier-free guidance).

Technique describes the clip in two halves of its sung time (rests excluded). A technique counts
in a half when it covers at least HALF_COVERAGE of it. The same set in both halves gives 'X'
(several as 'X and Y' in the order above, none as 'none'); different sets give 'A then B'.

Left out on purpose: the singer ID, the language (a separate model input), and everything the
input F0 and timings already fix (vibrato, glissando, pace, pitch range).
"""
import numpy as np

REST = ('<AP>', '<SP>')
TIMBRE_TECHS = ['mix', 'falsetto', 'breathy', 'pharyngeal']
GENDERS = ['female', 'male']
VOCAL_RANGES = ['soprano', 'alto', 'tenor', 'bass']
EMOTIONS = ['happy', 'sad']
SINGING_METHODS = ['pop', 'bel canto']
FIELDS = {'singer': 'Singer', 'emotion': 'Emotion', 'singing_method': 'Singing Method', 'technique': 'Technique'}
HALF_COVERAGE = 0.5


def _tech_set_text(techs):
    return ' and '.join(techs) if techs else 'none'


def technique_text(ph, ph_durs, flags, max_duration=None):
    """Technique value of a clip. flags: {technique: per-phoneme 0/1 list}; max_duration cuts the clip."""
    durs = np.asarray(ph_durs, dtype=float)
    ends = np.cumsum(durs)
    starts = ends - durs
    if max_duration is not None:
        ends, starts = np.minimum(ends, max_duration), np.minimum(starts, max_duration)
    w = (ends - starts) * np.array([p not in REST for p in ph])  # sung seconds per phoneme
    if w.sum() <= 0:
        return 'none'
    cum = np.cumsum(w)
    first = np.clip(np.minimum(cum, w.sum() / 2) - (cum - w), 0, None)  # the phoneme at the midpoint is split
    halves = []
    for part in (first, w - first):
        halves.append([t for t in TIMBRE_TECHS
                       if (part * (np.asarray(flags[t]) > 0)).sum() >= HALF_COVERAGE * part.sum()])
    if halves[0] == halves[1]:
        return _tech_set_text(halves[0])
    return f'{_tech_set_text(halves[0])} then {_tech_set_text(halves[1])}'


def parse_technique(text):
    """'mix then falsetto and breathy' -> [['mix'], ['falsetto', 'breathy']]; raises ValueError if malformed."""
    parts = []
    for part in text.split(' then '):
        techs = [] if part == 'none' else part.split(' and ')
        if techs != [t for t in TIMBRE_TECHS if t in techs]:  # unknown name, duplicate or wrong order
            raise ValueError(f'bad technique {text!r}: use none or {TIMBRE_TECHS} in this order, '
                             f"joined by ' and ', with at most one ' then '")
        parts.append(techs)
    if len(parts) > 2 or (len(parts) == 2 and parts[0] == parts[1]):
        raise ValueError(f"bad technique {text!r}: at most two different parts around ' then '")
    return parts


def _check(name, value, allowed):
    if value not in allowed:
        raise ValueError(f'bad {name} {value!r}: expected one of {allowed}')


def build_caption(singer=None, emotion=None, singing_method=None, technique=None):
    """Caption from field values; None leaves a field out. singer is '<gender> <vocal range>' or '<gender>'."""
    if singer is not None:
        gender, sep, vocal_range = singer.partition(' ')
        _check('singer gender', gender, GENDERS)
        if sep:
            _check('singer vocal range', vocal_range, VOCAL_RANGES)
    if emotion is not None:
        _check('emotion', emotion, EMOTIONS)
    if singing_method is not None:
        _check('singing method', singing_method, SINGING_METHODS)
    if technique is not None:
        parse_technique(technique)
    values = dict(singer=singer, emotion=emotion, singing_method=singing_method, technique=technique)
    return ', '.join(f'{FIELDS[k]}: {v}' for k, v in values.items() if v is not None)


def parse_caption(caption):
    """Inverse of build_caption: {field: value} of the fields present."""
    labels = {label: key for key, label in FIELDS.items()}
    out = {}
    for part in caption.split(', ') if caption else []:
        label, sep, value = part.partition(': ')
        if not sep or label not in labels or labels[label] in out:
            raise ValueError(f'bad caption {caption!r}')
        out[labels[label]] = value
    return out


def row_fields(row):
    """Field values of a manifest row; labels that are missing or 'no' become None."""
    def get(key):
        value = row[key]
        return None if not isinstance(value, str) or value in ('', 'no') else value
    gender, vocal_range = get('gender'), get('vocal_range')
    return dict(singer=f'{gender} {vocal_range}' if gender and vocal_range else gender, emotion=get('emotion'),
                singing_method=get('singing_method'), technique=get('technique'))


def caption_from_row(row):
    """Full caption of a manifest row."""
    return build_caption(**row_fields(row))


def sample_caption(row, rng=np.random, field_drop=0.1, caption_drop=0.2):
    """Training caption: empty with caption_drop, otherwise each field is left out with field_drop."""
    if rng.random() < caption_drop:
        return ''
    return build_caption(**{k: v if rng.random() >= field_drop else None for k, v in row_fields(row).items()})
