# Example scores

Sing one with `python scripts/sing.py --score examples/<file> --out samples/x.wav` (from `TCSinger2/`).
The score format is described at the top of `ldm/infer/score.py`.

## Notes and lyrics only (the model predicts the F0)

| file | what it is |
|---|---|
| `twinkle_en.json`, `twinkle_zh.json`, `twinkle_es.json` | hand-written, notes in beats with a tempo; the Spanish one gives the phonemes of one word |
| `clip_fr.json`, `clip_de.json`, `clip_it.json`, `clip_ru.json`, `clip_ja.json`, `clip_ko.json` | words and notes of a GTSinger validation clip (`scripts/clip_to_score.py`) |

## With an F0 curve (`"f0"` in the score; the output follows it)

The curves are the F0 of GTSinger validation clips, as preprocessing tracked it (RMVPE), 0 where unvoiced.
The notes are those of the same clip, so curve and score are on one time axis. Vibrato and glissando are
not part of the caption; they reach the model only through the curve.

| file | curve | words |
|---|---|---|
| `f0_vibrato_it.json` | an Italian bass singing a phrase with vibrato | phonemes and their lengths as annotated |
| `f0_plain_it.json` | the same singer and phrase, plain take | as above |
| `f0_glissando_zh.json` | a Chinese alto singing a phrase with glissando (slides between the notes) | text and notes only |
| `f0_plain_zh.json` | the same singer and phrase, plain take | text and notes only |
| `f0_glissando_en.json` | an English alto singing a phrase with glissando | text and notes only |
| `f0_plain_en.json` | the same singer and phrase, plain take | text and notes only |
| `f0_vibrato_ko_100fps.json` | a Korean tenor with vibrato; the curve has 100 values per second (`"f0_fps": 100`) instead of 187.5 | phonemes and their lengths as annotated |
| `clip_de_with_f0.json` | a German soprano; a third of the frames in its rests are voiced in the curve, which gives a warning | phonemes and their lengths as annotated |

```bash
python scripts/sing.py --score examples/f0_vibrato_it.json --out samples/vibrato.wav             # follows the curve
python scripts/sing.py --score examples/f0_vibrato_it.json --out samples/no_curve.wav --no_f0    # same score, F0 predicted
python scripts/sing.py --score examples/f0_vibrato_it.json --out samples/other.wav --prompt voice.wav   # the curve in another voice
```

`<out>.json` holds the curve that was sung as `"f0"`; `<out>_f0.png` shows it over the notes.
A score like these is written from any clip of the manifest by
`python scripts/clip_to_score.py --name "<item name>" --with_f0 [--with_durs] [--f0_fps 100] --out <file>`.
`python scripts/checks/check_f0_examples.py` sings all of them with and without their curve and measures how
closely the output follows it.
