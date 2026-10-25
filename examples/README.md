# Example scores

Sing one with `python scripts/sing.py --score examples/<file> --out samples/x.wav` (from `TCSinger2/`).

The score format:

```json
    {"language": "English", "tempo": 112,
     "style": {"singer": "female alto", "emotion": "happy", "singing_method": "pop", "technique": "breathy"},
     "words": [
       {"rest": 1.0},
       {"text": "where", "notes": [["C4", 1]]},
       {"text": "wind",  "notes": [["A#3", 2], [60, 1]]},
       {"rest": 0.5, "type": "AP"}],
     "f0": [0, 0, 0, 261.4, 262.0, 263.1, ...]}
```


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
