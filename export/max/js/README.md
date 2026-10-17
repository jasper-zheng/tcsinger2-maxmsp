# The TCSinger2 front-end in Max

`[v8 tcs_max.js]` turns a score into the conditions `dict` that the two TCSinger2 `neural.gen~` objects take (`score2f0` and `sing`). It does what `scripts/score_to_max.py` does, without Python:
- the score parser;
- g2p for the 9 languages, with the training lexicon, the MFA dictionaries and pypinyin's readings;
- syllables onto notes;
- the pieces, the conditions and the caption.

The models predict the lengths themselves, so no duration model runs here. See EWP3 in `EXPORTING_WORK_PACKAGES.md`.

## Setup

1. **Build the data** once, from the repository root, in `.venv`:
   ```
   python scripts/build_max_js_data.py
   ```
   It writes `data/tcs_data_*.js`: 58 MB, generated and not committed. It needs `data/mfa_dict/` (`scripts/setup_env_mac.sh`) and pypinyin.
2. **Open a patch in this folder:** `tcs_frontend_test.maxpat` (pick a score, try the style messages) or `tcs_probe.maxpat` (what Max's v8 supports). Max finds the modules next to the patch by name, and the data as `./data/…`. Max's `require` does not look into subfolders for a bare name, so the data is always required by its relative path. A patch elsewhere needs this folder in Max's file path (Options → File Preferences); whether `./data/…` then resolves is not checked yet.

## Messages

| message | what it does |
|---|---|
| `read <file.json>` | reads a score file and converts it |
| `dictionary <name>` | converts a score held in a Max `dict` |
| `bang` | converts the last score again, e.g. after a style change |
| `singer female alto`, `emotion sad`, `method bel canto`, `technique mix then falsetto` | style overrides, as the flags of `sing.py`; they apply from the next conversion |
| `caption "Singer: female alto, Emotion: sad"` | a whole caption instead. It must be quoted, because a comma splits a Max message |
| `style clear` | drops the overrides |
| `status` | posts the style overrides, the caption and the output name to the console |
| `out <name>` | the name of the output `dict` (default `tcs_cond`) |
| `write <file.json>` | writes the last result |
| `phonemes <Language>` | lists the phonemes a score's `"ph"` may use |

**Outlets:**
- **Outlet 0:** `dictionary tcs_cond`, to both `neural.gen~` objects. Its `caption` goes to `neural.tokenizer`, e.g. through `[dict.unpack caption:]`.
- **Outlet 1:** `ok <pieces> <seconds>`, `warning <text>`, `error <text>` or `phonemes …`. After an error, outlet 0 sends nothing, so the last good `dict` stays.

**Loading:** each language's dictionary is loaded on first use and kept while the object lives; the console shows the load time. Russian is the largest (22 MB, 0.2 s in Node).

## Differences from the Python file

- **Conditions, caption, names and warnings:** identical to Python's.
- **`meta.frontend`:** `"js"`.
- **`pieces[].start` and `dur`:** `null` where the model predicts the timing. Word boundaries, rests and `ph_durs` pieces have their times.
- **Error messages** point to the `phonemes` message instead of `sing.py --phonemes`.
- **Number kinds:** JSON numbers lose their int/float kind in JS. A `notes_per_syllable` entry of `2.0` is accepted here, while Python refuses it.

## Modules

They are CommonJS modules and require each other by bare name (`require("tcs_g2p.js")`), as Max resolves them. Max's `require` returns a new copy of a module on every call, so no module keeps state of its own. The data cache of `tcs_data.js` lives on `globalThis`, and every data file loads once. Only `tcs_max.js` uses Max's globals, so the others also run under Node.

| file | port of |
|---|---|
| `tcs_score.js` | `ldm/infer/score.py` `parse_score` |
| `tcs_g2p.js` | `ldm/infer/g2p.py` |
| `tcs_pinyin.js` | `g2p._chinese`, pypinyin's segmenter (`mmseg`) and `to_initials`/`to_finals` |
| `tcs_pieces.js` | `ldm/infer/timing.py` `build_pieces` |
| `tcs_caption.js` | `ldm/data/captions.py` |
| `tcs_conditions.js` | `scripts/score_to_max.py` `conditions` |
| `tcs_frontend.js` | all of the above: score → result |
| `tcs_data.js`, `tcs_util.js` | data loading; Python's `round`, `repr` and `searchsorted` |
| `tcs_max.js` | the Max object |

## Parity test

```
python scripts/checks/make_frontend_corpus.py          # .venv: 3,496 scores with Python's results
node export/max/js/test/parity.js export/artifacts/frontend_corpus.jsonl
```

**What it checks:**
- Every score must give the same conditions, caption, meta, words and pieces as Python.
- A failing score must fail with the same error class at the same `words[i]`.
- Display times may differ by one frame. A boundary exactly on the rounding edge (0.49 frame) can round either way with Python's predicted lengths in the sum.

**Python version:** the port adds floats left to right, like `sum()` in Python 3.11 (`.venv`). Python 3.12 and later use a compensated float `sum`, which could move a boundary in the last bit.
