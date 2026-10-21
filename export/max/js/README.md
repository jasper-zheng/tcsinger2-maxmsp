# The TCSinger2 front-end in Max

`[v8 tcs_max.js]` turns a score into the conditions `dict` that the two TCSinger2 `neural.gen~` objects take (`score2f0` and `sing`), and the caption for `neural.tokenizer`. It does what `scripts/score_to_max.py` does, without Python:
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
2. **Open a patch in this folder:** `tcsinger2.maxpat` (see "Singing in Max" below; `python scripts/build_max_patches.py` writes it). Max finds the modules next to the patch by name, and the data as `./data/…`. Max's `require` does not look into subfolders for a bare name, so the data is always required by its relative path. A patch elsewhere needs this folder in Max's file path (Options → File Preferences); whether `./data/…` then resolves is not checked yet.

## Messages

| message | what it does |
|---|---|
| `read <file.json>` | reads a score file and converts it |
| `dictionary <name>` | converts a score held in a Max `dict`, or the piano roll's notes (a `dict` with `"format": "tcs_notes"`; see below) |
| `bang` | converts the last score again, e.g. after a style change |
| `singer female alto`, `emotion sad`, `method bel canto`, `technique mix then falsetto` | style overrides, as the flags of `sing.py`; they apply from the next conversion |
| `caption "Singer: female alto, Emotion: sad"` | a whole caption instead. It must be quoted, because a comma splits a Max message |
| `style clear` | drops the overrides |
| `status` | posts the style overrides, the caption and the output name to the console |
| `out <name>` | the name of the output `dict` (default `tcs_cond`) |
| `write <file.json>` | writes the last result, with the caption, as `scripts/score_to_max.py` does |
| `phonemes <Language>` | lists the phonemes a score's `"ph"` may use |

**Outlets:**
- **Outlet 0:** `dictionary tcs_cond`, to both `neural.gen~` objects. The caption is not in it.
- **Outlet 1:** the caption, as one symbol, to `neural.tokenizer` through `[prepend prompt]`. It is one symbol because its commas would split a Max message.
- **Outlet 2:** `ok <pieces> <seconds>`, `warning <text>`, `error <text>` or `phonemes …`. After an error, outlets 0 and 1 send nothing, so the last good `dict` and caption stay.

The outlets send right to left: the status first, then the caption, then the `dict`.

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
| `tcs_notes.js` | the piano roll's notes ⇄ a score; makes notes singable, with warnings |
| `tcs_ui_model.js`, `tcs_ui_edit.js`, `tcs_ui_view.js`, `tcs_ui_draw.js`, `tcs_ui_state.js` | the piano roll: the notes and their edits, the mouse and keys, the axes, the drawing, the saved state |
| `tcs_ui.js` | the piano roll's Max object (`v8ui`) |

## Singing in Max (EWP7)

`tcsinger2.maxpat` puts the front-end together with the two models. It covers the two uses: notes only, and an F0 edited by hand.

**Setup, once:**
1. Build the data, as above.
2. Export the models (`export/export_f0.py`, `export/export_sing.py`, into `export/artifacts/`) and put them next to the patch, as git-ignored symlinks:
   ```
   cd export/max/js
   for f in ../../artifacts/tcsinger2_f0.{pte,json} ../../artifacts/tcsinger2_sing.{pte,json,tokenizer.json,tokenizer.config.json}; do ln -sf $f .; done
   ```
3. Make the patches: `python scripts/build_max_patches.py`. `*.maxpat` is git-ignored, so this script is their source.

**Using `tcsinger2.maxpat`:**
1. **Open a score:** the front-end sends its conditions `dict` to both objects, and its caption to `neural.tokenizer` → `sing`.
2. **`score2f0`:** `generate` writes the curve into `buffer~ tcs_f0`, as MIDI pitch at 50 Hz with 0 = unvoiced, then sends `init f0 tcs_f0` to `sing`. With the toggle on, `sing` runs straight after (notes only).
3. **Edit the curve (optional):** `mode draw` in the `waveform~`, then draw.
4. **`sing`:** `generate` writes `buffer~ tcs_out` at 48 kHz. `start` plays it and `writewave` saves it.

**Times on an M4 Max:**
- `score2f0`: about 0.37 s;
- `sing`: about 0.9 s;
- the first call after loading: a few seconds more.

**Memory:** about 20 GB while generating. MLX keeps that memory after a patch closes and reuses it, so it goes back to the system only when Max quits.

**The check:** the export's checks against Python, in Max as well, are in `export_check/`, which is not part of the repository.

**Pitfall:** the front-end's `dict` must go straight to its receivers. Through a `[t a …]` it arrived empty, and the objects then ran on zero conditions. Max sends an outlet's messages right to left by position.

## The piano roll (UI_WORK_PACKAGES.md)

`tcsinger2_ui.maxpat` (`python scripts/build_max_patches.py ui`) is `tcsinger2.maxpat` with `[v8ui tcs_ui.js]`, a piano roll, in front of the front-end.
- **Showing and editing:** the roll shows the notes, their lyrics, the rests (AP/SP), the phonemes the front-end gave each word and the F0 curve, and edits them by mouse and keys (below). The lyrics are typed in the line under the roll (below).
- **Sending them:** it sends its notes as `dictionary tcs_notes` to `tcs_max.js` after every edit, and shows the front-end's status and warnings, which come back through `[prepend fe]`.
- **The roll checks nothing:** notes may overlap and a lyric may be anything. `tcs_max.js` makes the notes singable (`tcs_notes.js`) and says what it changed, one console warning per kind:
  - of two overlapping notes, the later one wins and the earlier is cut where it starts (a note covered entirely is left out);
  - pitches are rounded into 1–98;
  - a note with no lyric becomes a rest;
  - a `-` with no note right before it is sung as a word of its own (the sung word before it, else the language's fallback syllable);
  - a word with no pronunciation in the language (a typo, or `la` in Russian) is sung as the language's fallback syllable: `la`, or ら in Japanese, 라 in Korean, да in Russian;
  - the phrase is cut at 20 s and at 256 phoneme pieces.

  A score file read straight into `tcs_max.js` (`read`, or a score `dict`) is still refused when it breaks these limits, or has a word with no pronunciation. For the notes, each of the result's `words` also has `ids`, the note ids it came from, and a fallback word has `"source": "fallback"` and `typed`, what the note said.
- **The phonemes:** the roll draws each word's phonemes above its notes, from the result's `words` (it reads `dict tcs_cond` on the front-end's `ok`): grey; amber when they come from the dictionary rather than the training lexicon; red, with a line under the notes, for a word sung as the fallback. Hovering over a note shows them in the status line. The toolbar's `ph` button (or `phonemes 0|1`) hides them.
- **The curve:** after `score2f0`, the patch sends `f0 refresh` to the roll. A curve from before the last change is drawn faded.
- **The "regenerate F0 after each change" toggle** (off by default) runs `score2f0` whenever the front-end has sent new conditions.
- **Saving:** the roll saves its score and both curves in its box when the patch is saved. When the patch is opened again, the roll writes the curve back into `buffer~ tcs_f0`, sends `init f0 tcs_f0` to `sing` (its outlet 1) and sends the score. No `score2f0` runs, so the reopened patch sings the same.
- **Presets:** `getvalueof` and `setvalueof` carry the same state for `[pattr … @bindto <the roll's scripting name>]`.

**Editing** (click the roll first, so it gets the keys; keys reach it only in a locked patch). The toolbar at the bottom has the select and pencil tools, the snap, undo, redo and fit.

| | does |
|---|---|
| select tool | click a note to select it (Shift adds or removes); drag it to move the selection (snapped; Cmd: free); drag an edge to resize, and where another note touches that edge both follow (Alt: only this note); drag on empty space for a rubber band; double-click empty space for a new note |
| pencil tool | click or drag on empty space for a new note; on a note it acts as the select tool |
| a rest's label | click it to switch AP ↔ SP |
| Backspace | delete the selection |
| ↑ ↓, Shift | a semitone, an octave |
| ← →, Shift | a snap step, a bar |
| Cmd-A, Esc | select all; select none (or cancel a drag) |
| Cmd-Z, Cmd-Shift-Z | undo, redo |
| Cmd-C, Cmd-X, Cmd-V, Cmd-D | copy, cut, paste (at the pointer), duplicate |
| B, S, J | pencil ↔ select; split the note under the pointer; join the selected notes |

Every edit is also a message (`tool`, `select`, `add`, `delete`, `move`, `resize`, `split`, `join`, `rest`, `undo`, `redo`, `copy`, `cut`, `paste`, `duplicate`); they are listed at the top of `tcs_ui.js`. `limit <seconds>` and `range <lo> <hi>` set the shaded limit and the pitch rows.

**Lyrics** (U5): `[textedit @outputmode 1 @keymode 1]` under the roll, cabled both ways (its `text <line>` into the roll; the roll's outlet 3 sends it `set <line>` and `select`).
- **Which notes:** the line fills the selected notes in time order, or all the notes when none is selected. The box always shows the lyrics of exactly those notes, so a line can be edited rather than retyped, and Enter fills them (one undo step).
- **Splitting:** with a `;` in the line, it is split at the `;`s only (`one two;three` is two lyrics; an empty entry, `a;;b`, leaves its note as it is). Without one, it is split at the spaces, and Chinese, Japanese and Korean characters are one lyric each, small kana too, as in the training lexicon (`一闪一闪亮晶晶`, `きょう` → き ょ う). `-` continues the word before it.
- **Double-click a note:** it selects that note, so the line starts there, and selects the box's text (click the box if typing doesn't go there).
- Entries beyond the notes the line fills are dropped, with a message in the status line.

## Parity test

The JS front-end equals the Python one on 3,496 scores (EWP3). That test is in `export_check/` with the export's other checks, which are not part of the repository.
