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
2. **Open a patch in this folder:** `tcsinger2_demo.maxpat`, the piano roll with the models (see "The demo patch" below), or `tcsinger2.maxpat`, the front-end and the models alone (see "Singing in Max"). `python scripts/build_max_patches.py` writes them. Max finds the modules next to the patch by name, and the data as `./data/…`. Max's `require` does not look into subfolders for a bare name, so the data is always required by its relative path. A patch elsewhere needs this folder in Max's file path (Options → File Preferences); whether `./data/…` then resolves is not checked yet.

## Messages (`tcs_max.js`)

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
| `tcs_ui_model.js`, `tcs_ui_edit.js`, `tcs_ui_view.js`, `tcs_ui_draw.js`, `tcs_ui_state.js`, `tcs_ui_f0.js` | the piano roll: the notes and their edits, the mouse and keys, the axes, the drawing, the saved state, the F0 curve's edits |
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
3. Make the patches: `python scripts/build_max_patches.py [main] [ui] [demo]` (`main` and `ui` when none is given). `*.maxpat` is git-ignored, so this script is their source. It overwrites a patch saved from Max, with whatever the roll had saved in it.

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

## The demo patch (UWP9)

`tcsinger2_demo.maxpat` (`python scripts/build_max_patches.py demo`) is the least the piano roll needs to sing: the roll and its lyrics line, the front-end, `score2f0`, `sing` and playback. It is laid out in the patching view as numbered sections, and opens with `examples/twinkle_en.json` in the roll (the builder embeds it as the roll's saved state, made by `node` from the roll's own modules; without `node` the roll starts empty). It needs the setup of "Singing in Max": the data and the model files.

1. **The notes:** open a score (with the language menu next to it), or draw notes with the pencil or a double-click, and type their lyrics in the line under the roll. The roll sends them to the front-end, whose status and phonemes come back to it. The style buttons on the right change the caption; bang after one.
2. **The F0:** `generate` runs `score2f0`; the roll reads the curve (`f0 refresh`) and `sing` is told (`init f0 tcs_f0`). Edit the curve in the roll's F0 mode (F), or with the F0 row under the lyrics line.
3. **Sing:** `generate` writes `buffer~ tcs_out`. Sing again after editing the curve; after editing the notes, generate the F0 first.
4. **Play:** turn the audio on (the `ezdac~`), then click the roll's ruler, or press Space with the roll focused, or use `play`/`stop`.

Save the patch to keep the roll's notes and curves in it. Left out, compared with `tcsinger2_ui.maxpat`: seeds, `cfg_scale`, the `waveform~`s, the tempo box, writing the score, the "regenerate" and "sing when ready" toggles, `writewave`.

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
- **The curve:** after `score2f0`, the patch sends `f0 refresh` to the roll, which keeps it as the generated curve and as the working curve, the one it edits (below). A curve from before the last change of the notes is drawn faded ("F0 stale" in the status line). A new `score2f0` replaces both curves, so hand edits are lost; the note edits stay undoable.
- **The "regenerate F0 after each change" toggle** (off by default) runs `score2f0` whenever the front-end has sent new conditions.
- **Saving:** the roll saves its score and both curves (with the hand edits) in its box when the patch is saved. When the patch is opened again, the roll writes the curve back into `buffer~ tcs_f0`, sends `init f0 tcs_f0` to `sing` (its outlet 1) and sends the score. No `score2f0` runs, so the reopened patch sings the same.
- **Presets:** `getvalueof` and `setvalueof` carry the same state for `[pattr … @bindto <the roll's scripting name>]`.

**Editing** (click the roll first, so it gets the keys; keys reach it only in a locked patch). The toolbar at the bottom has the F0 button (the mode), the select and pencil tools, the snap, undo, redo, fit and ph.

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
| F | the F0 mode (below) |
| Space | play from the playhead (or the start), or stop |

Every edit is also a message (see "The roll's messages" below).

**The F0 curve** (UWP6): **F**, or the toolbar's F0 button, switches the roll to the F0 mode, where the mouse edits the curve and the notes can't be edited (they are dimmed; F switches back). Its toolbar has the curve's tools:

| | does |
|---|---|
| draw (D) | drag to draw the curve (it voices the frames it crosses); Shift: whole semitones |
| line (L) | a straight line from where the mouse goes down to where it comes up; Shift: from semitone to semitone |
| smooth (S) | drag over the curve to smooth it (each pass a little more) |
| erase (E) | drag over the curve to put the generated curve back |
| unvoice (U) | drag over the curve to make it unvoiced |
| Cmd-Z, Cmd-Shift-Z, Cmd-A, Esc | undo, redo (one history for the notes and the curve); select all notes; cancel a drag or select none |

The hand-edited parts are drawn in blue, over a faint line of the generated curve. Each drag is one undo step, and after it the roll writes the curve into `buffer~ tcs_f0` and sends `init f0 tcs_f0` to `sing` (outlet 1), so the next `sing` follows it. Hovering shows the curve's value and the generated one.

The patch's F0 row under the lyrics line has the range commands: `f0 snap` (the notes' pitches), `f0 reset` (the generated curve), `f0 unvoice`, `f0 smooth`, `f0 offset -1`/`1` (semitones, voiced frames only) and `f0 vibrato <rate> <depth> <start> <end>` (Hz, semitones and seconds, from its four number boxes). They act on the selected notes' spans, or the whole curve when none is selected. With a range in seconds at the end (`f0 offset 2 1 2`: +2 semitones over 1–2 s; `f0 vibrato 5.5 0.5 2 3`: a vibrato from 2 s up to 3 s, its phase starting at 2 s) they act on that range; a range that doesn't end after it starts (the vibrato boxes' `0 0`) counts as none; `f0 draw <s> <midi>…` and `f0 line <s0> <m0> <s1> <m1>` draw by message. The curve's times are seconds; the note messages' are beats.

**Playback** (UWP7): click the ruler to play `buffer~ tcs_out` from that point, or press Space to play from the playhead (from the start when it is at the end) and to stop; a click on the ruler while playing jumps there. Playback runs to the phrase's end plus half a second. The patch's `play` and `stop` buttons do the same. The roll sends `play <from_ms> <to_ms> <dur_ms>` or `stop` out of its outlet 4. The patch drives `[play~ tcs_out]` with a `[line~]` of positions, fades the output in and out over 5 ms, and sends the same position back as `playhead <s>` (`[snapshot~ 30]`), with `stopped` at the end. So the playhead is where `play~` reads. It stays where playback stopped, marked on the ruler, and is redrawn only when it moves. The audio must be on.

**Lyrics** (U5): `[textedit @outputmode 1 @keymode 1]` under the roll, cabled both ways (its `text <line>` into the roll; the roll's outlet 3 sends it `set <line>` and `select`).
- **Which notes:** the line fills the selected notes in time order, or all the notes when none is selected. The box always shows the lyrics of exactly those notes, so a line can be edited rather than retyped, and Enter fills them (one undo step).
- **Splitting:** with a `;` in the line, it is split at the `;`s only (`one two;three` is two lyrics; an empty entry, `a;;b`, leaves its note as it is). Without one, it is split at the spaces, and Chinese, Japanese and Korean characters are one lyric each, small kana too, as in the training lexicon (`一闪一闪亮晶晶`, `きょう` → き ょ う). `-` continues the word before it.
- **Double-click a note:** it selects that note, so the line starts there, and selects the box's text (click the box if typing doesn't go there).
- Entries beyond the notes the line fills are dropped, with a message in the status line.

### The roll's messages

Beats for the notes, seconds for the curve and playback. Without ids, the note edits act on the selection.

| message | what it does |
|---|---|
| **Scores** | |
| `read <file.json>` | reads a score file (the format of `ldm/infer/score.py`); the view fits it |
| `dictionary <name>` | a score held in a Max `dict` |
| `write <file.json>` | writes the notes as a score, made singable as `tcs_max.js` does |
| `language <L>` | the notes' language |
| `tempo <bpm>` | the tempo; the notes keep their beats, so their seconds change |
| `bang` | sends the notes again |
| `clear` | no notes |
| **The front-end** | |
| `fe ok\|warning\|error …` | the front-end's status, from `tcs_max.js`'s last outlet through `[prepend fe]`; on `ok` the roll reads the words and phonemes from `dict tcs_cond` |
| `phonemes 0\|1` | hides or shows the phonemes (the toolbar's ph) |
| **Lyrics** | |
| `text <line>` | from the `[textedit]`: fills the selected notes, or all with none selected (see "Lyrics" above) |
| **The curve** | |
| `f0 refresh` | reads the curve from the buffer, after `score2f0`: it becomes the generated and the working curve; hand edits are dropped |
| `f0 buffer <name>` | the buffer to read and write (default `tcs_f0`) |
| `f0 clear` | forgets the curve |
| `f0 offset <st> [<s0> <s1>]` | semitones up or down, voiced frames only |
| `f0 vibrato <rate> <depth> [<s0> <s1>]` | Hz and semitones, voiced frames only; the phase starts at each span's start |
| `f0 unvoice\|reset\|snap\|smooth [<s0> <s1>]` | unvoiced; the generated curve back; the notes' pitches; smoothed |
| `f0 draw <s> <midi> <s> <midi>…` | draws through the points |
| `f0 line <s0> <m0> <s1> <m1>` | a straight line |
| **Editing** | |
| `mode notes\|f0` | what the mouse edits (the F0 button, F) |
| `tool select\|pencil\|draw\|line\|smooth\|erase\|unvoice` | the mouse's tool, and with it the mode |
| `select <ids…>\|all\|none` | the selection |
| `add <start> <dur> <midi> [lyric]` | a new note |
| `delete [ids]` | deletes notes |
| `move <dBeats> <dMidi>` | moves the selection |
| `resize <id> <start> <end>` | a note's start and end |
| `split <id> <beat>` | splits a note |
| `join [ids]` | joins notes into one |
| `rest <beat> [AP\|SP\|toggle]` | the type of the rest at a beat |
| `undo`, `redo` | one history for the notes and the curve |
| `copy`, `cut`, `paste [beat]`, `duplicate` | the clipboard; `paste` at a beat, else at the pointer |
| **The view** | |
| `view fit` | the whole phrase |
| `view <beat0> <beat1> <midiLo> <midiHi>` | a range |
| `zoom <fx> <fy>` | zooms by these factors |
| `scrollby <beats> <rows>` | scrolls |
| `snap 1/1\|1/2\|1/4\|1/8\|1/16\|off` | the grid |
| `bar <beats>` | the beats in a bar (4) |
| `limit <seconds>` | the shaded limit (20 s), display only |
| `range <lo> <hi>` | the pitch rows (1–98), display only |
| **Playback** | |
| `play [<s>]` | plays `buffer~ tcs_out` from `<s>`, else from the playhead (or the start), to the phrase's end plus 0.5 s |
| `stop` | stops |
| `stopped` | from the patch: the playback reached its end |
| `playhead <s>` | from the patch: the playback position |
| `playhead off` | hides the playhead |
| **Inspection** | |
| `snapshot <file.png> [scale]` | draws the roll into an image (scale 2) |
| `stats [reset]` | the paint times, out of outlet 2 |
| `report` | a digest of the state, out of outlet 2 |

The roll also takes Max's own messages to a `v8ui`: the mouse and keys, and `getvalueof`/`setvalueof` for `pattr` (above).

### Inlet and outlets

**Inlet:** every message above.

| outlet | sends | cabled to |
|---|---|---|
| 0 | `dictionary tcs_notes`, after every committed edit of the notes, after a load and after a restore | `[v8 tcs_max.js]` |
| 1 | `bang` when the roll has written the curve into the buffer itself: a curve edit, its undo, a restore | the `init f0 tcs_f0` message into `sing` |
| 2 | the status: `ok <notes> <seconds>`, `warning <text>`, `error <text>`, `message <text>`, `restored embed\|pattr <notes>`, `stats <json>`, `report <json>`, `snapshot <file>` | nothing needed (`print`, or a check) |
| 3 | `set <line>` (the lyrics of the notes the line fills, whenever they change) and `select` (after a double-click on a note) | the lyrics `[textedit @outputmode 1 @keymode 1]`, whose outlet 0 goes back into the roll |
| 4 | `play <from_ms> <to_ms> <dur_ms>` and `stop` | `[route play stop]` → `[line~]` → `[play~ tcs_out]`; the position comes back as `playhead <s>` (`[snapshot~ 30]`), and `stopped` at the end |

The status line at the roll's bottom shows the same status, the front-end's, the hovered note or curve value, "F0 stale" and the playback position.

## Parity test

The JS front-end equals the Python one on 3,496 scores (EWP3). That test is in `export_check/` with the export's other checks, which are not part of the repository.
