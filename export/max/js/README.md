# The TCSinger2 front-end in Max

The Max front-end includes three Max JS scripts, they are meant to prepare the inputs for the two TCSinger2 `neural.gen~` objects (`score2f0` and `sing`) from a score file or from a piano roll GUI:

 - `[v8 tcs_max.js]`: Process a score file or a score `dict` to the conditions `dict`, for the F0 / singing models in `neural.gen~`.
 - `[v8ui tcs_ui.js]`: The piano roll GUI with notes and lyrics. For manually entering score and editing the F0 curve. It converts its own notes to the conditions `dict` for the F0 / singing models in `neural.gen~`.
 - `[v8 tcs_prompt.js]`: The prompt for the singing model, from style prompt to `neural.tokenizer`.

Demo Max4Live device: [`TCSinger2_UI.amxd`](./TCSinger2_UI.amxd)

## `[v8 tcs_max.js]`

| message | what it does |
|---|---|
| `read <file.json>` | reads a score file and converts it |
| `dictionary <name>` | converts a score held in a Max `dict`, or a `dict` of piano-roll notes (`"format": "tcs_notes"`, `tcs_notes.payload`) |
| `bang` | converts the last score again and sends the `dict` |
| `status` | posts the output name and the last score to the console |
| `out <name>` | the name of the output `dict` (default `tcs_cond`) |
| `write <file.json>` | writes the last result, with the caption the score's own style gives, as `scripts/score_to_max.py` does (that caption is not sent anywhere) |
| `phonemes <Language>` | lists the phonemes a score's `"ph"` may use |

**Outlets:**
- **Outlet 0:** `dictionary tcs_cond`, to both `neural.gen~` objects. There is no caption in it.
- **Outlet 1:** `ok <pieces> <seconds>`, `warning <text>`, `error <text>` or `phonemes …`. After an error, outlet 0 sends nothing, so the last good `dict` stays.



## `[v8ui tcs_ui.js]` (Piano Roll GUI)

### UI

**Note editing mode**: 

| Tools | What they do |
|---|---|
| select tool | click a note to select it (Shift adds or removes); drag it to move the selection (snapped; Cmd: free); drag an edge to resize, and where another note touches that edge both follow (Alt: only this note); drag on empty space for a rubber band; double-click empty space for a new note |
| pencil tool | click or drag on empty space for a new note; on a note it acts as the select tool |
| a rest's label | click it to switch AP (Aspiration Pause) and SP (Silence Pause) |
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

**The F0 curve mode:** Click the toolbar's F0 button (or press F) switches the roll to the F0 mode:

| Tools | What they do |
|---|---|
| draw (D) | drag to draw the curve (it voices the frames it crosses); Shift: whole semitones |
| line (L) | a straight line from where the mouse goes down to where it comes up; Shift: from semitone to semitone |
| smooth (S) | drag over the curve to smooth it (each pass a little more) |
| erase (E) | drag over the curve to put the generated curve back |
| unvoice (U) | drag over the curve to make it unvoiced |
| vibrato button (V) | adds a vibrato of 0.05 semitone at 5.5 Hz over the range strip's range (none: the selected notes, or the whole curve). Clicks add up (3 clicks: 0.15 semitone), each one undo step; the status line says the depth so far |
| offset arrows (▲ ▼, right of the vibrato button) | move the voiced frames over the same range up or down by 0.2 semitone per click (fixed). Each click is one undo step; the status line says the total so far |
| clear button (a wave and ×, right of the arrows) | clear the generated and the working curves |
| Cmd-Z, Cmd-Shift-Z, Cmd-A, Esc | undo, redo (one history for the notes and the curve); select all notes; cancel a drag or select none |

**Edit F0 by message:** `f0 snap|reset|unvoice|smooth [<s0> <s1>]`, `f0 offset <st> [<s0> <s1>]`, `f0 vibrato <rate> <depth> [<s0> <s1>]`. `f0 draw <s> <midi>…` and `f0 line <s0> <m0> <s1> <m1>`.

**Syncing to a host** (e.g. Ableton): send `sync <beats>` with the host's playhead, and that beat becomes the roll's 0. After that, send the host's playheads as `seek <beats>` or `play <beats>`, both in the host's beats. After `sync 4`, `seek 4.5` puts the playhead 0.5 beat into the phrase, and `seek 2` puts it -2 beats before the start. 


### Messages

| message | what it does |
|---|---|
| **Scores** | |
| `read <file.json>` | reads a score file (the format of `ldm/infer/score.py`); the view fits it. The roll keeps its own tempo (120 bpm in a new roll): the notes keep their beats, and the score's `tempo` is not used, so a score written at another tempo sings faster or slower. Send `tempo` before the read to hear it at its own tempo (60 bpm for a score with none). Its `"f0"`, if any, becomes the roll's curve (see "The curve") |
| `dictionary <name>` | a score held in a Max `dict` (the roll's tempo kept, as `read`) |
| `write <file.json>` | writes the notes as a score, made singable as the conversion makes them, with the working curve as its `"f0"` (as outlet 4) |
| `writecond <file.json>` | writes the last conversion, with the caption the score's style gives (as `scripts/score_to_max.py`) |
| `out <name>` | the name of the conditions `dict` (default `tcs_cond`) |
| `outscore <name>` | the name of the score `dict` of outlet 4 (default `tcs_score`) |
| `language <L>` | the notes' language |
| `tempo <bpm>` | the tempo; the notes keep their beats, so their seconds change |
| `bang` | converts the notes again and sends the `dict` |
| `clear` | no notes |
| **The phonemes** | |
| `phonemes 0\|1` | hides or shows the phonemes (the toolbar's ph) |
| `phonemes <Language>` | lists the phonemes a score's `"ph"` may use, out of outlet 2 |
| **Lyrics** | |
| `text <line>` | from the `[textedit]`: fills the selected notes, or all with none selected (see "Lyrics" above) |
| **The curve** | |
| `f0 refresh` | reads the curve from the buffer, after `score2f0`: it becomes the generated and the working curve; hand edits are dropped |
| `f0 buffer <name>` | the buffer to read and write (default `tcs_f0`) |
| `f0 clear` | forgets both curves (the F0 toolbar's clear button); the buffer is left as it is |
| `f0 offset <st> [<s0> <s1>]` | semitones up or down, voiced frames only |
| `f0 vibrato <rate> <depth> [<s0> <s1>]` | Hz and semitones, voiced frames only; the phase starts at each span's start |
| `f0 unvoice\|reset\|snap\|smooth [<s0> <s1>]` | unvoiced; the generated curve back; the notes' pitches; smoothed |
| `f0 draw <s> <midi> <s> <midi>…` | draws through the points |
| `f0 line <s0> <m0> <s1> <m1>` | a straight line |
| `region <s0> <s1>`, `region clear` | the range strip's range, in seconds (also a drag on the strip; a click clears it) |
| `vibrato` | the F0 toolbar's vibrato button: +0.05 semitone at 5.5 Hz over the region (none: the selected notes, or the whole curve) |
| `offset <st>` | the F0 toolbar's offset arrows: the voiced frames over the region (none: as above) move by `<st>` semitones (the arrows: ±0.2) |
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
| `root <note>` | the key's root, display only: a name (`C`, `C#`, `Db` … `B`, any case) or a number mod 12, so a MIDI note works too (`root 62` is D). C by default |
| `scale <name>` | highlights a scale's rows and keys, display only: `major`, `minor` (natural), `dorian`, `mixolydian` or `lydian` (any case; also `ionian`, `aeolian`); `scale off` removes it |
| **Playback** | |
| `play [<beats>]` | plays `buffer~ tcs_out` from a beat (after a `sync`, the host's beat, as `seek`), else from the playhead (or the start), to the phrase's end plus 0.5 s. Before 0 it plays a lead-in of silence |
| `stop` | stops |
| `stopped` | from the patch: the playback reached its end |
| `seek <beats>` | moves the playhead to a beat (after a `sync`, the host's beat): while playing, plays from there (as a click on the ruler); stopped, the next `play` or Space starts there. Past the end it stops at the end; before 0 it plays a lead-in of silence |
| `sync <beats>` | the host's playhead now (in beats) becomes the roll's 0, and the playhead goes to 0. Later `seek`s are in the host's beats |
| `sync off` | no offset (also `sync 0`): `seek` is in the roll's own beats again |
| `playhead <s>` | from the patch: the playback position (`[snapshot~]` of the `[line~]`). Not a seek: `line~` holds its position, so `snapshot~` keeps sending it, even when stopped. After a `seek`, a read or a clear while stopped, it is ignored until playback starts |
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
| 0 | `dictionary tcs_cond`, the conditions without the caption: after every committed edit of the notes, a load, a restore and `bang`; nothing after a failed conversion | `score2f0` and `sing` (and the "regenerate" toggle's `[route dictionary]`) |
| 1 | `bang` when the roll has written the curve into the buffer itself: a curve edit, its undo, a restore | the `init f0 tcs_f0` message into `sing` |
| 2 | the status: `converted <pieces> <seconds>` (before the `dict`), `ok <notes> <seconds>` (a load), `warning <text>`, `error <text>`, `message <text>`, `phonemes <Language> <ph…>`, `restored embed\|pattr <notes>`, `stats <json>`, `report <json>`, `snapshot <file>` | nothing needed (`print`, or a check) |
| 3 | `set <line>` (the lyrics of the notes the line fills, whenever they change) and `select` (after a double-click on a note) | the lyrics `[textedit @outputmode 1 @keymode 1]`, whose outlet 0 goes back into the roll |
| 4 | `dictionary tcs_score`, the edited notes as a score: with every conversion and before it (also when the conversion then fails); nothing when nothing can be sung | anything that wants the score, e.g. `[dict.view]`, `[dict]` with `export_json`, or another roll's `dictionary` |
| 5 | `play <from_ms> <to_ms> <dur_ms>` and `stop` (`from_ms` is below 0 for a lead-in) | `[route play stop]` → `[line~]` → `[play~ tcs_out]`; the position comes back as `playhead <s>` (`[snapshot~ 30]`), and `stopped` at the end |


## `[v8 tcs_prompt.js]`

`[v8 tcs_prompt.js]` makes the prompt for `neural.tokenizer`.

| message | what it does |
|---|---|
| `singer female alto`, `emotion sad`, `method bel canto`, `technique mix then falsetto` | sets that field (as the flags of `sing.py`) and sends the caption. A bad value is an error and changes nothing |
| `caption "Singer: female alto, Emotion: sad"` | a whole caption instead of the fields (quoted, because a comma splits a Max message), checked |
| `style clear` | no fields and no whole caption: the empty caption |
| `bang` | sends the caption again |
| `status` | posts the fields and the caption to the console |

**Outlets:**
- **Outlet 0:** the caption as one symbol, sent on load and after every change. It goes through `[prepend prompt]` → `neural.tokenizer` → `sing`'s inlet 8.
- **Outlet 1:** `ok <caption>` or `error <text>`.
