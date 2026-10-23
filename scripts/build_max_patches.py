"""Writes the Max patch of the TCSinger2 export, tcsinger2.maxpat, into export/max/js/, next to the JS front-end
(Max's v8 `require` resolves the modules' "./lib/…" paths against the patch's folder) and the model files (symlinks, see export/max/js/README.md):
a score -> the conditions dict and the caption -> score2f0 -> the F0 buffer~ (shown and editable in a waveform~) -> sing -> the
audio buffer~ -> playback.

The front-end's dict goes straight to its receivers: through a [t a ...] it arrived empty. Max sends an outlet's
messages right to left by position.

*.maxpat is git-ignored in this repository; this script is how the patches are made again.

Also tcsinger2_ui.maxpat, the patch with the piano roll (UI_WORK_PACKAGES.md, UWP3), and tcsinger2_demo.maxpat, a
minimal patch of the roll with the two models (UWP9), which opens with examples/twinkle_en.json in the roll (its state
is made by node from the roll's own modules).

Usage (from TCSinger2/):
  python scripts/build_max_patches.py [main] [ui] [demo]     (main and ui by default; a patch saved from Max is overwritten)
"""
import json
import os
import shutil
import subprocess
import sys

OUT_DIR = 'export/max/js'
STYLES = ['singer female alto', 'singer male tenor', 'emotion sad', 'emotion happy', 'method bel canto', 'method pop',
          'technique breathy', 'style clear', 'bang', 'status']
DEMO_SCORE = 'examples/twinkle_en.json'
LANGUAGES = ['Chinese', 'English', 'French', 'German', 'Italian', 'Japanese', 'Korean', 'Russian', 'Spanish']


class Patch:
    def __init__(self, width=1200, height=900):
        self.boxes, self.lines, self.size = [], [], (width, height)
        self.offset = (0, 0)  # added to every box's place (a section drawn elsewhere)

    def box(self, maxclass, x, y, w=None, h=22, text=None, ins=1, outs=0, outtype=None, **extra):
        bid = f'obj-{len(self.boxes) + 1}'
        x, y = x + self.offset[0], y + self.offset[1]
        b = dict(id=bid, maxclass=maxclass, numinlets=ins, numoutlets=outs, patching_rect=[x, y, w or max(60, 7 * len(text or '') + 16), h])
        if text is not None:
            b['text'] = text
        if outs:
            b['outlettype'] = outtype or [''] * outs
        b.update(extra)
        self.boxes.append(dict(box=b))
        return bid

    def obj(self, text, x, y, ins=1, outs=1, outtype=None, w=None):
        return self.box('newobj', x, y, w=w, text=text, ins=ins, outs=outs, outtype=outtype)

    def msg(self, text, x, y, w=None):
        return self.box('message', x, y, w=w, text=text, ins=2, outs=1)

    def comment(self, text, x, y, w=300, h=None):
        lines = text.count('\n') + 1 + sum(len(line) * 6.5 // w for line in text.split('\n'))
        return self.box('comment', x, y, w=w, h=h or 14 * lines + 6, text=text, ins=1, outs=0)

    def link(self, a, outlet, b, inlet=0):
        self.lines.append(dict(patchline=dict(source=[a, outlet], destination=[b, inlet])))

    def write(self, path):
        patcher = dict(fileversion=1, appversion=dict(major=9, minor=0, revision=0, architecture='x64', modernui=1),
                       classnamespace='box', rect=[80, 80, *self.size], openinpresentation=0, default_fontsize=12.0,
                       gridsize=[15.0, 15.0], boxes=self.boxes, lines=self.lines)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(patcher=patcher), f, indent=1, ensure_ascii=False)
        print('wrote', path)


def transport(p, roll, play, gain, ids, x=420, y=730):
    """The roll's playback (UWP7): play~ reads the position of a [line~] in ms, which [snapshot~] also sends back to the
    roll, so its playhead is where play~ reads; a 5 ms fade in and out (another [line~] on a [*~]) avoids clicks and
    silences play~ when the position stops. A play from below 0 (a seek before the sync point) is a lead-in: the
    position ramps up from below 0, and [>=~ 0.] keeps the output silent until it reaches 0."""
    rt = p.obj('route play stop', x, y, outs=3, outtype=['', '', ''])
    p.link(roll, 5, rt)
    t_play = p.obj('t l b', x, y + 30, outs=2, outtype=['', 'bang'])
    p.link(rt, 0, t_play)
    ramp = p.msg('$1, $2 $3', x, y + 60)
    p.link(t_play, 0, ramp)
    pos = p.obj('line~', x, y + 90, ins=2, outs=2, outtype=['signal', 'bang'])
    p.link(ramp, 0, pos)
    p.link(pos, 0, play)
    fade_in = p.msg('1 5', x + 100, y + 60)
    p.link(t_play, 1, fade_in)
    fade_out = p.msg('0 5', x + 150, y + 60)
    fader = p.obj('line~', x + 100, y + 90, ins=2, outs=2, outtype=['signal', 'bang'])
    p.link(fade_in, 0, fader)
    p.link(fade_out, 0, fader)
    out = p.obj('*~', x + 160, y + 120, ins=2, outs=1, outtype=['signal'])
    p.link(play, 0, out)
    # the lead-in: silent while the position is below 0
    gate = p.obj('>=~ 0.', x + 270, y + 60, ins=2, outs=1, outtype=['signal'])
    p.link(pos, 0, gate)
    gated = p.obj('*~', x + 100, y + 150, ins=2, outs=1, outtype=['signal'])
    p.link(fader, 0, gated)
    p.link(gate, 0, gated, 1)
    p.link(gated, 0, out, 1)
    p.link(out, 0, gain)
    # stop: the position holds, the fade goes out
    t_stop = p.obj('t b b', x + 200, y + 30, outs=2, outtype=['bang', 'bang'])
    p.link(rt, 1, t_stop)
    halt = p.msg('stop', x + 200, y + 60)
    p.link(t_stop, 1, halt)
    p.link(t_stop, 0, fade_out)
    p.link(halt, 0, pos)
    # the end of the ramp: fade out, and tell the roll
    t_end = p.obj('t b b', x, y + 120, outs=2, outtype=['bang', 'bang'])
    p.link(pos, 1, t_end)
    p.link(t_end, 1, fade_out)
    ended = p.msg('stopped', x + 65, y + 120)
    p.link(t_end, 0, ended)
    p.link(ended, 0, roll)
    # the position back to the roll, in seconds
    snap = p.obj('snapshot~ 30', x + 250, y + 90, ins=2, outs=1)
    p.link(pos, 0, snap)
    secs = p.obj('/ 1000.', x + 250, y + 120, ins=2, outs=1)
    p.link(snap, 0, secs)
    head = p.obj('prepend playhead', x + 325, y + 120)
    p.link(secs, 0, head)
    p.link(head, 0, roll)
    p.comment('playback: the roll\'s ruler, Space, or play/stop; the playhead follows play~', x + 330, y, w=230)
    ids['playhead'] = head
    ids['play_out'] = out


def f0_row(p, ui, ids, shown=lambda bid: bid, aux=(600, 645)):
    """The F0 curve's range commands (UWP6) under the lyrics line, into the roll: snap, reset, unvoice, smooth,
    offset -1/1 and the vibrato (with its rate and depth boxes). One start and one end box (seconds) are the range of
    every button: each goes through [zl.join] (the range in its right inlet), so it sends f0 <command> ... <start> <end>;
    0 0 (a range that doesn't end after it starts) means the selected notes, or the whole curve. `shown` puts a box in
    presentation too; `aux` places the boxes behind the row; ids gets the buttons, the vibrato and the range boxes (the
    checks)."""
    ax, ay = aux
    rng = p.obj('zl.join', ax + 160, ay + 60, ins=2, outs=2, outtype=['', ''])  # (append's right inlet ignores a list)
    ids['f0_append'] = rng
    p.link(rng, 0, ui)
    p.link(p.obj('loadmess 0. 0.', ax + 230, ay + 60), 0, rng, 1)
    x = 20
    ids['f0_buttons'] = {}
    for text in ['f0 snap', 'f0 reset', 'f0 unvoice', 'f0 smooth', 'f0 offset -1', 'f0 offset 1']:
        m = shown(p.msg(text, x, 530))
        p.link(m, 0, rng)
        ids['f0_buttons'][text] = m
        x += 7 * len(text) + 16 + 8
    rate = shown(p.box('flonum', 20, 556, w=50, ins=1, outs=2, outtype=['', 'bang'], minimum=0.0))
    depth = shown(p.box('flonum', 75, 556, w=50, ins=1, outs=2, outtype=['', 'bang'], minimum=0.0))
    p.link(p.obj('loadmess 5.5', ax, ay), 0, rate)
    p.link(p.obj('loadmess 0.5', ax + 110, ay), 0, depth)
    vib_pak = p.obj('pak 5.5 0.5', ax, ay + 30, ins=2, outs=1)
    p.link(rate, 0, vib_pak)
    p.link(depth, 0, vib_pak, 1)
    vib = shown(p.msg('f0 vibrato 5.5 0.5', 130, 556, w=125))
    vib_set = p.obj('prepend set f0 vibrato', ax + 160, ay + 30)
    p.link(vib_pak, 0, vib_set)
    p.link(vib_set, 0, vib)
    p.link(vib, 0, rng)
    # the range of every button
    shown(p.comment('range (s):', 275, 556, w=70))
    start = shown(p.box('flonum', 345, 556, w=50, ins=1, outs=2, outtype=['', 'bang'], minimum=0.0))
    stop = shown(p.box('flonum', 400, 556, w=50, ins=1, outs=2, outtype=['', 'bang'], minimum=0.0))
    rng_pak = p.obj('pak 0. 0.', ax, ay + 60, ins=2, outs=1)
    ids['f0_range_pak'] = rng_pak
    p.link(start, 0, rng_pak)
    p.link(stop, 0, rng_pak, 1)
    p.link(rng_pak, 0, rng, 1)
    ids['f0_vibrato'] = vib
    ids['f0_start'] = ids['f0_vib_start'] = start
    ids['f0_end'] = ids['f0_vib_end'] = stop
    shown(p.comment('F0: F or the roll\'s F0 button, then draw on the curve (D draw, L line, S smooth, E erase, U unvoice; '
                    'Shift: semitones). The buttons act on the range start-end in seconds; with 0 0, on the selected notes '
                    '(none: the whole curve). Vibrato: rate (Hz), depth (semitones).', 20, 582, w=580))


def sing_section(p, inits, gate, fe, tok, roll=None, ids=None):
    """3. sing (generate, seed, cfg_scale; `inits` send init f0 to it, `gate` bangs generate) and the audio: buffer~
    tcs_out, its waveform~, playback and writewave. Returns the sing object. With a `roll` (UWP7), the roll drives the
    playback: its outlet 5 (play <from_ms> <to_ms> <dur_ms> | stop) -> [line~] -> [play~], and the position goes back
    to it as playhead <s> (and stopped at the end); `ids` gets the playhead's and the output's boxes (the checks)."""
    # 3. sing
    p.comment('3. Sing the curve (cfg_scale, seed)', 20, 560, w=300)
    sing_btn = p.msg('generate', 20, 590)
    sing_seed = p.box('number', 100, 590, w=60, ins=1, outs=2, outtype=['', 'bang'])
    pre_sseed = p.obj('prepend seed', 100, 620)
    p.link(sing_seed, 0, pre_sseed)
    cfg = p.box('flonum', 190, 590, w=60, ins=1, outs=2, outtype=['', 'bang'])
    pre_cfg = p.obj('prepend cfg_scale', 190, 620)
    p.link(cfg, 0, pre_cfg)
    p.link(p.obj('loadmess 3.', 260, 560), 0, cfg)  # the export's default
    sing = p.obj('neural.gen~ tcsinger2_sing sing tcs_out', 20, 660, ins=9, outs=1, outtype=['bang'], w=280)
    for src in (sing_btn, pre_sseed, pre_cfg) + tuple(inits):
        p.link(src, 0, sing)
    if gate is not None:
        p.link(gate, 0, sing_btn)
    p.link(fe, 0, sing)
    p.link(tok, 0, sing, 8)  # input_ids is sing's 9th condition (neural_tilde 0.1.0: one inlet per input)
    sing_done = p.obj('print sing_done', 20, 695)
    p.link(sing, 0, sing_done)

    # the audio
    p.obj('buffer~ tcs_out', 1060, 90, outs=2, outtype=['float', 'bang'])
    wave_out = p.box('waveform~', 320, 600, w=880, h=120, ins=5, outs=6, outtype=['float', 'float', 'float', 'float', 'list', ''])
    lb2 = p.obj('loadbang', 1080, 730, outs=1, outtype=['bang'])
    set_out = p.msg('set tcs_out', 1080, 755)
    p.link(lb2, 0, set_out)
    p.link(set_out, 0, wave_out)
    play_msg = p.msg('start' if roll is None else 'play', 20, 760)
    stop_msg = p.msg('stop', 70, 760)
    play = p.obj('play~ tcs_out', 20, 790, ins=1, outs=2, outtype=['signal', 'bang'])
    gain = p.obj('*~ 0.5', 20, 820, ins=2, outs=1, outtype=['signal'])
    if roll is None:
        p.link(play_msg, 0, play)
        p.link(stop_msg, 0, play)
        p.link(play, 0, gain)
    else:
        transport(p, roll, play, gain, ids if ids is not None else {})
        p.link(play_msg, 0, roll)
        p.link(stop_msg, 0, roll)
    dac = p.box('ezdac~', 20, 850, w=45, h=45, ins=2, outs=0)
    p.link(gain, 0, dac)
    p.link(gain, 0, dac, 1)
    save = p.msg('writewave', 160, 790)
    p.link(save, 0, p.obj('buffer~ tcs_out', 160, 820, outs=2, outtype=['float', 'bang']))
    p.comment('save the audio (48 kHz)', 240, 790, w=160)
    return sing


def main_patch(path=os.path.join(OUT_DIR, 'tcsinger2.maxpat'), hook=None):
    """tcsinger2.maxpat, without the piano roll: [v8 tcs_max.js] turns a score file into dict tcs_cond for both models;
    [v8 tcs_prompt.js] makes the caption from the style buttons -> neural.tokenizer -> sing (inlet 8). `hook(p, ids)` may
    add boxes before the patch is written (the checks). Returns ids."""
    p = Patch(1260, 980)
    ids = {}
    p.comment('TCSinger2 in Max: a score -> score2f0 (the F0 curve) -> sing (the voice). EWP7 of EXPORTING_WORK_PACKAGES.md.\n'
              'Setup: the model files next to this patch (symlinks, export/max/js/README.md) and the front-end data '
              '(python scripts/build_max_js_data.py). Phrases up to 20 s; generation: score2f0 about 0.4 s, sing about 0.9 s.',
              20, 10, w=1100)

    # 1. score -> conditions dict
    p.comment('1. Open a score (JSON, examples/*.json). The style buttons set the caption.', 20, 60, w=330)
    open_btn = p.box('button', 20, 90, w=24, h=24, ins=1, outs=1, outtype=['bang'])
    dialog = p.obj('opendialog', 50, 90, ins=2, outs=2, outtype=['', 'bang'])
    pre_read = p.obj('prepend read', 50, 120)
    p.link(open_btn, 0, dialog)
    p.link(dialog, 0, pre_read)
    fe = p.box('newobj', 20, 330, w=140, text='v8 tcs_max.js', ins=1, outs=2, outtype=['', ''])
    ids['fe'] = fe
    prompt_js = p.box('newobj', 180, 330, w=140, text='v8 tcs_prompt.js', ins=1, outs=2, outtype=['', ''])
    ids['prompt'] = prompt_js
    for k, s in enumerate(STYLES):
        m = p.msg(s, 180 + (k % 2) * 140, 90 + (k // 2) * 26)
        p.link(m, 0, prompt_js)
    p.link(pre_read, 0, fe)
    fe_status = p.obj('print tcs', 20, 400)
    p.link(fe, 1, fe_status)
    p.link(prompt_js, 1, p.obj('print tcs_prompt', 340, 330))
    p.obj('dict tcs_cond', 20, 370, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the front-end's dict alive
    show_caption = p.msg('', 20, 470, w=420)
    set_caption = p.obj('prepend set', 20, 440)
    p.link(prompt_js, 0, set_caption)
    p.link(set_caption, 0, show_caption)
    prompt = p.obj('prepend prompt', 220, 440)
    p.link(prompt_js, 0, prompt)
    tok = p.obj('neural.tokenizer tcsinger2_sing', 220, 500, ins=1, outs=2, outtype=['', ''], w=220)
    ids['tok'] = tok
    p.link(prompt, 0, tok)

    # 2. score2f0
    p.comment('2. The F0 curve: score2f0 (seed). Ready -> init f0 on sing; with the toggle, sing right away (notes only).', 500, 60, w=330)
    seed = p.box('number', 500, 110, w=60, ins=1, outs=2, outtype=['', 'bang'])
    pre_seed = p.obj('prepend seed', 500, 140)
    p.link(seed, 0, pre_seed)
    gen_btn = p.msg('generate', 600, 110)
    f0gen = p.obj('neural.gen~ tcsinger2_f0 score2f0 tcs_f0', 500, 180, ins=8, outs=1, outtype=['bang'], w=280)
    ids['f0gen'] = f0gen
    p.link(pre_seed, 0, f0gen)
    p.link(gen_btn, 0, f0gen)
    p.link(fe, 0, f0gen)
    f0_done = p.obj('t b b', 500, 215, outs=2, outtype=['bang', 'bang'])
    p.link(f0gen, 0, f0_done)
    init = p.msg('init f0 tcs_f0', 500, 245)
    p.link(f0_done, 1, init)
    auto = p.box('toggle', 640, 245, w=22, h=22, ins=1, outs=1, outtype=['int'])
    p.comment('sing when the curve is ready', 668, 245, w=170)
    gate = p.obj('gate', 600, 275, ins=2, outs=1)
    p.link(auto, 0, gate)
    p.link(f0_done, 0, gate, 1)

    # the curve
    p.comment('The curve: MIDI pitch at 50 Hz, 0 = unvoiced (buffer~ tcs_f0, 1,003 samples, 20 s). Draw to edit it, then sing.',
              500, 310, w=700)
    p.obj('buffer~ tcs_f0', 1060, 60, outs=2, outtype=['float', 'bang'])
    wave_f0 = p.box('waveform~', 500, 340, w=700, h=160, ins=5, outs=6, outtype=['float', 'float', 'float', 'float', 'list', ''])
    lb = p.obj('loadbang', 1080, 510, outs=1, outtype=['bang'])
    setup = p.msg('set tcs_f0, vzoom 0.0078', 1080, 535, w=170)
    p.link(lb, 0, setup)
    p.link(setup, 0, wave_f0)
    for k, s in enumerate(['mode draw', 'mode select']):
        m = p.msg(s, 500 + k * 90, 510)
        p.link(m, 0, wave_f0)

    ids['sing'] = sing_section(p, [init], gate, fe, tok)

    # inspection
    p.comment('The models\' inputs (neural.info):', 500, 760, w=300)
    p.obj('neural.info tcsinger2_f0', 500, 790, w=200)
    p.obj('neural.info tcsinger2_sing', 720, 790, w=200)
    if hook:
        hook(p, ids)
    p.write(path)
    return ids


def ui_patch(path=os.path.join(OUT_DIR, 'tcsinger2_ui.maxpat'), hook=None):
    """tcsinger2_ui.maxpat (UI_WORK_PACKAGES.md, UWP3-UWP9): the piano roll [v8ui tcs_ui.js] sends the conditions of its notes
    straight to both models, and [v8 tcs_prompt.js] the caption;
    score2f0 (by hand, or after each change with the toggle) -> the roll shows the curve, which its F0 mode edits -> sing;
    the roll plays the result (its ruler, Space) and shows the playhead.
    The roll saves its score and curves in its box. `hook(p, ids)` may add boxes before the patch is written (the
    checks). Returns ids."""
    p = Patch(1300, 1210)
    p.comment('TCSinger2 in Max with the piano roll (UI_WORK_PACKAGES.md). Open a score, or draw notes (the pencil, or a double-click), '
              'and type their lyrics in the line under the roll; the roll makes them singable (its status line says what it fixed) '
              'for both models, and shows their phonemes. Generate the F0 (or turn on "regenerate"), edit '
              'it in the F0 mode if you like, then sing, and play it (click the ruler, or Space). The roll and its curves are saved with '
              'the patch.', 20, 10, w=1100)
    ids = {}
    # the roll and its controls
    ui = p.box('v8ui', 20, 75, w=900, h=420, ins=1, outs=6, outtype=[''] * 6, filename='tcs_ui.js', varname='ui',
               parameter_enable=0, presentation=1, presentation_rect=[20, 75, 900, 420],
               textfile=dict(filename='tcs_ui.js', flags=0, embed=0, autowatch=1))
    ids['ui'] = ui
    open_btn = p.box('button', 20, 42, w=24, h=24, ins=1, outs=1, outtype=['bang'])
    dialog = p.obj('opendialog', 50, 42, ins=2, outs=2, outtype=['', 'bang'])
    pre_read = p.obj('prepend read', 140, 42)
    p.link(open_btn, 0, dialog)
    p.link(dialog, 0, pre_read)
    p.link(pre_read, 0, ui)
    menu = p.box('umenu', 250, 42, w=100, ins=1, outs=3, outtype=['int', '', ''], parameter_enable=0,
                 items=[x for lang in LANGUAGES for x in (lang, ',')][:-1])
    pre_lang = p.obj('prepend language', 360, 42)
    p.link(menu, 1, pre_lang)
    p.link(pre_lang, 0, ui)
    tempo = p.box('flonum', 480, 42, w=60, ins=1, outs=2, outtype=['', 'bang'])
    pre_tempo = p.obj('prepend tempo', 545, 42)
    p.link(tempo, 0, pre_tempo)
    p.link(pre_tempo, 0, ui)
    for k, text in enumerate(['clear', 'bang', 'view fit']):
        p.link(p.msg(text, 650 + k * 55, 42), 0, ui)
    write_btn = p.box('button', 820, 42, w=24, h=24, ins=1, outs=1, outtype=['bang'])
    save_dialog = p.obj('savedialog', 850, 42, ins=1, outs=3, outtype=['', '', 'bang'])
    pre_write = p.obj('prepend write', 940, 42)
    p.link(write_btn, 0, save_dialog)
    p.link(save_dialog, 0, pre_write)
    p.link(pre_write, 0, ui)
    p.comment('open | language | tempo | write the score', 1040, 42, w=240)
    # the lyrics line (U5): the selected notes' lyrics (or all, with none selected); Enter fills them
    lyrics = p.box('textedit', 20, 500, w=560, h=24, ins=1, outs=4, outtype=['', 'int', '', ''], varname='lyrics',
                   outputmode=1, keymode=1, lines=1, parameter_enable=0, presentation=1, presentation_rect=[20, 500, 560, 24])
    ids['lyrics'] = lyrics
    p.link(lyrics, 0, ui)
    p.link(ui, 3, lyrics)
    p.comment('Lyrics: select notes (none: all), type, Enter. ";" between notes, or spaces '
              '(Chinese, Japanese, Korean: one character per note); "-" continues a word. Double-click a note to type its lyric. '
              'Click the roll, then: B pencil, Backspace delete, arrows move (Shift: octave/bar), S split, J join, '
              'Cmd-Z/Shift-Z, Cmd-C/X/V/D. Drag an edge to resize (Alt: alone), Cmd: no snap. Click AP/SP to switch; ph: phonemes.',
              600, 500, w=320)

    # the F0 curve's range commands (UWP6): on the selected notes, or the whole curve; in presentation too
    def shown(bid):
        b = next(x['box'] for x in p.boxes if x['box']['id'] == bid)
        b.update(presentation=1, presentation_rect=list(b['patching_rect']))
        return bid

    f0_row(p, ui, ids, shown)

    # the roll converts its notes: its outlet 0 (dictionary tcs_cond) goes straight to both models; the caption comes
    # from tcs_prompt.js
    fy = 110  # below the F0 rows
    p.obj('dict tcs_cond', 20, 540 + fy, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the roll's dict alive
    p.link(ui, 2, p.obj('print tcs_ui', 140, 540 + fy))
    prompt_js = p.box('newobj', 20, 575 + fy, w=140, text='v8 tcs_prompt.js', ins=1, outs=2, outtype=['', ''])
    ids['prompt'] = prompt_js
    p.link(prompt_js, 1, p.obj('print tcs_prompt', 180, 575 + fy))
    p.comment('Style (the caption changes at once):', 940, 75, w=300)
    for k, s in enumerate(STYLES):
        p.link(p.msg(s, 940 + (k % 2) * 160, 100 + (k // 2) * 26), 0, prompt_js)
    show_caption = p.msg('', 20, 640 + fy, w=420)
    set_caption = p.obj('prepend set', 20, 610 + fy)
    p.link(prompt_js, 0, set_caption)
    p.link(set_caption, 0, show_caption)
    prompt = p.obj('prepend prompt', 460, 610 + fy)
    p.link(prompt_js, 0, prompt)
    tok = p.obj('neural.tokenizer tcsinger2_sing', 460, 640 + fy, ins=1, outs=2, outtype=['', ''], w=220)
    ids['tok'] = tok
    p.link(prompt, 0, tok)

    # score2f0
    p.comment('2. The F0 curve: score2f0 (seed)', 940, 240, w=300)
    seed = p.box('number', 940, 265, w=60, ins=1, outs=2, outtype=['', 'bang'])
    ids['f0_seed'] = seed
    pre_seed = p.obj('prepend seed', 940, 290)
    p.link(seed, 0, pre_seed)
    gen_btn = p.msg('generate', 1010, 265)
    f0gen = p.obj('neural.gen~ tcsinger2_f0 score2f0 tcs_f0', 940, 360, ins=8, outs=1, outtype=['bang'], w=280)
    ids['f0gen'] = f0gen
    p.link(pre_seed, 0, f0gen)
    p.link(gen_btn, 0, f0gen)
    p.link(ui, 0, f0gen)
    # regenerate after each change: after the dict has reached score2f0 (deferlow), generate
    auto = p.box('toggle', 1080, 265, w=22, h=22, ins=1, outs=1, outtype=['int'])
    ids['auto'] = auto
    p.comment('regenerate F0 after each change', 1105, 265, w=180)
    on_dict = p.obj('route dictionary', 1080, 300, ins=1, outs=2, outtype=['', ''])
    p.link(ui, 0, on_dict)
    t_dict = p.obj('t b', 1080, 325, outs=1, outtype=['bang'])
    p.link(on_dict, 0, t_dict)
    auto_gate = p.obj('gate', 1140, 325, ins=2, outs=1)
    p.link(auto, 0, auto_gate)
    p.link(t_dict, 0, auto_gate, 1)
    defer = p.obj('deferlow', 1190, 325)
    p.link(auto_gate, 0, defer)
    p.link(defer, 0, gen_btn)
    # done: the roll reads the curve, then sing reads it, then (toggle) sing
    f0_done = p.obj('t b b b', 940, 395, outs=3, outtype=['bang', 'bang', 'bang'])
    p.link(f0gen, 0, f0_done)
    refresh = p.msg('f0 refresh', 1060, 425)
    p.link(f0_done, 2, refresh)
    p.link(refresh, 0, ui)
    init = p.msg('init f0 tcs_f0', 1000, 450)
    p.link(f0_done, 1, init)
    p.link(ui, 1, init)  # the roll wrote the curve (an edit, its undo, a restore)
    sing_auto = p.box('toggle', 940, 480, w=22, h=22, ins=1, outs=1, outtype=['int'])
    p.comment('sing when the curve is ready', 965, 480, w=170)
    sing_gate = p.obj('gate', 940, 505, ins=2, outs=1)
    p.link(sing_auto, 0, sing_gate)
    p.link(f0_done, 0, sing_gate, 1)
    p.obj('buffer~ tcs_f0', 1140, 395, outs=2, outtype=['float', 'bang'])
    wave_f0 = p.box('waveform~', 940, 540, w=340, h=80, ins=5, outs=6, outtype=['float', 'float', 'float', 'float', 'list', ''])
    lb = p.obj('loadbang', 1140, 630, outs=1, outtype=['bang'])
    setup = p.msg('set tcs_f0, vzoom 0.0078', 1140, 655, w=150)
    p.link(lb, 0, setup)
    p.link(setup, 0, wave_f0)
    p.comment('buffer~ tcs_f0 (for comparison)', 940, 625, w=200)

    # 3. sing and the audio, below
    p.offset = (0, 140 + fy)
    sing = sing_section(p, [init], sing_gate, ui, tok, roll=ui, ids=ids)
    p.offset = (0, 0)
    out_buf = next(b['box'] for b in p.boxes if b['box'].get('text') == 'buffer~ tcs_out')  # off the auto-F0 objects
    out_buf['patching_rect'][:2] = [1140, 455]
    ids['sing'] = sing
    ids['init'] = init
    if hook:
        hook(p, ids)
    p.write(path)
    return ids


# a score as the roll's saved state (tcs_ui_state.js), as the roll would save it after reading the file: the notes, the
# view fitted to them, no curve. Printed as {"json": <the state>, "digest": <its digest>}.
ROLL_STATE_JS = r"""
require(process.env.TCS_NODE_LIB); // the modules in lib/, with their "./lib/..." requires (export_check/node_lib.js)
const fs = require("fs");
const N = require("tcs_notes.js");
const V = require("tcs_ui_view.js");
const ST = require("tcs_ui_state.js");
const [file, w, h] = process.argv.slice(1);
const { model } = N.fromScore(JSON.parse(fs.readFileSync(file, "utf8")), file.split("/").pop().replace(/\.json$/i, ""));
const st = ST.pack({ model, view: V.fit(V.makeView(Number(w), Number(h)), model), f0: null });
process.stdout.write(JSON.stringify({ json: ST.encode(st), digest: ST.digest(st) }));
"""


def roll_state(score, w=900, h=420):
    """The roll's saved state of `score` and its digest (ST.digest), or None without node."""
    node = shutil.which('node')
    if not node:
        print(f'warning: no node, so the demo starts empty instead of with {score}')
        return None
    env = dict(os.environ, TCS_NODE_LIB=os.path.abspath('export_check/node_lib.js'))
    out = subprocess.run([node, '-e', ROLL_STATE_JS, os.path.abspath(score), str(w), str(h)], env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def demo_patch(path=os.path.join(OUT_DIR, 'tcsinger2_demo.maxpat'), hook=None, score=DEMO_SCORE):
    """tcsinger2_demo.maxpat (UWP9): the least the roll needs to sing. The roll (the conditions of its notes) ->
    score2f0 -> the roll shows and edits the curve -> sing (with tcs_prompt.js's caption) -> the roll plays it. Without the UI patch's seeds, cfg_scale,
    waveform~s and auto toggles. It opens with `score` in the roll. `hook(p, ids)` may add boxes before the patch is
    written (the check). Returns ids."""
    p = Patch(1300, 1090)
    p.comment('TCSinger2 with the piano roll: open or draw notes, type their lyrics, generate the F0 (edit it if you like), '
              'sing, and play (click the ruler, or Space). See export/max/js/README.md.', 20, 10, w=1100)
    ids = {}
    # 1. the notes: the roll, its score and language, the lyrics line, the F0 row
    open_btn = p.box('button', 20, 42, w=24, h=24, ins=1, outs=1, outtype=['bang'])
    dialog = p.obj('opendialog', 50, 42, ins=2, outs=2, outtype=['', 'bang'])
    pre_read = p.obj('prepend read', 140, 42)
    p.link(open_btn, 0, dialog)
    p.link(dialog, 0, pre_read)
    menu = p.box('umenu', 250, 42, w=100, ins=1, outs=3, outtype=['int', '', ''], parameter_enable=0,
                 items=[x for lang in LANGUAGES for x in (lang, ',')][:-1])
    pre_lang = p.obj('prepend language', 360, 42)
    p.link(menu, 1, pre_lang)
    p.comment('1. The notes: open a score | its language. Or draw notes (the pencil, or a double-click).', 500, 42, w=420)
    state = roll_state(score) if score else None
    extra = dict(embedstate=[['state', state['json']]]) if state else {}
    ui = p.box('v8ui', 20, 75, w=900, h=420, ins=1, outs=6, outtype=[''] * 6, filename='tcs_ui.js', varname='ui',
               parameter_enable=0, textfile=dict(filename='tcs_ui.js', flags=0, embed=0, autowatch=1), **extra)
    ids['ui'] = ui
    ids['state'] = state
    p.link(pre_read, 0, ui)
    p.link(pre_lang, 0, ui)
    lyrics = p.box('textedit', 20, 500, w=560, h=24, ins=1, outs=4, outtype=['', 'int', '', ''], varname='lyrics',
                   outputmode=1, keymode=1, lines=1, parameter_enable=0)
    ids['lyrics'] = lyrics
    p.link(lyrics, 0, ui)
    p.link(ui, 3, lyrics)
    p.comment('Lyrics: select notes (none: all), type, Enter. Click the roll for its keys: B pencil, Backspace delete, '
              'arrows move, S split, J join, Cmd-Z undo, F the F0 mode.', 600, 500, w=320)
    f0_row(p, ui, ids, aux=(940, 240))

    # the roll's outlet 0: the conditions (dictionary tcs_cond) of its notes, straight to both models; the caption
    p.comment('The roll\'s outlet 0 -> both models; the style -> the caption.', 20, 650, w=440)
    p.obj('dict tcs_cond', 20, 675, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the roll's dict alive
    prompt_js = p.box('newobj', 240, 675, w=140, text='v8 tcs_prompt.js', ins=1, outs=2, outtype=['', ''])
    ids['prompt'] = prompt_js
    prompt = p.obj('prepend prompt', 240, 710)
    p.link(prompt_js, 0, prompt)
    tok = p.obj('neural.tokenizer tcsinger2_sing', 240, 745, ins=1, outs=2, outtype=['', ''], w=220)
    ids['tok'] = tok
    p.link(prompt, 0, tok)
    p.comment('Style: the caption changes at once.', 940, 75, w=300)
    for k, s in enumerate(STYLES):
        p.link(p.msg(s, 940 + (k % 2) * 160, 100 + (k // 2) * 26), 0, prompt_js)

    # 2. score2f0: the curve into buffer~ tcs_f0, then the roll reads it and sing is told
    p.comment('2. The F0 curve (score2f0): the roll shows it; edit it in its F0 mode.', 500, 650, w=420)
    f0_btn = p.msg('generate', 500, 675)
    f0gen = p.obj('neural.gen~ tcsinger2_f0 score2f0 tcs_f0', 500, 705, ins=8, outs=1, outtype=['bang'], w=280)
    ids['f0_generate'] = f0_btn
    ids['f0gen'] = f0gen
    p.link(f0_btn, 0, f0gen)
    p.link(ui, 0, f0gen)
    p.obj('buffer~ tcs_f0', 800, 705, outs=2, outtype=['float', 'bang'])
    f0_done = p.obj('t b b', 500, 740, outs=2, outtype=['bang', 'bang'])
    p.link(f0gen, 0, f0_done)
    refresh = p.msg('f0 refresh', 500, 770)
    p.link(f0_done, 1, refresh)
    p.link(refresh, 0, ui)
    init = p.msg('init f0 tcs_f0', 590, 770)
    p.link(f0_done, 0, init)
    p.link(ui, 1, init)  # the roll wrote the curve (an edit, its undo, a restore)
    ids['init'] = init

    # 3. sing: the voice into buffer~ tcs_out
    p.comment('3. Sing (after the F0, and after each edit of the curve).', 20, 810, w=400)
    sing_btn = p.msg('generate', 20, 835)
    sing = p.obj('neural.gen~ tcsinger2_sing sing tcs_out', 20, 865, ins=9, outs=1, outtype=['bang'], w=280)
    ids['sing_generate'] = sing_btn
    ids['sing'] = sing
    for src in (sing_btn, init, ui):
        p.link(src, 0, sing)
    p.link(tok, 0, sing, 8)  # input_ids is sing's 9th condition (neural_tilde 0.1.0: one inlet per input)
    p.obj('buffer~ tcs_out', 320, 865, outs=2, outtype=['float', 'bang'])

    # 4. playback: the roll's transport (its ruler, Space, or play/stop) -> [line~] -> [play~]; the playhead back
    p.comment('4. Play: click the roll\'s ruler, or Space (click the roll first). Turn the audio on.', 500, 810, w=440)
    play_msg = p.msg('play', 500, 835)
    stop_msg = p.msg('stop', 570, 835)
    p.link(play_msg, 0, ui)
    p.link(stop_msg, 0, ui)
    play = p.obj('play~ tcs_out', 500, 955, ins=1, outs=2, outtype=['signal', 'bang'])
    gain = p.obj('*~ 0.5', 800, 1015, ins=2, outs=1, outtype=['signal'])
    transport(p, ui, play, gain, ids, x=640, y=865)
    dac = p.box('ezdac~', 800, 1040, w=45, h=45, ins=2, outs=0)
    p.link(gain, 0, dac)
    p.link(gain, 0, dac, 1)
    if hook:
        hook(p, ids)
    p.write(path)
    return ids


if __name__ == '__main__':
    which = sys.argv[1:] or ['main', 'ui']
    if 'main' in which:
        main_patch()
    if 'ui' in which:
        ui_patch()
    if 'demo' in which:
        demo_patch()
