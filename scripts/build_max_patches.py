"""Writes the Max patch of the TCSinger2 export, tcsinger2.maxpat, into export/max/js/, next to the JS front-end
(Max's v8 `require` finds its modules only there) and the model files (symlinks, see export/max/js/README.md):
a score -> the conditions dict and the caption -> score2f0 -> the F0 buffer~ (shown and editable in a waveform~) -> sing -> the
audio buffer~ -> playback.

The front-end's dict goes straight to its receivers: through a [t a ...] it arrived empty. Max sends an outlet's
messages right to left by position.

*.maxpat is git-ignored in this repository; this script is how the patches are made again.

Also tcsinger2_ui.maxpat, the patch with the piano roll (UI_WORK_PACKAGES.md, UWP3).

Usage (from TCSinger2/):
  python scripts/build_max_patches.py [main] [ui]     (both by default; a patch saved from Max is overwritten)
"""
import json
import os
import sys

OUT_DIR = 'export/max/js'
STYLES = ['singer female alto', 'singer male tenor', 'emotion sad', 'emotion happy', 'method bel canto', 'method pop',
          'technique breathy', 'style clear', 'bang', 'status']
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


def sing_section(p, inits, gate, fe, tok):
    """3. sing (generate, seed, cfg_scale; `inits` send init f0 to it, `gate` bangs generate) and the audio: buffer~
    tcs_out, its waveform~, playback and writewave. Returns the sing object."""
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
    p.link(tok, 0, sing)
    sing_done = p.obj('print sing_done', 20, 695)
    p.link(sing, 0, sing_done)

    # the audio
    p.obj('buffer~ tcs_out', 1060, 90, outs=2, outtype=['float', 'bang'])
    wave_out = p.box('waveform~', 320, 600, w=880, h=120, ins=5, outs=6, outtype=['float', 'float', 'float', 'float', 'list', ''])
    lb2 = p.obj('loadbang', 1080, 730, outs=1, outtype=['bang'])
    set_out = p.msg('set tcs_out', 1080, 755)
    p.link(lb2, 0, set_out)
    p.link(set_out, 0, wave_out)
    play_msg = p.msg('start', 20, 760)
    stop_msg = p.msg('stop', 70, 760)
    play = p.obj('play~ tcs_out', 20, 790, ins=1, outs=2, outtype=['signal', 'bang'])
    p.link(play_msg, 0, play)
    p.link(stop_msg, 0, play)
    gain = p.obj('*~ 0.5', 20, 820, ins=2, outs=1, outtype=['signal'])
    p.link(play, 0, gain)
    dac = p.box('ezdac~', 20, 850, w=45, h=45, ins=2, outs=0)
    p.link(gain, 0, dac)
    p.link(gain, 0, dac, 1)
    save = p.msg('writewave', 160, 790)
    p.link(save, 0, p.obj('buffer~ tcs_out', 160, 820, outs=2, outtype=['float', 'bang']))
    p.comment('save the audio (48 kHz)', 240, 790, w=160)
    return sing


def main_patch():
    p = Patch(1260, 980)
    p.comment('TCSinger2 in Max: a score -> score2f0 (the F0 curve) -> sing (the voice). EWP7 of EXPORTING_WORK_PACKAGES.md.\n'
              'Setup: the model files next to this patch (symlinks, export/max/js/README.md) and the front-end data '
              '(python scripts/build_max_js_data.py). Phrases up to 20 s; generation: score2f0 about 0.4 s, sing about 0.9 s.',
              20, 10, w=1100)

    # 1. score -> conditions dict
    p.comment('1. Open a score (JSON, examples/*.json), or change the style and bang', 20, 60, w=330)
    open_btn = p.box('button', 20, 90, w=24, h=24, ins=1, outs=1, outtype=['bang'])
    dialog = p.obj('opendialog', 50, 90, ins=2, outs=2, outtype=['', 'bang'])
    pre_read = p.obj('prepend read', 50, 120)
    p.link(open_btn, 0, dialog)
    p.link(dialog, 0, pre_read)
    fe = p.box('newobj', 20, 330, w=140, text='v8 tcs_max.js', ins=1, outs=3, outtype=['', '', ''])
    for k, s in enumerate(STYLES):
        m = p.msg(s, 180 + (k % 2) * 140, 90 + (k // 2) * 26)
        p.link(m, 0, fe)
    p.link(pre_read, 0, fe)
    fe_status = p.obj('print tcs', 180, 370)
    p.link(fe, 2, fe_status)
    p.obj('dict tcs_cond', 20, 370, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the front-end's dict alive
    show_caption = p.msg('', 20, 470, w=420)
    set_caption = p.obj('prepend set', 20, 440)
    p.link(fe, 1, set_caption)
    p.link(set_caption, 0, show_caption)
    prompt = p.obj('prepend prompt', 220, 440)
    p.link(fe, 1, prompt)
    tok = p.obj('neural.tokenizer tcsinger2_sing', 220, 500, ins=1, outs=2, outtype=['', ''], w=220)
    p.link(prompt, 0, tok)

    # 2. score2f0
    p.comment('2. The F0 curve: score2f0 (seed). Ready -> init f0 on sing; with the toggle, sing right away (notes only).', 500, 60, w=330)
    seed = p.box('number', 500, 110, w=60, ins=1, outs=2, outtype=['', 'bang'])
    pre_seed = p.obj('prepend seed', 500, 140)
    p.link(seed, 0, pre_seed)
    gen_btn = p.msg('generate', 600, 110)
    f0gen = p.obj('neural.gen~ tcsinger2_f0 score2f0 tcs_f0', 500, 180, ins=8, outs=1, outtype=['bang'], w=280)
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

    sing_section(p, [init], gate, fe, tok)

    # inspection
    p.comment('The models\' inputs (neural.info):', 500, 760, w=300)
    p.obj('neural.info tcsinger2_f0', 500, 790, w=200)
    p.obj('neural.info tcsinger2_sing', 720, 790, w=200)
    p.write(os.path.join(OUT_DIR, 'tcsinger2.maxpat'))


def ui_patch(path=os.path.join(OUT_DIR, 'tcsinger2_ui.maxpat'), hook=None):
    """tcsinger2_ui.maxpat (UI_WORK_PACKAGES.md, UWP3-UWP5): the piano roll [v8ui tcs_ui.js] sends its notes to the front-end;
    score2f0 (by hand, or after each change with the toggle) -> the roll shows the curve -> sing. The roll saves its
    score and curves in its box. `hook(p, ids)` may add boxes before the patch is written (the checks). Returns ids."""
    p = Patch(1300, 1120)
    p.comment('TCSinger2 in Max with the piano roll (UI_WORK_PACKAGES.md). Open a score, or draw notes (the pencil, or a double-click), '
              'and type their lyrics in the line under the roll; the roll sends them to the front-end, which makes them singable (see the '
              'console for what it fixed) and gives the phonemes shown over the notes. Generate the F0 (or turn on "regenerate"), then '
              'sing. The roll and its curve are saved with the patch.', 20, 10, w=1100)
    ids = {}
    # the roll and its controls
    ui = p.box('v8ui', 20, 75, w=900, h=420, ins=1, outs=4, outtype=['', '', '', ''], filename='tcs_ui.js', varname='ui',
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

    # the front-end
    fe = p.box('newobj', 20, 540, w=140, text='v8 tcs_max.js', ins=1, outs=3, outtype=['', '', ''])
    ids['fe'] = fe
    p.link(ui, 0, fe)
    p.obj('dict tcs_notes', 300, 540, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the roll's dict alive
    p.obj('dict tcs_cond', 20, 575, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the front-end's dict alive
    p.link(ui, 2, p.obj('print tcs_ui', 480, 540))
    pre_fe = p.obj('prepend fe', 180, 575)
    p.link(fe, 2, pre_fe)
    p.link(pre_fe, 0, ui)
    p.link(fe, 2, p.obj('print tcs', 270, 575))
    p.comment('Style (then bang the front-end):', 940, 75, w=300)
    for k, s in enumerate(STYLES):
        p.link(p.msg(s, 940 + (k % 2) * 160, 100 + (k // 2) * 26), 0, fe)
    show_caption = p.msg('', 20, 640, w=420)
    set_caption = p.obj('prepend set', 20, 610)
    p.link(fe, 1, set_caption)
    p.link(set_caption, 0, show_caption)
    prompt = p.obj('prepend prompt', 460, 610)
    p.link(fe, 1, prompt)
    tok = p.obj('neural.tokenizer tcsinger2_sing', 460, 640, ins=1, outs=2, outtype=['', ''], w=220)

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
    p.link(fe, 0, f0gen)
    # regenerate after each change: after the dict has reached score2f0 (deferlow), generate
    auto = p.box('toggle', 1080, 265, w=22, h=22, ins=1, outs=1, outtype=['int'])
    ids['auto'] = auto
    p.comment('regenerate F0 after each change', 1105, 265, w=180)
    on_dict = p.obj('route dictionary', 1080, 300, ins=1, outs=2, outtype=['', ''])
    p.link(fe, 0, on_dict)
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
    p.link(ui, 1, init)  # the roll wrote the curve (restore)
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
    p.offset = (0, 140)
    sing = sing_section(p, [init], sing_gate, fe, tok)
    p.offset = (0, 0)
    ids['sing'] = sing
    ids['init'] = init
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
