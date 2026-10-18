"""Writes the Max patch of the TCSinger2 export, tcsinger2.maxpat, into export/max/js/, next to the JS front-end
(Max's v8 `require` finds its modules only there) and the model files (symlinks, see export/max/js/README.md):
a score -> the conditions dict -> score2f0 -> the F0 buffer~ (shown and editable in a waveform~) -> sing -> the
audio buffer~ -> playback.

The front-end's dict goes straight to its receivers: through a [t a ...] it arrived empty. Max sends an outlet's
messages right to left by position.

*.maxpat is git-ignored in this repository; this script is how the patches are made again.

Usage (from TCSinger2/):
  python scripts/build_max_patches.py
"""
import json
import os

OUT_DIR = 'export/max/js'


class Patch:
    def __init__(self, width=1200, height=900):
        self.boxes, self.lines, self.size = [], [], (width, height)

    def box(self, maxclass, x, y, w=None, h=22, text=None, ins=1, outs=0, outtype=None, **extra):
        bid = f'obj-{len(self.boxes) + 1}'
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
    styles = ['singer female alto', 'singer male tenor', 'emotion sad', 'emotion happy', 'method bel canto', 'method pop',
              'technique breathy', 'style clear', 'bang', 'status']
    fe = p.box('newobj', 20, 330, w=140, text='v8 tcs_max.js', ins=1, outs=2, outtype=['', ''])
    for k, s in enumerate(styles):
        m = p.msg(s, 180 + (k % 2) * 140, 90 + (k // 2) * 26)
        p.link(m, 0, fe)
    p.link(pre_read, 0, fe)
    fe_status = p.obj('print tcs', 180, 370)
    p.link(fe, 1, fe_status)
    p.obj('dict tcs_cond', 20, 370, ins=2, outs=4, outtype=['dictionary', '', '', ''])  # keeps the front-end's dict alive
    caption = p.obj('dict.unpack caption:', 20, 410, outs=2, outtype=['', ''])
    p.link(fe, 0, caption)
    show_caption = p.msg('', 20, 470, w=420)
    set_caption = p.obj('prepend set', 20, 440)
    p.link(caption, 0, set_caption)
    p.link(set_caption, 0, show_caption)
    prompt = p.obj('prepend prompt', 220, 440)
    p.link(caption, 0, prompt)
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
    for src in (sing_btn, pre_sseed, pre_cfg, init):
        p.link(src, 0, sing)
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

    # inspection
    p.comment('The models\' inputs (neural.info):', 500, 760, w=300)
    p.obj('neural.info tcsinger2_f0', 500, 790, w=200)
    p.obj('neural.info tcsinger2_sing', 720, 790, w=200)
    p.write(os.path.join(OUT_DIR, 'tcsinger2.maxpat'))


if __name__ == '__main__':
    main_patch()
