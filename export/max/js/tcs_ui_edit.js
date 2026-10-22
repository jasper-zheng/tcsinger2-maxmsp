// tcs_ui_edit.js: editing the piano roll by mouse, keys and commands (UI_WORK_PACKAGES.md, UWP4, UWP6). No Max globals
// and no state: tcs_ui.js routes its handlers here and keeps what comes back, so Node can run the same events as Max.
//
//   ed = {model, curve, gen, history, selection: [ids], mode: "notes" | "f0", tool: "select" | "pencil",
//         f0tool: "draw" | "line" | "smooth" | "erase" | "unvoice", clip: [notes], gesture, pointer: {x, y} | null}
//     curve, gen  the working F0 curve (edited, previewed while a stroke is drawn) and the generated one (U3), or null
//   every function -> {ed, view?, commit?: label, curve?: label, message?: text, consumed?: bool}
//     view    the view, when it changed (snap, fit)
//     commit  the model changed for good: tcs_ui.js sends the notes (history already has the step)
//     curve   the working curve changed for good: tcs_ui.js writes it into the buffer (one poke) and bangs outlet 1
//     message a short note for the status line (e.g. "join: only touching notes")
//     focus   "lyrics": the lyrics line should take the keys (a double-click on a note, UWP5)
//     transport  {at: seconds} (a press on the ruler: play from there) | "toggle" (Space: play or stop) (UWP7)
//
// One history holds both (UWP6): a step changes the notes or the curve, and undo puts back whichever it was. A new
// generated curve (setCurve, on f0 refresh) replaces the curve in every step and drops the curve-only ones (U4).
//
// A gesture (press, drag..., release) previews its result in ed.model, always computed from the model as it was when
// the gesture began (gesture.base), so what is previewed is what is committed; history gets one step, on release.
// Nothing here checks the notes (U10): moves and resizes may overlap other notes, and tcs_notes.js sorts that out.
//
// Mouse (select tool): a note's body selects it (Shift toggles; a selected note keeps the group) and drags the
// selection (snapped; Cmd: free); a note's edge resizes it, and where another note touches that edge the boundary moves
// both (Alt: only this note); empty space draws a rubber band (Shift adds); a rest label toggles AP/SP; a double-click
// on empty space adds a note, and on a note selects it for its lyric (UWP5). Pencil tool: a press on empty space adds a note and dragging right sets its length; on a
// note it acts as the select tool.
// Keys: Backspace deletes; up/down transpose (Shift: an octave); left/right nudge by the snap (Shift: a bar); Cmd-A,
// Esc; Cmd-Z, Cmd-Shift-Z; Cmd-C/X/V/D; B toggles the pencil; S splits at the pointer; J joins; F: the F0 mode.
// Playback (UWP7, both modes): a press on the ruler plays from there; Space plays or stops.
//
// The F0 mode (UWP6): the mouse edits the curve and the notes can't be edited. draw: a stroke, linear between the
// pointer's points; line: from the press to the release; smooth, erase (back to the generated curve) and unvoice:
// brushes over the frames within BRUSH_PX of the pointer's path. Shift: a stroke in whole semitones, a line from
// semitone to semitone. Each
// gesture is one step. Keys: Esc, Cmd-Z, Cmd-Shift-Z, Cmd-A (the notes for the range commands), F back to the notes,
// D/L/S/E/U the tools; the note keys do nothing there (and are not consumed). The range commands (f0: offset,
// vibrato, unvoice, reset, snap, smooth) act on the selected notes' time spans, or the whole curve with none
// selected, or a range given in seconds (one that doesn't end after it starts, e.g. 0 0, counts as none).

"use strict";

const M = require("tcs_ui_model.js");
const V = require("tcs_ui_view.js");
const F = require("tcs_ui_f0.js");

const DRAG_PX = 3; // a press moves nothing until the pointer has moved this far
const MIN_OFF = 1 / 32; // the shortest note with snap off, in beats
const NUDGE_OFF = 1 / 4; // the arrow step with snap off
const BRUSH_PX = 8; // the F0 brushes: the frames this close to the pointer's path
const NOTE_TOOLS = ["select", "pencil"];
const F0_TOOLS = ["draw", "line", "smooth", "erase", "unvoice"];
const F0_KEYS = { d: "draw", l: "line", s: "smooth", e: "erase", u: "unvoice" };
const EPS = M.EPS;

const KEY = { backspace: -7, esc: -3, space: -2 };
const PLAY_TAIL = 0.5; // s: playback goes this far past the phrase's end
const CHAR = { left: 28, right: 29, up: 30, down: 31, backspace: 127, del: 8, esc: 27 };

const end = (n) => n.start + n.dur;
const round9 = (x) => Math.round(x * 1e9) / 1e9;

function create(model, tool = "select", curve = null, gen = curve) {
  return { model, curve, gen, history: M.history(model, curve), selection: [], mode: "notes", tool, f0tool: "draw", clip: [],
    gesture: null, pointer: null };
}

// a new model (a load or a restore): no history, no selection; the curve stays unless one is given (a restore)
function reset(ed, model, curve = ed.history.curve, gen = ed.gen) {
  return { ...ed, model, curve, gen, history: M.history(model, curve), selection: [], gesture: null };
}

// a new generated curve (f0 refresh), or none: it replaces the working one with no step, and the curve-only steps go
// (U4); a stroke being drawn is dropped. -> {ed, dropped: the hand-edited frames that were lost}
function setCurve(ed, gen) {
  const dropped = F.edited(ed.gen, ed.history.curve).frames;
  const f0Gesture = ed.gesture && ed.gesture.kind.startsWith("f0");
  return { ed: { ...ed, gen, curve: gen, history: M.newCurve(ed.history, gen), gesture: f0Gesture ? null : ed.gesture }, dropped };
}

// ---- helpers

const step = (v) => V.snapBeats(v.snap) || 1; // a new note's length
const minLen = (v, free) => (free ? MIN_OFF : V.snapBeats(v.snap) || MIN_OFF); // one snap step (Cmd: free)
const range = (v) => [v.midiLo, v.midiHi];

function floorSnap(v, b) {
  const s = V.snapBeats(v.snap);
  return s ? Math.floor(b / s + 1e-9) * s : b;
}

function snapTo(v, b, free) {
  return free ? b : V.snapBeat(v, b);
}

function rowIn(v, y) {
  return Math.min(v.midiHi, Math.max(v.midiLo, V.rowAt(v, y)));
}

const existing = (model, ids) => ids.filter((id) => model.notes.some((n) => n.id === id));

const same = (a, b) => a === b || JSON.stringify(a) === JSON.stringify(b);

// a committed change: history gets the step (from history.present, the model before the change); no change, no step
function commit(ed, model, label, extra) {
  if (same(model, ed.history.present)) return { ed: { ...ed, ...(extra || {}), model: ed.history.present, gesture: null } };
  const history = M.push(ed.history, model, label);
  return { ed: { ...ed, ...(extra || {}), model, history, gesture: null }, commit: label };
}

// a committed change of the working curve; no change, no step
function commitCurve(ed, curve, label) {
  if (same(curve, ed.history.curve)) return { ed: { ...ed, curve: ed.history.curve, gesture: null } };
  const history = M.push(ed.history, ed.history.present, label, curve);
  return { ed: { ...ed, curve, history, gesture: null }, curve: label };
}

// a command on the committed model; a ModelError becomes a message
function apply(ed, label, fn, extra) {
  let model;
  try {
    model = fn(ed.history.present);
  } catch (e) {
    return { ed, message: `${label}: ${e.message}` };
  }
  return commit(ed, model, label, typeof extra === "function" ? extra(model) : extra);
}

// ---- commands (also the messages of tcs_ui.js)

// a tool of either mode; it selects its mode
function setTool(ed, tool) {
  if (NOTE_TOOLS.includes(tool)) return { ed: { ...ed, tool, mode: "notes" } };
  if (F0_TOOLS.includes(tool)) return { ed: { ...ed, f0tool: tool, mode: "f0" } };
  return { ed, message: `tool: expected ${NOTE_TOOLS.concat(F0_TOOLS).join(", ")}, not ${tool}` };
}

function setMode(ed, mode) {
  if (mode !== "notes" && mode !== "f0") return { ed, message: `mode: expected notes or f0, not ${mode}` };
  if (ed.gesture) return { ed }; // not in the middle of a drag
  return { ed: { ...ed, mode } };
}

function select(ed, ids) {
  return { ed: { ...ed, selection: existing(ed.model, ids) } };
}

function selectAll(ed) {
  return { ed: { ...ed, selection: ed.model.notes.map((n) => n.id) } };
}

function addNote(ed, start, dur, midi, lyric) {
  let id = null;
  const r = apply(ed, "add note", (m) => { const a = M.addNote(m, { start, dur, midi, lyric }); id = a.id; return a.model; });
  if (r.commit) r.ed.selection = [id];
  return r;
}

function del(ed, ids) {
  const which = ids && ids.length ? ids : ed.selection;
  if (!which.length) return { ed };
  return apply(ed, "delete", (m) => M.deleteNotes(m, which), { selection: ed.selection.filter((id) => !which.includes(id)) });
}

function move(ed, v, dBeats, dMidi, ids) {
  const which = ids && ids.length ? ids : ed.selection;
  if (!which.length) return { ed };
  return apply(ed, dBeats ? "move" : "transpose", (m) => M.moveNotes(m, which, dBeats, dMidi, range(v)));
}

function resize(ed, id, start, stop) {
  return apply(ed, "resize", (m) => M.resizeNote(m, id, start, stop));
}

function split(ed, ids, at) {
  return apply(ed, "split", (m) => {
    let out = m;
    let any = false;
    for (const id of ids) {
      const n = out.notes.find((x) => x.id === id);
      if (n && at > n.start + EPS && at < end(n) - EPS) { out = M.splitNote(out, id, at); any = true; }
    }
    if (!any) throw new Error(`no selected note crosses beat ${round9(at)}`);
    return out;
  });
}

function join(ed, ids) {
  const which = ids && ids.length ? ids : ed.selection;
  return apply(ed, "join", (m) => M.joinNotes(m, which), (m) => ({ selection: existing(m, which) }));
}

// the rest at beat b: its type set (AP, SP) or toggled
function rest(ed, b, type = "toggle") {
  const r = M.restAt(ed.history.present, b);
  if (!r) return { ed, message: `rest: no rest at beat ${round9(b)}` };
  const t = type === "toggle" ? (r.type === "AP" ? "SP" : "AP") : type;
  return apply(ed, "rest type", (m) => M.setRestType(m, r.gapKey, r.index, t));
}

// a step back or forward: `commit` if the notes changed, `curve` if the curve did
function travel(ed, history, label) {
  const r = { ed: { ...ed, history, model: history.present, curve: history.curve, gesture: null,
    selection: existing(history.present, ed.selection) } };
  if (history.present !== ed.history.present) r.commit = label;
  if (history.curve !== ed.history.curve) r.curve = label;
  return r;
}

function undo(ed) {
  if (!ed.history.past.length) return { ed };
  return travel(ed, M.undo(ed.history), "undo");
}

function redo(ed) {
  if (!ed.history.future.length) return { ed };
  return travel(ed, M.redo(ed.history), "redo");
}

function copy(ed) {
  return { ed: { ...ed, clip: M.copyNotes(ed.model, ed.selection) } };
}

function cut(ed) {
  const c = copy(ed).ed;
  return del(c, c.selection);
}

function pasteAt(ed, label, clip, at) {
  if (!clip.length) return { ed };
  let ids = [];
  const r = apply(ed, label, (m) => { const p = M.pasteNotes(m, clip, at); ids = p.ids; return p.model; });
  if (r.commit) r.ed.selection = ids;
  return r;
}

// at beat `at`; else at the snapped pointer, if it is over the roll; else after the last note
function paste(ed, v, at) {
  let b = at;
  if (b === undefined || b === null) {
    const p = ed.pointer;
    if (p && V.inRoll(v, p.x, p.y)) b = Math.max(0, V.snapBeat(v, V.xToBeat(v, p.x)));
    else b = ed.model.notes.reduce((mx, n) => Math.max(mx, end(n)), 0);
  }
  return pasteAt(ed, "paste", ed.clip, b);
}

// the selection again, right after its last note
function duplicate(ed) {
  const picked = ed.model.notes.filter((n) => ed.selection.includes(n.id));
  if (!picked.length) return { ed };
  return pasteAt(ed, "duplicate", M.copyNotes(ed.model, ed.selection), Math.max(...picked.map(end)));
}

// the lyrics line of the [textedit] (U5, UWP5): it fills the notes of M.lyricTargets (the selection, or all when
// nothing is selected); entries past the notes it fills are dropped, with a message
function lyrics(ed, line) {
  let warnings = [];
  const r = apply(ed, "lyrics", (m) => { const s = M.setLyrics(m, ed.selection, line); warnings = s.warnings; return s.model; });
  if (warnings.length && !r.message) r.message = `lyrics: ${warnings.join("; ")}`;
  return r;
}

// ---- the F0 curve (UWP6)

const NO_CURVE = "no F0 curve: generate it first";

// a range command on the committed curve, one step: offset <st>, vibrato <rate> <depth>, unvoice, reset, snap, smooth,
// each with an optional range <s0> <s1> in seconds (else, or when s1 <= s0, e.g. 0 0: the selected notes' spans, or the
// whole curve); and
// draw <s> <midi> <s> <midi>..., line <s0> <m0> <s1> <m1>
function f0(ed, v, op, args = []) {
  const curve = ed.history.curve;
  const label = `F0 ${op}`;
  if (!curve) return { ed, message: `${label}: ${NO_CURVE}` };
  const a = args.map(Number);
  if (a.some((x) => !Number.isFinite(x))) return { ed, message: `${label}: expected numbers` };
  const model = ed.history.present;
  const nargs = { offset: 1, vibrato: 2, unvoice: 0, reset: 0, snap: 0, smooth: 0 }[op];
  if (op === "draw" || op === "line") {
    if (a.length < 2 || a.length % 2 || (op === "line" && a.length !== 4)) {
      return { ed, message: `${label}: expected ${op === "line" ? "<s0> <m0> <s1> <m1>" : "<s> <midi> pairs"}` };
    }
    const points = [];
    for (let k = 0; k < a.length; k += 2) points.push([a[k] * F.FPS, a[k + 1]]);
    return commitCurve(ed, F.draw(curve, points, v.midiLo, v.midiHi), label);
  }
  if (nargs === undefined) return { ed, message: `f0: unknown command ${op}` };
  if (a.length !== nargs && a.length !== nargs + 2) return { ed, message: `${label}: expected ${nargs} number(s), then <s0> <s1> or nothing` };
  const given = a.length === nargs + 2 && a[nargs + 1] > a[nargs]; // a range that ends before it starts (0 0): none
  const sel = existing(model, ed.selection);
  const ranges = given ? [[a[nargs], a[nargs + 1]]] : sel.length ? F.spans(model, sel) : F.whole();
  const js = F.frames(curve.length, ranges);
  let out;
  if (op === "offset") out = F.offset(curve, js, a[0]);
  else if (op === "vibrato") out = F.vibrato(curve, ranges.map(([s0, s1]) => [s0, Math.min(s1, curve.length / F.FPS)]), a[0], a[1]);
  else if (op === "unvoice") out = F.unvoice(curve, js);
  else if (op === "reset") out = F.reset(curve, ed.gen, js);
  else if (op === "smooth") out = F.smooth(curve, js);
  else out = F.snap(curve, given || !sel.length ? model.notes : model.notes.filter((n) => sel.includes(n.id)), model.tempo, js);
  return commitCurve(ed, out, label);
}

// the frame (not rounded) and the pitch at a pointer position
function framePoint(ed, v, x, y, mods) {
  const n = ed.history.curve.length;
  const f = Math.min(n - 1, Math.max(0, V.beatsToSeconds(V.xToBeat(v, x), ed.history.present.tempo) * F.FPS));
  const m = Math.min(v.midiHi, Math.max(v.midiLo, V.yToMidi(v, y)));
  return [f, mods.shift ? Math.round(m) : m];
}

// a brush over the frames within BRUSH_PX of the path from xa to xb
function brush(ed, v, curve, tool, xa, xb) {
  const tempo = ed.history.present.tempo;
  const at = (x) => V.beatsToSeconds(V.xToBeat(v, x), tempo) * F.FPS;
  const j0 = Math.max(0, Math.ceil(at(Math.min(xa, xb) - BRUSH_PX)));
  const j1 = Math.min(curve.length - 1, Math.floor(at(Math.max(xa, xb) + BRUSH_PX)));
  const js = [];
  for (let j = j0; j <= j1; j++) js.push(j);
  if (tool === "smooth") return F.smooth(curve, js);
  if (tool === "erase") return F.reset(curve, ed.gen, js);
  return F.unvoice(curve, js);
}

function f0Press(e, v, x, y, mods) {
  if (!e.history.curve) return { ed: e, message: NO_CURVE };
  const curve0 = e.history.curve;
  const p = framePoint(e, v, x, y, mods);
  const tool = e.f0tool;
  if (tool === "draw") return { ed: { ...e, curve: F.draw(curve0, [p], v.midiLo, v.midiHi), gesture: { kind: "f0draw", curve0, last: p } } };
  // (Shift: the point is a semitone already; a stroke rounds every frame, a line glides from semitone to semitone)
  if (tool === "line") return { ed: { ...e, curve: F.draw(curve0, [p], v.midiLo, v.midiHi), gesture: { kind: "f0line", curve0, p0: p } } };
  return { ed: { ...e, curve: brush(e, v, curve0, tool, x, x), gesture: { kind: "f0brush", curve0, tool, lastX: x } } };
}

function f0Drag(e, v, g, x, y, mods) {
  const p = framePoint(e, v, x, y, mods);
  if (g.kind === "f0draw") return { ed: { ...e, curve: F.draw(e.curve, [g.last, p], v.midiLo, v.midiHi, !!mods.shift), gesture: { ...g, last: p } } };
  if (g.kind === "f0line") return { ed: { ...e, curve: F.draw(g.curve0, [g.p0, p], v.midiLo, v.midiHi) } };
  return { ed: { ...e, curve: brush(e, v, e.curve, g.tool, g.lastX, x), gesture: { ...g, lastX: x } } };
}

// ---- the mouse

function press(ed, v, x, y, mods = {}) {
  const e = { ...ed, pointer: { x, y }, gesture: null };
  const button = V.hitToolbar(v, x, y, e.mode);
  if (button) return toolbar(e, v, button);
  if (x >= v.keysW && x < v.w && y >= 0 && y < v.rulerH) {
    return { ed: e, transport: { at: Math.max(0, V.beatsToSeconds(V.xToBeat(v, x), e.history.present.tempo)) } };
  }
  if (!V.inRoll(v, x, y)) return { ed: e };
  if (e.mode === "f0") return f0Press(e, v, x, y, mods);
  const base = e.history.present;
  const hit = V.hitNote(v, base.notes, x, y);
  if (hit) {
    const was = e.selection.includes(hit.id);
    let selection = e.selection;
    if (mods.shift) selection = was ? selection.filter((id) => id !== hit.id) : selection.concat([hit.id]);
    else if (!was) selection = [hit.id];
    if (mods.shift && was) return { ed: { ...e, selection } }; // toggled off: nothing to drag
    const n = base.notes.find((m) => m.id === hit.id);
    if (hit.part === "body") return { ed: { ...e, selection, gesture: { kind: "move", id: hit.id, base, x0: x, y0: y, moved: false, was } } };
    const partners = mods.alt ? [] : base.notes.filter((p) => p.id !== n.id &&
      (hit.part === "end" ? Math.abs(p.start - end(n)) < EPS : Math.abs(end(p) - n.start) < EPS)).map((p) => p.id);
    return { ed: { ...e, selection, gesture: { kind: "resize", id: hit.id, edge: hit.part, base, partners, x0: x } } };
  }
  if (V.inRestBand(v, x, y) && M.restAt(base, V.xToBeat(v, x))) return rest(e, V.xToBeat(v, x));
  if (e.tool === "pencil") {
    const start = Math.max(0, floorSnap(v, V.xToBeat(v, x)));
    const midi = rowIn(v, y);
    const a = M.addNote(base, { start, dur: step(v), midi });
    return { ed: { ...e, model: a.model, selection: [a.id], gesture: { kind: "draw", id: a.id, base, start, midi, moved: false } } };
  }
  const before = mods.shift ? e.selection : [];
  return { ed: { ...e, selection: before, gesture: { kind: "rubber", x0: x, y0: y, x1: x, y1: y, before } } };
}

function toolbar(ed, v, id) {
  if (id === "mode") return setMode(ed, ed.mode === "f0" ? "notes" : "f0");
  if (NOTE_TOOLS.includes(id) || F0_TOOLS.includes(id)) return setTool(ed, id);
  if (id.startsWith("snap:")) return { ed, view: V.setSnap(v, id.slice(5)) };
  if (id === "undo") return undo(ed);
  if (id === "redo") return redo(ed);
  if (id === "fit") return { ed, view: V.fit(v, ed.model) };
  if (id === "ph") return { ed, view: V.setPhonemes(v, !v.showPh) };
  return { ed };
}

function drag(ed, v, x, y, mods = {}) {
  const g = ed.gesture;
  const e = { ...ed, pointer: { x, y } };
  if (!g) return { ed: e };
  if (g.kind.startsWith("f0")) return f0Drag(e, v, g, x, y, mods);
  const free = !!mods.cmd;
  if (g.kind === "move") {
    if (!g.moved && Math.hypot(x - g.x0, y - g.y0) < DRAG_PX) return { ed: e };
    const n = g.base.notes.find((m) => m.id === g.id);
    const target = snapTo(v, n.start + V.xToBeat(v, x) - V.xToBeat(v, g.x0), free);
    const model = M.moveNotes(g.base, e.selection, target - n.start, V.rowAt(v, y) - V.rowAt(v, g.y0), range(v));
    return { ed: { ...e, model, gesture: { ...g, moved: true } } };
  }
  if (g.kind === "resize") {
    const n = g.base.notes.find((m) => m.id === g.id);
    const ps = g.base.notes.filter((p) => g.partners.includes(p.id));
    const min = minLen(v, free);
    const b = snapTo(v, V.xToBeat(v, x), free);
    let list;
    if (g.edge === "end") {
      const lo = n.start + min;
      const hi = ps.length ? Math.min(...ps.map(end)) - min : Infinity;
      if (hi < lo) return { ed: e };
      const at = Math.min(hi, Math.max(lo, b));
      list = [{ id: n.id, start: n.start, stop: at }].concat(ps.map((p) => ({ id: p.id, start: at, stop: end(p) })));
    } else {
      const lo = ps.length ? Math.max(...ps.map((p) => p.start)) + min : 0;
      const hi = end(n) - min;
      if (hi < lo) return { ed: e };
      const at = Math.min(hi, Math.max(lo, b));
      list = [{ id: n.id, start: at, stop: end(n) }].concat(ps.map((p) => ({ id: p.id, start: p.start, stop: at })));
    }
    return { ed: { ...e, model: M.resizeNotes(g.base, list) } };
  }
  if (g.kind === "draw") {
    const stop = Math.max(g.start + minLen(v, free), snapTo(v, V.xToBeat(v, x), free));
    const model = M.addNote(g.base, { start: g.start, dur: round9(stop - g.start), midi: g.midi }).model;
    return { ed: { ...e, model, gesture: { ...g, moved: true } } };
  }
  if (g.kind === "rubber") {
    const inside = V.notesInRect(v, e.model.notes, g.x0, g.y0, x, y);
    const selection = g.before.concat(inside.filter((id) => !g.before.includes(id)));
    return { ed: { ...e, selection, gesture: { ...g, x1: x, y1: y } } };
  }
  return { ed: e };
}

function release(ed, v, x, y, mods = {}) {
  const g = ed.gesture;
  if (!g) return { ed: { ...ed, pointer: { x, y } } };
  const r = drag(ed, v, x, y, mods);
  const e = { ...r.ed, gesture: null };
  if (g.kind.startsWith("f0")) return commitCurve(e, e.curve, { f0draw: "draw F0", f0line: "F0 line" }[g.kind] || `F0 ${g.tool}`);
  if (g.kind === "move" && !r.ed.gesture.moved) {
    // a click: a note of a group becomes the selection
    return { ed: { ...e, model: g.base, selection: g.was && !mods.shift ? [g.id] : e.selection } };
  }
  if (g.kind === "move") return commit({ ...e, model: g.base }, e.model, "move");
  if (g.kind === "resize") return commit({ ...e, model: g.base }, e.model, "resize");
  if (g.kind === "draw") return commit({ ...e, model: g.base }, e.model, "add note");
  return { ed: e };
}

// select tool: a note on empty space (pencil: the presses have done it)
function dblclick(ed, v, x, y) {
  const e = { ...ed, pointer: { x, y }, gesture: null };
  if (e.mode !== "notes" || e.tool !== "select" || !V.inRoll(v, x, y)) return { ed: e };
  const base = e.history.present;
  const hit = V.hitNote(v, base.notes, x, y);
  if (hit) return { ed: { ...e, selection: [hit.id] }, focus: "lyrics" }; // its lyric: the line of that note alone
  if (V.inRestBand(v, x, y) && M.restAt(base, V.xToBeat(v, x))) return { ed: e };
  return addNote(e, Math.max(0, floorSnap(v, V.xToBeat(v, x))), step(v), rowIn(v, y));
}

// the gesture undone (Esc)
function cancel(ed) {
  if (!ed.gesture) return { ed };
  if (ed.gesture.kind.startsWith("f0")) return { ed: { ...ed, curve: ed.history.curve, gesture: null } };
  return { ed: { ...ed, model: ed.gesture.base || ed.model, gesture: null } };
}

// ---- the keys: onkeydown(code, char, updown, cmd, shift, capslock, alt, ctrl) (UWP0)

function letter(code, char) {
  const c = code > 0 ? code : char;
  return c > 0 ? String.fromCharCode(c).toLowerCase() : "";
}

function key(ed, v, code, char, mods = {}) {
  const done = (r) => ({ ...r, consumed: true });
  if (code === KEY.esc || char === CHAR.esc) return done(ed.gesture ? cancel(ed) : { ed: { ...ed, selection: [] } });
  if (ed.gesture) return { ed, consumed: true }; // no keys in the middle of a drag
  if ((code === KEY.space || char === 32) && !mods.cmd) return { ed, transport: "toggle", consumed: true };
  if (ed.mode === "f0") return f0Key(ed, code, char, mods);
  if (code === KEY.backspace || char === CHAR.backspace || char === CHAR.del) return done(del(ed));
  if (char === CHAR.up || char === CHAR.down) return done(move(ed, v, 0, (char === CHAR.up ? 1 : -1) * (mods.shift ? 12 : 1)));
  if (char === CHAR.left || char === CHAR.right) {
    const d = mods.shift ? v.beatsPerBar : V.snapBeats(v.snap) || NUDGE_OFF;
    return done(move(ed, v, char === CHAR.right ? d : -d, 0));
  }
  const k = letter(code, char);
  if (mods.cmd) {
    if (k === "a") return done(selectAll(ed));
    if (k === "z") return done(mods.shift ? redo(ed) : undo(ed));
    if (k === "c") return done(copy(ed));
    if (k === "x") return done(cut(ed));
    if (k === "v") return done(paste(ed, v));
    if (k === "d") return done(duplicate(ed));
    return { ed, consumed: false };
  }
  if (k === "b") return done(setTool(ed, ed.tool === "pencil" ? "select" : "pencil"));
  if (k === "f") return done(setMode(ed, "f0"));
  if (k === "s") return done(splitAtPointer(ed, v));
  if (k === "j") return done(join(ed));
  return { ed, consumed: false };
}

// the keys of the F0 mode: the note keys do nothing (and are not consumed), so a note can't be deleted unseen
function f0Key(ed, code, char, mods) {
  const done = (r) => ({ ...r, consumed: true });
  const k = letter(code, char);
  if (mods.cmd) {
    if (k === "a") return done(selectAll(ed));
    if (k === "z") return done(mods.shift ? redo(ed) : undo(ed));
    return { ed, consumed: false };
  }
  if (k === "f") return done(setMode(ed, "notes"));
  if (F0_KEYS[k]) return done(setTool(ed, F0_KEYS[k]));
  return { ed, consumed: false };
}

// ---- playback (UWP7)

// what a playback from `at` seconds plays: to the phrase's end and a little more (at least PLAY_TAIL), within 20 s
function playRange(model, at) {
  const to = Math.min(V.MAX_SECONDS, Math.max(M.seconds(model), at) + PLAY_TAIL);
  return { from: Math.min(Math.max(0, at), to), to };
}

// S: the note under the pointer (else the selected notes) split at the snapped pointer beat
function splitAtPointer(ed, v) {
  const p = ed.pointer;
  if (!p || !V.inRoll(v, p.x, p.y)) return { ed, message: "split: point at a note" };
  const at = V.snapBeat(v, V.xToBeat(v, p.x));
  const hit = V.hitNote(v, ed.model.notes, p.x, p.y);
  return split(ed, hit ? [hit.id] : ed.selection, at);
}

module.exports = {
  create, reset, setCurve, playRange, apply, commit, commitCurve, setTool, setMode, f0, select, selectAll, addNote, del, move, resize, split, join, rest, lyrics, undo, redo,
  copy, cut, paste, duplicate, press, drag, release, dblclick, cancel, key, KEY, CHAR, NOTE_TOOLS, F0_TOOLS,
};
