// tcs_ui_edit.js: editing the piano roll by mouse, keys and commands (UI_WORK_PACKAGES.md, UWP4). No Max globals and
// no state: tcs_ui.js routes its handlers here and keeps what comes back, so Node can run the same events as Max.
//
//   ed = {model, history, selection: [ids], tool: "select" | "pencil", clip: [notes], gesture, pointer: {x, y} | null}
//   every function -> {ed, view?, commit?: label, message?: text, consumed?: bool}
//     view    the view, when it changed (snap, fit)
//     commit  the model changed for good: tcs_ui.js sends the notes (history already has the step)
//     message a short note for the status line (e.g. "join: only touching notes")
//     focus   "lyrics": the lyrics line should take the keys (a double-click on a note, UWP5)
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
// Esc; Cmd-Z, Cmd-Shift-Z; Cmd-C/X/V/D; B toggles the pencil; S splits at the pointer; J joins.

"use strict";

const M = require("tcs_ui_model.js");
const V = require("tcs_ui_view.js");

const DRAG_PX = 3; // a press moves nothing until the pointer has moved this far
const MIN_OFF = 1 / 32; // the shortest note with snap off, in beats
const NUDGE_OFF = 1 / 4; // the arrow step with snap off
const EPS = M.EPS;

const KEY = { backspace: -7, esc: -3 };
const CHAR = { left: 28, right: 29, up: 30, down: 31, backspace: 127, del: 8, esc: 27 };

const end = (n) => n.start + n.dur;
const round9 = (x) => Math.round(x * 1e9) / 1e9;

function create(model, tool = "select") {
  return { model, history: M.history(model), selection: [], tool, clip: [], gesture: null, pointer: null };
}

// a new model (a load or a restore): no history, no selection
function reset(ed, model) {
  return { ...ed, model, history: M.history(model), selection: [], gesture: null };
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

function setTool(ed, tool) {
  if (tool !== "select" && tool !== "pencil") return { ed, message: `tool: expected select or pencil, not ${tool}` };
  return { ed: { ...ed, tool } };
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

function undo(ed) {
  if (!ed.history.past.length) return { ed };
  const history = M.undo(ed.history);
  return { ed: { ...ed, history, model: history.present, gesture: null, selection: existing(history.present, ed.selection) }, commit: "undo" };
}

function redo(ed) {
  if (!ed.history.future.length) return { ed };
  const history = M.redo(ed.history);
  return { ed: { ...ed, history, model: history.present, gesture: null, selection: existing(history.present, ed.selection) }, commit: "redo" };
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

// ---- the mouse

function press(ed, v, x, y, mods = {}) {
  const e = { ...ed, pointer: { x, y }, gesture: null };
  const button = V.hitToolbar(v, x, y);
  if (button) return toolbar(e, v, button);
  if (!V.inRoll(v, x, y)) return { ed: e };
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
  if (id === "select" || id === "pencil") return setTool(ed, id);
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
  if (e.tool !== "select" || !V.inRoll(v, x, y)) return { ed: e };
  const base = e.history.present;
  const hit = V.hitNote(v, base.notes, x, y);
  if (hit) return { ed: { ...e, selection: [hit.id] }, focus: "lyrics" }; // its lyric: the line of that note alone
  if (V.inRestBand(v, x, y) && M.restAt(base, V.xToBeat(v, x))) return { ed: e };
  return addNote(e, Math.max(0, floorSnap(v, V.xToBeat(v, x))), step(v), rowIn(v, y));
}

// the gesture undone (Esc)
function cancel(ed) {
  if (!ed.gesture) return { ed };
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
  if (k === "s") return done(splitAtPointer(ed, v));
  if (k === "j") return done(join(ed));
  return { ed, consumed: false };
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
  create, reset, apply, commit, setTool, select, selectAll, addNote, del, move, resize, split, join, rest, lyrics, undo, redo,
  copy, cut, paste, duplicate, press, drag, release, dblclick, cancel, key, KEY, CHAR,
};
