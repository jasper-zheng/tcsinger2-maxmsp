// tcs_ui_view.js: the axes of the piano roll (UI_WORK_PACKAGES.md, UWP2): time <-> x, MIDI <-> y, scroll, zoom, the
// grid and snap, and the hit tests. No Max globals and no state: every function returns a new view, so tcs_ui.js keeps
// the only copy (Max's `require` gives a new copy of a module on every call).
//
//   {w, h, keysW, rulerH, statusH,        the box and its fixed parts, in px (statusH: the toolbar and status line)
//    beat0, pxPerBeat, midiTop, rowH,     the scroll (the beat at the left edge, the pitch at the top edge) and the zoom
//    maxBeats, beatsPerBar, snap,         how far the roll goes; 4/4 by default; snap "1/1" ... "1/16" or "off"
//    limitSeconds, midiLo, midiHi,        display settings: the shaded limit and the pitch rows (TCSinger2's by default)
//    showPh}                              the front-end's phonemes drawn above the words (UWP5)
// (The mode, notes or F0, is the editor's, tcs_ui_edit.js: the toolbar's layout takes it.)
//
// The roll is the box less the keys (left), the ruler (top) and the toolbar with the status line (bottom). Row m (a
// MIDI note) is centred at midiToY(m), so a note's box and its F0 curve value land on the same y. Snap values are note
// values: 1/4 is one beat. Every function that moves or zooms keeps the view inside its limits (clampView).

"use strict";

const { MAX_SECONDS, MAX_MIDI } = require("tcs_score.js");

const CURVE_FPS = 50; // buffer~ tcs_f0 (EWP D7)
const ROW_MAX = 40;
const ROW_MIN = 3;
const PX_PER_BEAT_MAX = 4000;
const SNAPS = ["1/1", "1/2", "1/4", "1/8", "1/16", "off"];
const EDGE_PX = 6;
const GRID_MIN_PX = 8;
const REST_BAND = 14; // px under the ruler: the rest labels

const DEFAULT = { keysW: 46, rulerH: 32, statusH: 22, beat0: 0, pxPerBeat: 60, midiTop: 76.5, rowH: 14, maxBeats: 20,
  beatsPerBar: 4, snap: "1/4", limitSeconds: MAX_SECONDS, midiLo: 1, midiHi: MAX_MIDI, showPh: true };

const rollW = (v) => Math.max(1, v.w - v.keysW);
const rollH = (v) => Math.max(1, v.h - v.rulerH - v.statusH);
const secondsToBeats = (s, tempo) => (s * tempo) / 60;
const beatsToSeconds = (b, tempo) => (b * 60) / tempo;

function makeView(w, h, opts) {
  return clampView({ ...DEFAULT, ...(opts || {}), w, h });
}

function resize(v, w, h) {
  return clampView({ ...v, w, h });
}

// ---- the axes

const beatToX = (v, b) => v.keysW + (b - v.beat0) * v.pxPerBeat;
const xToBeat = (v, x) => v.beat0 + (x - v.keysW) / v.pxPerBeat;
const midiToY = (v, m) => v.rulerH + (v.midiTop - m) * v.rowH;
const yToMidi = (v, y) => v.midiTop - (y - v.rulerH) / v.rowH;
const rowAt = (v, y) => Math.round(yToMidi(v, y));
const beatEnd = (v) => xToBeat(v, v.w);
const midiBottom = (v) => yToMidi(v, v.h - v.statusH);

// ---- limits

// the length of the roll in beats: the limit (20 s) or the score's end, whichever is later, and a bit more
function extentBeats(model, limitSeconds = MAX_SECONDS) {
  if (!model) return secondsToBeats(limitSeconds, 60) * 1.05;
  const last = model.notes.reduce((m, n) => Math.max(m, n.start + n.dur), 0) + model.tail.reduce((s, r) => s + r.dur, 0);
  return Math.max(secondsToBeats(limitSeconds, model.tempo), last) * 1.05;
}

function clampView(v) {
  const out = { ...v };
  const rw = rollW(out);
  const rh = rollH(out);
  out.pxPerBeat = Math.min(PX_PER_BEAT_MAX, Math.max(rw / out.maxBeats, out.pxPerBeat));
  out.beat0 = Math.min(Math.max(0, out.maxBeats - rw / out.pxPerBeat), Math.max(0, out.beat0));
  const span = out.midiHi - out.midiLo + 1;
  out.rowH = Math.min(ROW_MAX, Math.max(ROW_MIN, rh / span, out.rowH));
  const rows = rh / out.rowH;
  const top = out.midiHi + 0.5;
  const bottom = out.midiLo - 0.5;
  out.midiTop = rows >= top - bottom ? top : Math.min(top, Math.max(bottom + rows, out.midiTop));
  return out;
}

// the roll's length after a new score (or tempo); the scroll is kept where it can be
function setExtent(v, model) {
  return clampView({ ...v, maxBeats: extentBeats(model, v.limitSeconds) });
}

// the shaded limit (seconds; 0 for none) and the pitch rows shown and edited
function setLimit(v, seconds, model) {
  const s = Number(seconds);
  if (!(s >= 0)) throw new Error(`bad limit ${seconds}: seconds, 0 for none`);
  return setExtent({ ...v, limitSeconds: s }, model);
}

function setRange(v, lo, hi) {
  const a = Number(lo);
  const b = Number(hi);
  if (!(Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b <= 127 && a < b)) throw new Error(`bad range ${lo} ${hi}: MIDI 0..127, low < high`);
  return clampView({ ...v, midiLo: a, midiHi: b });
}

// ---- moving and zooming

function scroll(v, dBeats, dRows) {
  return clampView({ ...v, beat0: v.beat0 + dBeats, midiTop: v.midiTop + dRows });
}

// the beat under anchorX stays under it (unless a limit is reached)
function zoomTime(v, factor, anchorX) {
  const x = anchorX === undefined ? v.keysW + rollW(v) / 2 : anchorX;
  const b = xToBeat(v, x);
  const z = clampView({ ...v, pxPerBeat: v.pxPerBeat * factor });
  return clampView({ ...z, beat0: b - (x - z.keysW) / z.pxPerBeat });
}

function zoomPitch(v, factor, anchorY) {
  const y = anchorY === undefined ? v.rulerH + rollH(v) / 2 : anchorY;
  const m = yToMidi(v, y);
  const z = clampView({ ...v, rowH: v.rowH * factor });
  return clampView({ ...z, midiTop: m + (y - z.rulerH) / z.rowH });
}

// show beats b0..b1 and MIDI lo..hi (the rows centred on lo and hi)
function showRange(v, b0, b1, lo, hi) {
  const z = clampView({ ...v, pxPerBeat: rollW(v) / Math.max(1e-6, b1 - b0), rowH: rollH(v) / Math.max(1, hi - lo + 1) });
  const rows = rollH(z) / z.rowH;
  return clampView({ ...z, beat0: b0, midiTop: (lo + hi) / 2 + rows / 2 });
}

// the whole score across the width (from beat 0, at least two bars), its pitch range +-4 rows
function fit(v, model) {
  const z = setExtent(v, model);
  if (!model || !model.notes.length) return showRange(z, 0, secondsToBeats(z.limitSeconds || MAX_SECONDS, model ? model.tempo : 60), 55, 79);
  const last = model.notes.reduce((m, n) => Math.max(m, n.start + n.dur), 0) + model.tail.reduce((s, r) => s + r.dur, 0);
  const lo = Math.min(...model.notes.map((n) => n.midi));
  const hi = Math.max(...model.notes.map((n) => n.midi));
  return showRange(z, 0, Math.max(2 * z.beatsPerBar, last * 1.03), lo - 4, hi + 4);
}

// ---- grid and snap

// the snap value in beats (null for "off")
function snapBeats(snap) {
  if (snap === "off") return null;
  const m = /^1\/(\d+)$/.exec(String(snap));
  if (!m) throw new Error(`bad snap ${snap}: expected one of ${SNAPS.join(", ")}`);
  return 4 / Number(m[1]);
}

function setSnap(v, snap) {
  snapBeats(snap);
  return { ...v, snap: String(snap) };
}

function setBar(v, beats) {
  const n = Number(beats);
  if (!Number.isInteger(n) || n < 1 || n > 16) throw new Error(`bad bar ${beats}: 1..16 beats`);
  return { ...v, beatsPerBar: n };
}

// the step of the finest grid lines: the snap (a beat when off), doubled until the lines are GRID_MIN_PX apart
function gridStep(v) {
  let step = snapBeats(v.snap) || 1;
  while (step * v.pxPerBeat < GRID_MIN_PX) step *= 2;
  return step;
}

function snapBeat(v, b) {
  const s = snapBeats(v.snap);
  return s ? Math.round(b / s) * s : b;
}

// a label step from `steps` (in units) at least minPx apart, at pxPerUnit
function labelStep(pxPerUnit, minPx, steps) {
  for (const s of steps) if (s * pxPerUnit >= minPx) return s;
  let s = steps[steps.length - 1];
  while (s * pxPerUnit < minPx) s *= 2;
  return s;
}

// ---- what is in view, and the hit tests

function visible(v, notes) {
  const b0 = v.beat0;
  const b1 = beatEnd(v);
  const top = v.midiTop + 0.5;
  const bottom = midiBottom(v) - 0.5;
  return notes.filter((n) => n.start < b1 && n.start + n.dur > b0 && n.midi <= top && n.midi >= bottom);
}

// {id, part: "body" | "start" | "end"} of the note under (x, y), or null; the later note wins where notes overlap
function hitNote(v, notes, x, y) {
  if (x < v.keysW || y < v.rulerH || y > v.h - v.statusH) return null;
  let hit = null;
  let hitStart = -Infinity;
  for (const n of notes) {
    const x0 = beatToX(v, n.start);
    const x1 = beatToX(v, n.start + n.dur);
    if (x < x0 || x >= x1 || Math.abs(y - midiToY(v, n.midi)) > v.rowH / 2) continue;
    if (n.start < hitStart || (n.start === hitStart && hit && n.id < hit.id)) continue;
    const e = Math.min(EDGE_PX, (x1 - x0) / 4);
    hit = { id: n.id, part: x < x0 + e ? "start" : x >= x1 - e ? "end" : "body" };
    hitStart = n.start;
  }
  return hit;
}

// the x of a curve frame (50 Hz, in seconds) at a tempo
const frameToX = (v, j, tempo) => beatToX(v, secondsToBeats(j / CURVE_FPS, tempo));

// the ids of the notes that the rectangle (x0, y0)-(x1, y1) touches
function notesInRect(v, notes, x0, y0, x1, y1) {
  const b0 = xToBeat(v, Math.min(x0, x1));
  const b1 = xToBeat(v, Math.max(x0, x1));
  const top = yToMidi(v, Math.min(y0, y1));
  const bottom = yToMidi(v, Math.max(y0, y1));
  return notes.filter((n) => n.start < b1 && n.start + n.dur > b0 && n.midi - 0.5 < top && n.midi + 0.5 > bottom).map((n) => n.id);
}

// the band under the ruler with the rest labels (a click there toggles a rest's type)
const inRestBand = (v, x, y) => x >= v.keysW && y >= v.rulerH && y < v.rulerH + REST_BAND;
const inRoll = (v, x, y) => x >= v.keysW && x < v.w && y >= v.rulerH && y < v.h - v.statusH;

// ---- the toolbar, at the left of the bottom bar: [{id, x, y, w, h, kind}] and where the status text starts. Its
// tools are the mode's (UWP6): the notes' select and pencil and the snap, or the F0 curve's tools.

const TOOLS = {
  notes: [["select", 20, "tool"], ["pencil", 20, "tool"], [null, 8]].concat(SNAPS.map((s) => [`snap:${s}`, s === "1/16" ? 30 : 26, "snap"])),
  f0: [["draw", 20, "tool"], ["line", 20, "tool"], ["smooth", 20, "tool"], ["erase", 20, "tool"], ["unvoice", 20, "tool"]],
};
const toolbar = (mode) => [["mode", 24, "mode"], [null, 8]].concat(TOOLS[mode === "f0" ? "f0" : "notes"])
  .concat([[null, 8], ["undo", 20, "action"], ["redo", 20, "action"], [null, 8], ["fit", 26, "action"], ["ph", 24, "toggle"]]);

// the phonemes shown or not
function setPhonemes(v, on) {
  return { ...v, showPh: !!on };
}

function toolbarLayout(v, mode = "notes") {
  const y = v.h - v.statusH + 2;
  const h = Math.max(1, v.statusH - 4);
  const buttons = [];
  let x = 4;
  for (const [id, w, kind] of toolbar(mode)) {
    if (id) buttons.push({ id, x, y, w, h, kind });
    x += w + (id ? 2 : 0);
  }
  return { buttons, textX: x + 10 };
}

function hitToolbar(v, x, y, mode = "notes") {
  if (y < v.h - v.statusH) return null;
  const b = toolbarLayout(v, mode).buttons.find((r) => x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h);
  return b ? b.id : null;
}

// {frame, midi} of the voiced curve point nearest to (x, y), if it is within tolPx; null otherwise
function hitCurve(v, curve, tempo, x, y, tolPx = 5) {
  if (!curve || !curve.length) return null;
  const pxPerFrame = (v.pxPerBeat * tempo) / 60 / CURVE_FPS;
  const j0 = Math.round(beatsToSeconds(xToBeat(v, x), tempo) * CURVE_FPS);
  const k = Math.ceil(tolPx / pxPerFrame) + 1;
  let best = null;
  let bestD = tolPx;
  for (let j = Math.max(0, j0 - k); j <= Math.min(curve.length - 1, j0 + k); j++) {
    if (!(curve[j] > 0)) continue;
    const d = Math.hypot(frameToX(v, j, tempo) - x, midiToY(v, curve[j]) - y);
    if (d <= bestD) { bestD = d; best = { frame: j, midi: curve[j] }; }
  }
  return best;
}

module.exports = {
  makeView, resize, beatToX, xToBeat, midiToY, yToMidi, rowAt, beatEnd, midiBottom, rollW, rollH, secondsToBeats,
  beatsToSeconds, extentBeats, clampView, setExtent, scroll, zoomTime, zoomPitch, showRange, fit, snapBeats, setSnap,
  setBar, setLimit, setRange, setPhonemes, gridStep, snapBeat, labelStep, visible, hitNote, notesInRect, inRestBand, inRoll, toolbarLayout,
  hitToolbar, frameToX, hitCurve, SNAPS, MAX_MIDI, MAX_SECONDS, CURVE_FPS, REST_BAND,
};
