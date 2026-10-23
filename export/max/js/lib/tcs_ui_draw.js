// tcs_ui_draw.js: drawing the piano roll (UI_WORK_PACKAGES.md, UWP2), given an mgraphics context `g` (the v8ui's
// mgraphics, or an offscreen MGraphics for a snapshot). No other Max globals, so Node can run it on a fake context.
//
//   draw(g, view, scene) -> summary
//   scene: {model, gaps (tcs_ui_model.gaps), overlaps (Set of ids, drawn see-through), curve (MIDI at 50 Hz, 0 =
//           unvoiced: the working curve), gen (the generated one), curveStale, selection (Set of ids), hover ({id, part}
//           | null), hoverRest ({start} | null), playhead (seconds | null), rubber ({x0, y0, x1, y1} | null), mode
//           ("notes" | "f0"), tool ("select" | "pencil"), f0tool, canUndo, canRedo, status (text), phonemes
//           (phonemeLabels of the front-end's words, or null), region ({s0, s1} seconds | null: the range strip, UWP9)}
//   summary: {notes: [ids drawn], lyrics: [{id, text}], rests: [{type, start, dur, shown}], curveRuns, curvePoints,
//             editedRuns (drawn in the accent), buttons: [ids of the toolbar], phonemes: [{id, text, source}],
//             playhead (the x of its marker on the ruler, or null), region ({x0, x1} of its bar in the strip, or null)}
//   phonemeLabels(words) -> Map(first note id -> {text, source, typed, ids})   the front-end's words (dict tcs_cond, with
//             the note ids tcs_notes.js gives them) as labels: the phonemes go above a word's notes (UWP5)
//
// Layers, back to front: rows and grid, the area past the limit (20 s), rest labels, notes, phonemes, curve, the region's
// span (F0 mode), playhead and rubber band, then the keys, the ruler with the range strip at its foot, and the toolbar
// with the status line over everything. The curve's hand-edited runs
// (UWP6: where it differs from the generated one) are drawn in an accent over a faint line of the generated curve; in
// the F0 mode the notes are dimmed. No offscreen cache: UWP0
// measured no gain. Nothing here checks the notes (U10): overlapping notes are simply drawn see-through.

"use strict";

const V = require("./lib/tcs_ui_view.js");

const FONT = "Arial";
const BLACK = [false, true, false, true, false, false, true, false, true, false, true, false];

const C = {
  bg: [0.118, 0.118, 0.118, 1],
  rowBlack: [0.098, 0.098, 0.098, 1],
  rowLine: [1, 1, 1, 0.045],
  octaveLine: [1, 1, 1, 0.11],
  bar: [1, 1, 1, 0.22],
  beat: [1, 1, 1, 0.1],
  sub: [1, 1, 1, 0.045],
  limitShade: [0, 0, 0, 0.32],
  limit: [0.95, 0.35, 0.35, 0.8],
  rest: [1, 1, 1, 0.38],
  restHover: [1, 1, 1, 0.85],
  note: [0.92, 0.78, 0.50, 1],
  melisma: [0.92, 0.78, 0.50, 0.72],
  noteSel: [1.0, 0.86, 0.58, 1],
  noteLine: [0.55, 0.25, 0.28, 1],
  noteText: [0.13, 0.06, 0.07, 1],
  hover: [1, 1, 1, 0.22],
  edge: [1, 1, 1, 0.9],
  join: [0.91, 0.52, 0.55, 0.6],
  curve: [0.92, 0.92, 0.92, 0.95],
  curveStale: [0.92, 0.92, 0.92, 0.35],
  curveEdited: [0.45, 0.85, 1, 1],
  curveEditedStale: [0.45, 0.85, 1, 0.4],
  curveGhost: [0.92, 0.92, 0.92, 0.3],
  playhead: [0.55, 0.75, 1, 0.95],
  rubber: [0.55, 0.75, 1, 0.15],
  rubberLine: [0.55, 0.75, 1, 0.7],
  keyWhite: [0.93, 0.93, 0.93, 1],
  keyBlack: [0.12, 0.12, 0.12, 1],
  keyLine: [0.6, 0.6, 0.6, 1],
  keyText: [0.25, 0.25, 0.25, 1],
  ruler: [0.16, 0.16, 0.16, 1],
  rulerLine: [1, 1, 1, 0.3],
  rulerText: [0.82, 0.82, 0.82, 1],
  rulerDim: [0.6, 0.6, 0.6, 1],
  strip: [0.12, 0.12, 0.12, 1], // the range strip (UWP9)
  stripHint: [0.6, 0.6, 0.6, 0.55],
  region: [0.55, 0.75, 1, 0.75],
  regionEdge: [0.55, 0.75, 1, 1],
  regionShade: [0.55, 0.75, 1, 0.07], // its span over the roll, in the F0 mode
  status: [0.09, 0.09, 0.09, 1],
  statusText: [0.78, 0.78, 0.78, 1],
  button: [1, 1, 1, 0.06],
  buttonOn: [0.97, 0.83, 0.55, 0.85],
  buttonText: [0.82, 0.82, 0.82, 1],
  buttonTextOn: [0.13, 0.06, 0.07, 1],
  buttonOff: [0.82, 0.82, 0.82, 0.3],
  ph: [1, 1, 1, 0.55],
  phDict: [0.97, 0.76, 0.36, 0.9], // from the dictionary, not the training lexicon
  phFallback: [1, 0.4, 0.4, 0.95], // no pronunciation: sung as the language's fallback
  phFallbackLine: [0.8, 0.05, 0.12, 1], // under its notes (a darker red, seen on the notes' pink)
};

const PH_SIZE = 8;

const OVERLAP_ALPHA = 0.75;
const F0_MODE_ALPHA = 0.45; // the notes, while the F0 mode edits the curve

const EPS = 1e-6;
const px = (x) => Math.round(x) + 0.5; // a 1-px line on the pixel grid

function rgba(g, c) {
  g.set_source_rgba(c[0], c[1], c[2], c[3]);
}

function fillRect(g, c, x, y, w, h) {
  rgba(g, c);
  g.rectangle(x, y, w, h);
  g.fill();
}

function line(g, c, x0, y0, x1, y1, width = 1) {
  rgba(g, c);
  g.set_line_width(width);
  g.move_to(x0, y0);
  g.line_to(x1, y1);
  g.stroke();
}

function text(g, c, size, x, y, s) {
  rgba(g, c);
  g.set_font_size(size);
  g.move_to(x, y);
  g.show_text(s);
}

// s, shortened with "…" to fit maxW (at the current font size); "" if nothing fits
function fitText(g, s, maxW) {
  if (g.text_measure(s)[0] <= maxW) return s;
  const cps = Array.from(s);
  for (let k = cps.length - 1; k >= 1; k--) {
    const t = cps.slice(0, k).join("") + "…";
    if (g.text_measure(t)[0] <= maxW) return t;
  }
  return "";
}

// ---- layers

function drawRows(g, v) {
  const x0 = v.keysW;
  const w = v.w - x0;
  fillRect(g, C.bg, x0, v.rulerH, w, v.h - v.rulerH);
  const lo = Math.max(v.midiLo, Math.floor(V.midiBottom(v)));
  const hi = Math.min(v.midiHi, Math.ceil(v.midiTop));
  for (let m = lo; m <= hi; m++) {
    const top = V.midiToY(v, m) - v.rowH / 2;
    if (BLACK[m % 12]) fillRect(g, C.rowBlack, x0, top, w, v.rowH);
    const bottom = px(top + v.rowH);
    line(g, m % 12 === 0 ? C.octaveLine : C.rowLine, x0, bottom, v.w, bottom);
  }
}

function drawGrid(g, v) {
  const step = V.gridStep(v);
  const b1 = V.beatEnd(v);
  for (let k = Math.floor(v.beat0 / step); k * step <= b1; k++) {
    const b = k * step;
    const x = px(V.beatToX(v, b));
    if (x < v.keysW) continue;
    const onBar = Math.abs(b / v.beatsPerBar - Math.round(b / v.beatsPerBar)) < EPS;
    const onBeat = Math.abs(b - Math.round(b)) < EPS;
    line(g, onBar ? C.bar : onBeat ? C.beat : C.sub, x, v.rulerH, x, v.h - v.statusH);
  }
}

function limitX(v, tempo) {
  return v.limitSeconds > 0 ? V.beatToX(v, V.secondsToBeats(v.limitSeconds, tempo)) : Infinity;
}

function drawLimit(g, v, tempo) {
  const x = limitX(v, tempo);
  if (x >= v.w) return;
  const a = Math.max(v.keysW, x);
  fillRect(g, C.limitShade, a, v.rulerH, v.w - a, v.h - v.rulerH - v.statusH);
  if (x >= v.keysW) line(g, C.limit, px(x), 0, px(x), v.h - v.statusH);
}

function drawRests(g, v, gaps, hoverRest, sum) {
  g.set_font_size(9);
  const y = v.rulerH + 11;
  for (const gap of gaps) {
    let s = gap.start;
    for (const r of gap.rests) {
      const x0 = Math.max(v.keysW, V.beatToX(v, s));
      const x1 = Math.min(v.w, V.beatToX(v, s + r.rest));
      const tw = g.text_measure(r.type)[0];
      const shown = x1 - x0 >= tw + 6;
      if (shown) {
        const c = hoverRest && Math.abs(hoverRest.start - s) < EPS ? C.restHover : C.rest;
        line(g, c, x0 + 2, v.rulerH + 2.5, x1 - 2, v.rulerH + 2.5);
        text(g, c, 9, (x0 + x1 - tw) / 2, y, r.type);
      }
      sum.rests.push({ type: r.type, start: s, dur: r.rest, shown });
      s += r.rest;
    }
  }
}

function drawNotes(g, v, scene, sum) {
  const notes = scene.model.notes; // sorted by start (tcs_ui_model keeps them so)
  const shown = new Set(V.visible(v, notes).map((n) => n.id));
  const sel = scene.selection || new Set();
  const overlaps = scene.overlaps || new Set();
  const hover = scene.hover;
  const h = Math.max(1, v.rowH - 1);
  const size = Math.max(7, Math.min(11, v.rowH - 3));
  const r = Math.min(3, h / 3);
  notes.forEach((n, i) => {
    if (!shown.has(n.id)) return;
    const x0 = V.beatToX(v, n.start);
    const x1 = V.beatToX(v, n.start + n.dur);
    const w = Math.max(1, x1 - x0 - 1);
    const y = V.midiToY(v, n.midi) - v.rowH / 2 + 0.5;
    const melisma = n.lyric === "-";
    const prev = i > 0 ? notes[i - 1] : null;
    if (melisma && prev && Math.abs(prev.start + prev.dur - n.start) < EPS && prev.midi !== n.midi) {
      line(g, C.join, px(x0), V.midiToY(v, prev.midi), px(x0), V.midiToY(v, n.midi), 1.5);
    }
    const c = sel.has(n.id) ? C.noteSel : melisma ? C.melisma : C.note;
    const a = (overlaps.has(n.id) ? OVERLAP_ALPHA : 1) * (scene.mode === "f0" ? F0_MODE_ALPHA : 1);
    rgba(g, [c[0], c[1], c[2], c[3] * a]);
    g.rectangle_rounded(x0, y, w, h, r, r);
    g.fill();
    if (sel.has(n.id)) {
      rgba(g, C.edge);
      g.set_line_width(1);
      g.rectangle_rounded(x0 + 0.5, y + 0.5, Math.max(1, w - 1), Math.max(1, h - 1), r, r);
      g.stroke();
    }
    if (hover && hover.id === n.id) {
      if (hover.part === "body") fillRect(g, C.hover, x0, y, w, h);
      else fillRect(g, C.edge, hover.part === "start" ? x0 : x0 + w - 3, y, 3, h);
    }
    sum.notes.push(n.id);
    if (w >= 10 && v.rowH >= 9) {
      g.set_font_size(size);
      const s = fitText(g, n.lyric, w - 6);
      if (s) {
        text(g, C.noteText, size, x0 + 3, y + h / 2 + size * 0.36, s);
        sum.lyrics.push({ id: n.id, text: s });
      }
    }
  });
}

// the front-end's words as labels by their first note's id; a Max dict may turn a one-element list into a value
function phonemeLabels(words) {
  const out = new Map();
  const list = (x) => (x === undefined || x === null ? [] : [].concat(x));
  for (const w of words || []) {
    if (!w || w.source === "rest") continue;
    const ids = list(w.ids).map(Number);
    if (!ids.length) continue;
    out.set(ids[0], { text: list(w.ph).map(String).join(" "), source: w.source, typed: w.typed === undefined ? null : w.typed, ids });
  }
  return out;
}

// above each word, from its first note to the end of its last: grey; amber from the dictionary; red (and a red line
// under the notes) sung as the fallback
function drawPhonemes(g, v, scene, sum) {
  if (!v.showPh || !scene.phonemes || !scene.phonemes.size) return;
  const byId = new Map(scene.model.notes.map((n) => [n.id, n]));
  const top = v.rulerH + PH_SIZE + 1;
  const bottom = v.h - v.statusH;
  g.set_font_size(PH_SIZE);
  for (const [id, lab] of scene.phonemes) {
    const first = byId.get(id);
    if (!first) continue;
    const notes = lab.ids.map((k) => byId.get(k)).filter(Boolean);
    const x0 = V.beatToX(v, first.start);
    const x1 = Math.max(...notes.map((n) => V.beatToX(v, n.start + n.dur)));
    const y = V.midiToY(v, first.midi) - v.rowH / 2 - 2;
    if (x1 < v.keysW || x0 > v.w || y < top || y > bottom) continue;
    const c = lab.source === "fallback" ? C.phFallback : lab.source === "dict" ? C.phDict : C.ph;
    if (lab.source === "fallback") {
      for (const n of notes) {
        const yb = V.midiToY(v, n.midi) + v.rowH / 2 - 1;
        line(g, C.phFallbackLine, Math.max(v.keysW, V.beatToX(v, n.start)), yb, V.beatToX(v, n.start + n.dur) - 1, yb, 2);
      }
    }
    const xs = Math.max(v.keysW + 2, x0 + 1);
    const s = fitText(g, lab.text, Math.min(x1, v.w) - xs - 2);
    if (!s) continue;
    text(g, c, PH_SIZE, xs, y, s);
    sum.phonemes.push({ id, text: s, source: lab.source });
  }
}

// the frames j0..j1 of a curve as runs of voiced points; `keep(j)` picks the frames
function curvePath(g, v, curve, tempo, j0, j1, keep) {
  let pen = false;
  let runs = 0;
  let points = 0;
  for (let j = j0; j <= j1; j++) {
    const m = curve[j];
    if (!(m > 0) || !keep(j)) { pen = false; continue; }
    const x = V.frameToX(v, j, tempo);
    const y = V.midiToY(v, m);
    if (pen) g.line_to(x, y);
    else { g.move_to(x, y); runs++; }
    pen = true;
    points++;
  }
  if (points) g.stroke();
  return { runs, points };
}

// the working curve; where it differs from the generated one (and a frame on either side, so a run joins its
// neighbours), the generated one faintly and the working one in the accent
function drawCurve(g, v, scene, tempo, sum) {
  const curve = scene.curve;
  if (!curve || !curve.length) return;
  const gen = scene.gen && scene.gen.length === curve.length ? scene.gen : null;
  const stale = scene.curveStale;
  const fps = V.CURVE_FPS;
  const j0 = Math.max(0, Math.floor(V.beatsToSeconds(v.beat0, tempo) * fps) - 1);
  const j1 = Math.min(curve.length - 1, Math.ceil(V.beatsToSeconds(V.beatEnd(v), tempo) * fps) + 1);
  const diff = (j) => !!gen && j >= 0 && j < curve.length && curve[j] !== gen[j];
  const near = (j) => diff(j) || diff(j - 1) || diff(j + 1);
  let edited = false;
  for (let j = j0 - 1; j <= j1 + 1 && gen && !edited; j++) edited = diff(j);
  g.set_line_width(1.5);
  if (edited) {
    rgba(g, C.curveGhost);
    curvePath(g, v, gen, tempo, j0, j1, near);
  }
  rgba(g, stale ? C.curveStale : C.curve);
  const all = curvePath(g, v, curve, tempo, j0, j1, () => true);
  sum.curveRuns += all.runs;
  sum.curvePoints += all.points;
  if (edited) {
    rgba(g, stale ? C.curveEditedStale : C.curveEdited);
    g.set_line_width(2);
    sum.editedRuns += curvePath(g, v, curve, tempo, j0, j1, near).runs;
  }
}

// the region's span over the roll, in the F0 mode (under the curve)
function drawRegionShade(g, v, scene, tempo) {
  if (scene.mode !== "f0" || !scene.region) return;
  const [x0, x1] = regionX(v, scene.region, tempo);
  if (x1 > x0) fillRect(g, C.regionShade, x0, v.rulerH, x1 - x0, v.h - v.rulerH - v.statusH);
}

function drawOverlays(g, v, scene, tempo) {
  if (scene.playhead !== null && scene.playhead !== undefined) {
    const x = V.beatToX(v, V.secondsToBeats(scene.playhead, tempo));
    if (x >= v.keysW && x <= v.w) line(g, C.playhead, px(x), 0, px(x), v.h - v.statusH, 1.5);
  }
  const rb = scene.rubber;
  if (rb) {
    const x = Math.min(rb.x0, rb.x1);
    const y = Math.min(rb.y0, rb.y1);
    const w = Math.abs(rb.x1 - rb.x0);
    const h = Math.abs(rb.y1 - rb.y0);
    fillRect(g, C.rubber, x, y, w, h);
    rgba(g, C.rubberLine);
    g.set_line_width(1);
    g.rectangle(px(x), px(y), w, h);
    g.stroke();
  }
}

function drawKeys(g, v) {
  const w = v.keysW;
  const top = v.rulerH;
  fillRect(g, C.keyWhite, 0, top, w, v.h - top - v.statusH);
  const lo = Math.max(v.midiLo, Math.floor(V.midiBottom(v)));
  const hi = Math.min(v.midiHi, Math.ceil(v.midiTop));
  const blackW = Math.round(w * 0.6);
  const size = Math.max(7, Math.min(10, v.rowH - 2));
  for (let m = lo; m <= hi; m++) {
    const y = V.midiToY(v, m);
    if (BLACK[m % 12]) {
      fillRect(g, C.keyBlack, 0, y - v.rowH / 2, blackW, v.rowH);
      line(g, C.keyLine, blackW, px(y), w, px(y)); // the white keys meet behind the black one
    } else if (!BLACK[(m + 1) % 12]) {
      line(g, C.keyLine, 0, px(y - v.rowH / 2), w, px(y - v.rowH / 2)); // E-F and B-C
    }
    if (m % 12 === 0 && v.rowH >= 7) {
      g.set_font_size(size);
      const s = `C${m / 12 - 1}`;
      text(g, C.keyText, size, w - 3 - g.text_measure(s)[0], y + size * 0.36, s);
    }
  }
  line(g, C.keyLine, px(w - 1), top, px(w - 1), v.h - v.statusH);
}

function fmtSeconds(t, step) {
  return `${step < 1 ? t.toFixed(1) : String(Math.round(t))}s`;
}

// the region's x span, clipped to the roll's width
function regionX(v, r, tempo) {
  const x0 = Math.max(v.keysW, V.beatToX(v, V.secondsToBeats(r.s0, tempo)));
  const x1 = Math.min(v.w, V.beatToX(v, V.secondsToBeats(r.s1, tempo)));
  return [x0, x1];
}

// the range strip at the ruler's foot (UWP9): the region as a bar with its ends marked, or a hint to drag
function drawStrip(g, v, tempo, region, sum) {
  const y = V.timeH(v);
  const h = v.rulerH - y;
  fillRect(g, C.strip, v.keysW, y, v.w - v.keysW, h);
  if (!region) {
    text(g, C.stripHint, 8, v.keysW + 4, y + h - 3, "drag here to edit f0 (click to clear)");
    return;
  }
  const [x0, x1] = regionX(v, region, tempo);
  if (x1 <= x0) return;
  fillRect(g, C.region, x0, y + 3, x1 - x0, h - 6);
  for (const s of [region.s0, region.s1]) {
    const x = V.beatToX(v, V.secondsToBeats(s, tempo));
    if (x >= v.keysW && x <= v.w) line(g, C.regionEdge, px(x), y + 1, px(x), v.rulerH - 1, 1.5);
  }
  sum.region = { x0, x1 };
}

function drawRuler(g, v, tempo, playhead, sum, region) {
  const tH = V.timeH(v); // the time part; the range strip is under it
  fillRect(g, C.ruler, 0, 0, v.w, v.rulerH);
  const mid = Math.round(tH / 2);
  const b1 = V.beatEnd(v);
  // bars and beats (top half)
  const bpb = v.beatsPerBar;
  const barStep = V.labelStep(v.pxPerBeat * bpb, 44, [1, 2, 4, 8, 16]);
  const beatsShown = v.pxPerBeat >= 30 && barStep === 1;
  for (let b = Math.max(0, Math.floor(v.beat0)); b <= b1; b++) {
    const x = V.beatToX(v, b);
    if (x < v.keysW) continue;
    const bar = Math.floor(b / bpb);
    const onBar = b % bpb === 0;
    if (onBar && bar % barStep === 0) {
      line(g, C.rulerLine, px(x), 2, px(x), mid);
      text(g, C.rulerText, 10, x + 3, mid - 4, String(bar + 1));
    } else if (beatsShown) {
      line(g, C.rulerLine, px(x), mid - 5, px(x), mid);
      text(g, C.rulerDim, 9, x + 3, mid - 4, `${bar + 1}.${(b % bpb) + 1}`);
    }
  }
  // seconds (bottom half)
  const pxPerSec = (v.pxPerBeat * tempo) / 60;
  const step = V.labelStep(pxPerSec, 44, [0.1, 0.2, 0.5, 1, 2, 5, 10]);
  const s0 = V.beatsToSeconds(v.beat0, tempo);
  const s1 = V.beatsToSeconds(b1, tempo);
  for (let k = Math.max(0, Math.floor(s0 / step)); k * step <= s1; k++) {
    const t = k * step;
    const x = V.beatToX(v, V.secondsToBeats(t, tempo));
    if (x < v.keysW) continue;
    line(g, C.rulerLine, px(x), tH - 5, px(x), tH);
    text(g, C.rulerDim, 9, x + 3, tH - 4, fmtSeconds(t, step));
  }
  drawStrip(g, v, tempo, region, sum);
  const lx = limitX(v, tempo);
  if (lx >= v.keysW && lx < v.w) line(g, C.limit, px(lx), 0, px(lx), v.rulerH, 1.5);
  line(g, C.rulerLine, 0, px(tH - 1), v.w, px(tH - 1));
  line(g, C.rulerLine, 0, px(v.rulerH - 1), v.w, px(v.rulerH - 1));
  // the corner: the tempo
  fillRect(g, C.ruler, 0, 0, v.keysW, v.rulerH);
  text(g, C.rulerText, 9, 4, mid - 4, `${Math.round(tempo * 100) / 100}`);
  text(g, C.rulerDim, 9, 4, tH - 4, "bpm");
  // the playhead's marker (UWP7): a triangle at the ruler's foot, so a stopped playhead is easy to find
  if (playhead !== null && playhead !== undefined) {
    const x = V.beatToX(v, V.secondsToBeats(playhead, tempo));
    if (x >= v.keysW && x <= v.w) {
      rgba(g, C.playhead);
      g.move_to(x - 5, tH - 9); g.line_to(x + 5, tH - 9); g.line_to(x, tH - 1); g.close_path(); g.fill();
      line(g, C.playhead, px(x), 0, px(x), v.rulerH, 1.5);
      sum.playhead = x;
    }
  }
}

// the toolbar's icons, in a w x h box at (x, y)
function icon(g, id, x, y, w, h) {
  const cx = x + w / 2;
  const cy = y + h / 2;
  if (id === "select") { // an arrow pointer
    g.move_to(cx - 3, cy - 6); g.line_to(cx - 3, cy + 5); g.line_to(cx, cy + 2); g.line_to(cx + 2, cy + 6);
    g.line_to(cx + 3.5, cy + 5.2); g.line_to(cx + 1.5, cy + 1.5); g.line_to(cx + 5, cy + 1.5); g.close_path(); g.fill();
  } else if (id === "pencil") { // a pencil, tip down-left
    g.set_line_width(3); g.move_to(cx - 2.5, cy + 2.5); g.line_to(cx + 4.5, cy - 4.5); g.stroke();
    g.move_to(cx - 5, cy + 5); g.line_to(cx - 3.8, cy + 1.2); g.line_to(cx - 1.2, cy + 3.8); g.close_path(); g.fill();
  } else if (id === "draw") { // a free stroke
    g.set_line_width(1.5);
    g.move_to(cx - 6, cy + 3); g.curve_to(cx - 3, cy - 6, cx, cy + 6, cx + 6, cy - 4); g.stroke();
  } else if (id === "line") { // a segment between two points
    g.set_line_width(1.5);
    g.move_to(cx - 5, cy + 4); g.line_to(cx + 5, cy - 4); g.stroke();
    g.rectangle(cx - 6.5, cy + 2.5, 3, 3); g.fill(); g.rectangle(cx + 3.5, cy - 5.5, 3, 3); g.fill();
  } else if (id === "smooth") { // a gentle wave
    g.set_line_width(1.5);
    g.move_to(cx - 6, cy); g.curve_to(cx - 3, cy - 4, cx - 1, cy - 4, cx, cy); g.curve_to(cx + 1, cy + 4, cx + 3, cy + 4, cx + 6, cy); g.stroke();
  } else if (id === "erase") { // an eraser, tilted
    g.set_line_width(1.5);
    g.move_to(cx - 6, cy + 2); g.line_to(cx - 1, cy - 4); g.line_to(cx + 5, cy + 1); g.line_to(cx, cy + 6); g.close_path(); g.stroke();
    g.move_to(cx - 3.5, cy - 1); g.line_to(cx + 2.5, cy + 4); g.stroke();
  } else if (id === "vibrato") { // a tight wave, and a plus: one click adds depth
    g.set_line_width(1.3);
    g.move_to(cx - 8, cy + 1);
    for (let k = 0; k < 3; k++) {
      const x0 = cx - 8 + k * 4;
      g.curve_to(x0 + 1, cy - 3, x0 + 1, cy - 3, x0 + 2, cy + 1);
      g.curve_to(x0 + 3, cy + 5, x0 + 3, cy + 5, x0 + 4, cy + 1);
    }
    g.stroke();
    g.set_line_width(1.5);
    g.move_to(cx + 5.5, cy - 6); g.line_to(cx + 5.5, cy - 1); g.stroke();
    g.move_to(cx + 3, cy - 3.5); g.line_to(cx + 8, cy - 3.5); g.stroke();
  } else if (id === "offsetUp" || id === "offsetDown") { // the offset's arrows: a triangle up or down
    const k = id === "offsetUp" ? -1 : 1;
    g.move_to(cx - 4, cy - 2 * k); g.line_to(cx + 4, cy - 2 * k); g.line_to(cx, cy + 2.5 * k); g.close_path(); g.fill();
  } else if (id === "unvoice") { // a line with a gap
    g.set_line_width(1.5);
    g.move_to(cx - 7, cy); g.line_to(cx - 2.5, cy); g.stroke(); g.move_to(cx + 2.5, cy); g.line_to(cx + 7, cy); g.stroke();
    g.move_to(cx - 1.5, cy - 4); g.line_to(cx + 1.5, cy + 4); g.stroke();
  } else { // undo / redo: a hooked arrow
    const k = id === "undo" ? 1 : -1;
    g.set_line_width(1.5);
    g.move_to(cx - 4 * k, cy - 1); g.line_to(cx + 2 * k, cy - 1);
    g.curve_to(cx + 6 * k, cy - 1, cx + 6 * k, cy + 5, cx + 2 * k, cy + 5); g.line_to(cx - 1 * k, cy + 5); g.stroke();
    g.move_to(cx - 6 * k, cy - 1); g.line_to(cx - 2.5 * k, cy - 4.5); g.line_to(cx - 2.5 * k, cy + 2.5); g.close_path(); g.fill();
  }
}

function drawBar(g, v, scene, sum) {
  if (v.statusH <= 0) return;
  const y = v.h - v.statusH;
  fillRect(g, C.status, 0, y, v.w, v.statusH);
  const f0 = scene.mode === "f0";
  const { buttons, textX } = V.toolbarLayout(v, scene.mode);
  g.set_font_size(9);
  for (const b of buttons) {
    const on = b.kind === "tool" ? (f0 ? scene.f0tool : scene.tool) === b.id : b.kind === "snap" ? `snap:${v.snap}` === b.id :
      b.kind === "toggle" ? !!v.showPh : b.kind === "mode" ? f0 : false;
    const off = (b.id === "undo" && !scene.canUndo) || (b.id === "redo" && !scene.canRedo);
    fillRect(g, on ? C.buttonOn : C.button, b.x, b.y, b.w, b.h);
    rgba(g, on ? C.buttonTextOn : off ? C.buttonOff : C.buttonText);
    if (b.kind === "snap" || b.kind === "mode" || b.id === "fit" || b.id === "ph") {
      const s = b.kind === "snap" ? b.id.slice(5) : b.kind === "mode" ? "F0" : b.id;
      g.move_to(b.x + (b.w - g.text_measure(s)[0]) / 2, b.y + b.h / 2 + 3.2);
      g.show_text(s);
    } else icon(g, b.id, b.x, b.y, b.w, b.h);
    sum.buttons.push(b.id);
  }
  if (!scene.status) return;
  g.set_font_size(10);
  text(g, C.statusText, 10, textX, y + v.statusH / 2 + 3.6, fitText(g, scene.status, v.w - textX - 6));
}

function draw(g, v, scene) {
  const sum = { notes: [], lyrics: [], rests: [], curveRuns: 0, curvePoints: 0, editedRuns: 0, buttons: [], phonemes: [], playhead: null,
    region: null };
  const m = scene.model;
  const tempo = m ? m.tempo : 60;
  g.select_font_face(FONT);
  drawRows(g, v);
  drawGrid(g, v);
  drawLimit(g, v, tempo);
  if (m) {
    drawRests(g, v, scene.gaps || [], scene.hoverRest, sum);
    drawNotes(g, v, scene, sum);
    drawPhonemes(g, v, scene, sum);
  }
  drawRegionShade(g, v, scene, tempo);
  drawCurve(g, v, scene, tempo, sum);
  drawOverlays(g, v, scene, tempo);
  drawKeys(g, v);
  drawRuler(g, v, tempo, scene.playhead, sum, scene.region || null);
  drawBar(g, v, scene, sum);
  return sum;
}

module.exports = { draw, fitText, phonemeLabels, COLORS: C };
