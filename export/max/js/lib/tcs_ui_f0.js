// tcs_ui_f0.js: the F0 curve operations of the piano roll (UI_WORK_PACKAGES.md, UWP6). No Max globals and no state:
// every function returns a new curve and never changes the one it gets, so tcs_ui.js keeps the only copy.
//
// A curve is MIDI pitch at 50 Hz, 0 = unvoiced (buffer~ tcs_f0, EWP D7). Every value an operation writes is rounded
// to float32 (Math.fround), so the working curve is exactly what the buffer holds after a poke, and an edit made here
// gives the same buffer as the same edit computed in doubles and poked (EWP7's check does that).
//
//   frames(n, ranges)            the frames j of [[s0, s1), ...] in seconds: j / 50 >= s0 && j / 50 < s1 (EWP7's test)
//   spans(model, ids)            the selected notes' time spans in seconds, merged where they touch; [] with no ids
//   draw(curve, points, lo, hi, round)   a polyline of [frame, midi] points, linear between them, clamped to lo..hi,
//                                each frame rounded to a semitone with `round`; it voices the frames it draws
//   smooth(curve, frames)        a 5-frame moving average over the voiced neighbours in the same voiced run
//   unvoice, reset(.., gen, ..)  0, or the generated values back
//   offset(curve, frames, st)    + st semitones on the voiced frames
//   vibrato(curve, ranges, rate, depth)   + depth * sin(2 pi rate (t - s0)) on the voiced frames, s0 each range's start
//   snap(curve, notes, tempo, frames)     the voiced frames to the pitch of the note sounding there (the later note wins)
//   edited(gen, work)            {frames, runs: [[j0, j1), ...]} where they differ
//   fromHz(f0, fps, n, stretch)  a score's "f0" (Hz at fps, 0 = unvoiced) as a curve of n frames, read as
//                                ldm/infer/f0_input.curve_at reads it (log-linear between voiced frames, unvoiced where the
//                                nearest frame is, and outside the span) at score time j / 50 * stretch: {curve, cut}, cut =
//                                the voiced frames past the curve's end
//   toHz(curve)                  the curve as a score's "f0" at 50 Hz (f0_fps 50): Hz to 1e-3, the unvoiced end left out

"use strict";

const FPS = 50;
const FRAMES = 1003; // the curve of score2f0 and buffer~ tcs_f0 (EWP D7): 20.06 s
const SMOOTH_HALF = 2; // the moving average: 2 frames on either side

const f32 = Math.fround;
const voiced = (m) => m > 0;

function frames(n, ranges) {
  const out = [];
  for (let j = 0; j < n; j++) {
    const t = j / FPS;
    if (ranges.some(([s0, s1]) => t >= s0 && t < s1)) out.push(j);
  }
  return out;
}

// the frames of every range, and the whole curve for [[0, Infinity)]
const whole = () => [[0, Infinity]];

function spans(model, ids) {
  if (!ids || !ids.length) return [];
  const k = 60 / model.tempo;
  const list = model.notes.filter((n) => ids.includes(n.id)).map((n) => [n.start * k, (n.start + n.dur) * k])
    .sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const s of list) {
    const last = out[out.length - 1];
    if (last && s[0] <= last[1] + 1e-9) last[1] = Math.max(last[1], s[1]);
    else out.push(s.slice());
  }
  return out;
}

const clamp = (m, lo, hi) => Math.min(hi, Math.max(lo, m));

function draw(curve, points, lo = 1, hi = 98, round = false) {
  const out = curve.slice();
  const n = out.length;
  const set = (j, m) => { if (j >= 0 && j < n) out[j] = f32(clamp(round ? Math.round(m) : m, lo, hi)); };
  if (points.length === 1) set(Math.round(points[0][0]), points[0][1]);
  for (let k = 1; k < points.length; k++) {
    const [fa, ma] = points[k - 1];
    const [fb, mb] = points[k];
    const ja = Math.round(fa);
    const jb = Math.round(fb);
    if (ja === jb) { set(jb, mb); continue; }
    const step = jb > ja ? 1 : -1;
    for (let j = ja; j !== jb + step; j += step) set(j, ma + ((mb - ma) * (j - ja)) / (jb - ja));
  }
  return out;
}

function smooth(curve, js) {
  const out = curve.slice();
  for (const j of js) {
    if (!voiced(curve[j])) continue;
    let sum = curve[j];
    let count = 1;
    for (const dir of [-1, 1]) {
      for (let d = 1; d <= SMOOTH_HALF; d++) {
        const k = j + dir * d;
        if (k < 0 || k >= curve.length || !voiced(curve[k])) break; // the voiced run ends
        sum += curve[k];
        count++;
      }
    }
    out[j] = f32(sum / count);
  }
  return out;
}

function unvoice(curve, js) {
  const out = curve.slice();
  for (const j of js) out[j] = 0;
  return out;
}

function reset(curve, gen, js) {
  const out = curve.slice();
  for (const j of js) out[j] = gen[j];
  return out;
}

function offset(curve, js, st) {
  const out = curve.slice();
  for (const j of js) if (voiced(curve[j])) out[j] = f32(curve[j] + st);
  return out;
}

// written as EWP7's edit() writes it, so the doubles are the same before the rounding
function vibrato(curve, ranges, rate, depth) {
  const out = curve.slice();
  for (const [s0, s1] of ranges) {
    for (const j of frames(curve.length, [[s0, s1]])) {
      const m = curve[j];
      if (!voiced(m)) continue;
      const t = j / FPS;
      out[j] = f32(m + depth * Math.sin(2 * Math.PI * rate * (t - s0)));
    }
  }
  return out;
}

// notes in time order, so where they overlap the later note's pitch is the one left (as tcs_notes.js cuts, U10)
function snap(curve, notes, tempo, js) {
  const out = curve.slice();
  const inside = new Set(js);
  const k = 60 / tempo;
  const sorted = notes.slice().sort((a, b) => a.start - b.start || a.id - b.id);
  for (const n of sorted) {
    for (const j of frames(curve.length, [[n.start * k, (n.start + n.dur) * k]])) {
      if (inside.has(j) && voiced(curve[j])) out[j] = f32(n.midi);
    }
  }
  return out;
}

function edited(gen, work) {
  const runs = [];
  let count = 0;
  if (!gen || !work) return { frames: 0, runs };
  let start = -1;
  for (let j = 0; j <= work.length; j++) {
    const diff = j < work.length && work[j] !== gen[j];
    if (diff) { count++; if (start < 0) start = j; } else if (start >= 0) { runs.push([start, j]); start = -1; }
  }
  return { frames: count, runs };
}

const LN440 = Math.log(440);

function fromHz(f0, fps, n = FRAMES, stretch = 1) {
  const len = f0.length;
  const curve = new Array(n).fill(0);
  const voicedAt = (i) => f0[i] > 0;
  // the voiced frame at or before / at or after each frame
  const prev = new Array(len);
  const next = new Array(len);
  for (let i = 0, p = -1; i < len; i++) { if (voicedAt(i)) p = i; prev[i] = p; }
  for (let i = len - 1, q = -1; i >= 0; i--) { if (voicedAt(i)) q = i; next[i] = q; }
  const logf = (i) => Math.log(f0[i]);
  for (let j = 0; j < n && len; j++) {
    const k = (j / FPS) * stretch * fps; // the position in the score's frames
    if (k < -0.5 || k > len - 1 + 0.5) continue;
    const right = Math.min(Math.max(Math.ceil(k), 1), Math.max(len - 1, 0));
    const nearest = len === 1 ? 0 : k - (right - 1) <= right - k ? right - 1 : right;
    if (!voicedAt(nearest)) continue;
    const a = prev[Math.min(Math.max(Math.floor(k), 0), len - 1)];
    const b = Math.ceil(k) <= len - 1 ? next[Math.max(Math.ceil(k), 0)] : -1;
    let lf;
    if (a < 0) lf = logf(b);
    else if (b < 0 || a === b) lf = logf(a);
    else lf = logf(a) + ((logf(b) - logf(a)) * (k - a)) / (b - a);
    curve[j] = f32(69 + (12 * (lf - LN440)) / Math.LN2);
  }
  let cut = 0;
  for (let i = 0; i < len; i++) if (voicedAt(i) && i / fps / stretch >= n / FPS) cut++;
  return { curve, cut };
}

function toHz(curve) {
  const hz = Array.from(curve, (m) => (voiced(m) ? Math.round(440 * Math.pow(2, (m - 69) / 12) * 1000) / 1000 : 0));
  let end = hz.length;
  while (end > 1 && hz[end - 1] === 0) end--;
  return hz.slice(0, Math.max(end, 1));
}

module.exports = { frames, whole, spans, draw, smooth, unvoice, reset, offset, vibrato, snap, edited, fromHz, toHz, FPS, FRAMES };
