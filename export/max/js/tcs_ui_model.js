// tcs_ui_model.js: the notes of the piano roll (UI_WORK_PACKAGES.md, UWP1, UWP4): the edits, the gaps between the
// notes and undo. No Max globals and no state: every function returns a new model and never changes the one it gets,
// so tcs_ui.js keeps the only copy (Max's `require` gives a new copy of a module on every call).
//
//   {language, tempo, name, style, nextId,
//    notes: [{id, start, dur, midi, lyric, word?, restBefore?}],   start and dur in beats; lyric "-" = a melisma
//    tail: [{type, dur}]}                                         the rests after the last note
//
// The roll checks nothing (U10): notes may overlap, and a lyric may be anything. tcs_notes.js turns the notes into a
// TCSinger2 score (and reads one), fixing what the model can't take. A word of k notes is k notes, the first with its
// text and the others "-". Rests are the gaps between notes: a gap keeps its rests (restBefore on the note after it)
// while their lengths still add up to it, and otherwise becomes one rest (the first one's type, else SP for the first
// gap and AP for the others). "word" (a word's ph, ph_durs, notes_per_syllable, on its first note) is carried through.

"use strict";

const { TcsError, isNumber } = require("tcs_util.js");

const EPS = 1e-6; // beats: closer than this is the same time
const UNDO_STEPS = 200;
const REST_TYPES = ["AP", "SP"];

const modelError = (m) => new TcsError("ModelError", m);
const round9 = (x) => Math.round(x * 1e9) / 1e9;
const end = (n) => n.start + n.dur;
const byTime = (a, b) => a.start - b.start || a.id - b.id;
const sorted = (notes) => notes.slice().sort(byTime);
const sumDur = (segs) => segs.reduce((s, x) => s + x.dur, 0);

// a roll with no notes (a phrase written from nothing)
function emptyModel(language = "English", tempo = 120, name = "untitled") {
  return { language, tempo, name, style: {}, nextId: 1, notes: [], tail: [] };
}

// ---- the gaps

// the rests of a gap: the kept ones while they still fill it, else one
function restsFor(gap, segs, first) {
  if (segs && segs.length && Math.abs(sumDur(segs) - gap) <= EPS) return segs.map((s) => ({ rest: s.dur, type: s.type }));
  return [{ rest: round9(gap), type: segs && segs.length ? segs[0].type : first ? "SP" : "AP" }];
}

// the gaps between the notes in time order, each with the rests a score gets for it ({rest, type}): {start, dur, rests,
// before: the id of the note after it}, and the tail last ({..., tail: true}). Overlapping notes leave no gap.
function gaps(model) {
  const out = [];
  let prevEnd = 0;
  let first = true;
  for (const n of sorted(model.notes)) {
    const gap = n.start - prevEnd;
    if (gap > EPS) out.push({ start: prevEnd, dur: gap, rests: restsFor(gap, n.restBefore, first), before: n.id });
    first = false;
    prevEnd = Math.max(prevEnd, end(n));
  }
  if (model.tail.length) {
    out.push({ start: prevEnd, dur: sumDur(model.tail), rests: model.tail.map((s) => ({ rest: s.dur, type: s.type })), tail: true });
  }
  return out;
}

// the type of rest `index` of a gap (gapKey: the id of the note after it, or "tail"); the gap's rests are written out,
// so the type stays with the gap when notes move
function setRestType(model, gapKey, index, type) {
  if (!REST_TYPES.includes(type)) throw modelError(`bad rest type ${type}: expected ${REST_TYPES.join(" or ")}`);
  const g = gaps(model).find((x) => (gapKey === "tail" ? x.tail : x.before === gapKey));
  if (!g || !(index >= 0 && index < g.rests.length)) throw modelError(`no rest ${index} before note ${gapKey}`);
  const rests = g.rests.map((r, k) => ({ type: k === index ? type : r.type, dur: r.rest }));
  if (g.tail) return { ...model, tail: rests };
  return withNotes(model, model.notes.map((n) => (n.id === gapKey ? { ...n, restBefore: rests } : n)));
}

// the gap and rest at beat b: {gapKey, index, type} or null
function restAt(model, b) {
  for (const g of gaps(model)) {
    let s = g.start;
    for (let k = 0; k < g.rests.length; k++) {
      const r = g.rests[k];
      if (b >= s - EPS && b < s + r.rest - EPS) return { gapKey: g.tail ? "tail" : g.before, index: k, type: r.type, start: s, dur: r.rest };
      s += r.rest;
    }
  }
  return null;
}

function seconds(model) {
  const last = model.notes.reduce((m, n) => Math.max(m, end(n)), 0);
  return (last + sumDur(model.tail)) * 60 / model.tempo;
}

function noteSeconds(model, note) {
  const k = 60 / model.tempo;
  return { start: note.start * k, dur: note.dur * k };
}

// the ids of notes that overlap another one (drawn see-through)
function overlapping(model) {
  const out = new Set();
  const all = sorted(model.notes);
  for (let i = 0; i < all.length; i++) {
    for (let j = i + 1; j < all.length && all[j].start < end(all[i]) - EPS; j++) { out.add(all[i].id); out.add(all[j].id); }
  }
  return out;
}

// ---- edits: each returns a new model (and leaves the notes it doesn't touch shared)

function withNotes(model, notes, extra) {
  return { ...model, ...(extra || {}), notes: sorted(notes) };
}

function find(model, id) {
  const n = model.notes.find((x) => x.id === id);
  if (!n) throw modelError(`no note ${id}`);
  return n;
}

function checkLength(start, dur) {
  if (!isNumber(start) || !isNumber(dur) || start < 0 || dur <= EPS) throw modelError(`bad note: start ${start}, length ${dur}`);
}

// {model, id}; a new note gets "la" unless a lyric is given
function addNote(model, { start, dur, midi, lyric = "la" }) {
  checkLength(start, dur);
  const note = { id: model.nextId, start, dur, midi, lyric: String(lyric).trim() };
  return { model: withNotes(model, model.notes.concat([note]), { nextId: model.nextId + 1 }), id: note.id };
}

function deleteNotes(model, ids) {
  return withNotes(model, model.notes.filter((n) => !ids.includes(n.id)));
}

// dBeats is limited so that no note starts before 0, and dMidi (given a range [lo, hi]) so that every note stays in it
function moveNotes(model, ids, dBeats, dMidi = 0, range = null) {
  const moving = model.notes.filter((n) => ids.includes(n.id));
  if (!moving.length) return model;
  const d = Math.max(dBeats, -Math.min(...moving.map((n) => n.start)));
  let dm = dMidi;
  if (range) dm = Math.min(Math.max(dm, range[0] - Math.min(...moving.map((n) => n.midi))), range[1] - Math.max(...moving.map((n) => n.midi)));
  return withNotes(model, model.notes.map((n) => (ids.includes(n.id) ? { ...n, start: round9(n.start + d), midi: n.midi + dm } : n)));
}

function resizeNote(model, id, start, stop) {
  return resizeNotes(model, [{ id, start, stop }]);
}

// several notes at once: [{id, start, stop}] (a boundary moved between touching notes)
function resizeNotes(model, list) {
  const by = new Map();
  for (const r of list) {
    find(model, r.id);
    checkLength(r.start, r.stop - r.start);
    by.set(r.id, r);
  }
  return withNotes(model, model.notes.map((n) => (by.has(n.id) ? { ...n, start: round9(by.get(n.id).start), dur: round9(by.get(n.id).stop - by.get(n.id).start) } : n)));
}

// the second part gets "-", so the word keeps both notes
function splitNote(model, id, at) {
  const n = find(model, id);
  if (!(at > n.start + EPS && at < end(n) - EPS)) throw modelError(`cannot split the note at beat ${n.start} at beat ${at}`);
  const second = { id: model.nextId, start: at, dur: round9(end(n) - at), midi: n.midi, lyric: "-" };
  const notes = model.notes.map((x) => (x.id === id ? { ...x, dur: round9(at - n.start) } : x)).concat([second]);
  return withNotes(model, notes, { nextId: model.nextId + 1 });
}

// adjacent notes become the first of them, as long as all of them together
function joinNotes(model, ids) {
  const all = sorted(model.notes);
  const picked = all.filter((n) => ids.includes(n.id));
  if (picked.length < 2) throw modelError("join needs two notes or more");
  const i0 = all.indexOf(picked[0]);
  for (let k = 1; k < picked.length; k++) {
    if (all[i0 + k] !== picked[k] || Math.abs(picked[k].start - end(picked[k - 1])) > EPS)
      throw modelError("only adjacent notes, one right after the other, can be joined");
  }
  const first = { ...picked[0], dur: round9(end(picked[picked.length - 1]) - picked[0].start) };
  return withNotes(model, model.notes.filter((n) => !ids.includes(n.id) || n.id === first.id).map((n) => (n.id === first.id ? first : n)));
}

function setLyric(model, id, text) {
  find(model, id);
  return withNotes(model, model.notes.map((n) => (n.id === id ? { ...n, lyric: String(text).trim() } : n)));
}

// the notes keep their beats, so their seconds change (U1)
function setTempo(model, bpm) {
  if (!isNumber(bpm) || bpm <= 0) throw modelError(`bad tempo ${bpm}`);
  return { ...model, tempo: bpm };
}

// any name: the front-end says whether it knows it
function setLanguage(model, language) {
  const l = String(language).trim();
  if (!l) throw modelError("no language given");
  return { ...model, language: l };
}

// ---- the clipboard: notes relative to the first one's start ({dt, dur, midi, lyric, word?})

function copyNotes(model, ids) {
  const picked = sorted(model.notes).filter((n) => ids.includes(n.id));
  if (!picked.length) return [];
  const t0 = picked[0].start;
  return picked.map((n) => {
    const c = { dt: round9(n.start - t0), dur: n.dur, midi: n.midi, lyric: n.lyric };
    if (n.word) c.word = n.word;
    return c;
  });
}

// {model, ids}: the clip's notes from beat `at` on, with new ids
function pasteNotes(model, clip, at) {
  if (!clip.length) return { model, ids: [] };
  let id = model.nextId;
  const added = clip.map((c) => {
    const n = { id: id++, start: round9(Math.max(0, at + c.dt)), dur: c.dur, midi: c.midi, lyric: c.lyric };
    if (c.word) n.word = c.word;
    return n;
  });
  return { model: withNotes(model, model.notes.concat(added), { nextId: id }), ids: added.map((n) => n.id) };
}

// ---- the lyric line of the [textedit] (U5, UWP5)
//
// With a ";" in it, the line is split at the ";"s only ("so;tell;-;me", "one two;three"); spaces around an entry are
// trimmed, an empty entry leaves its note's lyric, and one ";" at the end is ignored. Without one, it is split at the
// spaces, and Chinese, Japanese and Korean characters are one entry each ("一闪一闪", "我love你" -> 我, love, 你), small
// kana too, as in the training lexicon (きょ -> き, ょ). "-" is a melisma either way.

const CJK = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}\u30fc]/u; // u30fc: ー

// a word without spaces into its entries
function splitWord(w) {
  const out = [];
  let run = "";
  for (const ch of Array.from(w)) {
    if (CJK.test(ch)) {
      if (run) { out.push(run); run = ""; }
      out.push(ch);
    } else run += ch;
  }
  if (run) out.push(run);
  return out;
}

function splitLyricLine(line) {
  const s = String(line).normalize("NFC");
  if (s.includes(";")) return s.replace(/;\s*$/, "").split(";").map((x) => x.trim());
  return [].concat(...s.split(/\s+/).filter(Boolean).map(splitWord));
}

// the notes a line fills (U5, as changed in UWP5): the selected notes in time order, or all when none is selected
function lyricTargets(model, ids) {
  const all = sorted(model.notes);
  const picked = ids && ids.length ? all.filter((n) => ids.includes(n.id)) : [];
  return picked.length ? picked : all;
}

// {model, warnings}: an empty entry leaves its note's lyric; entries beyond the notes it fills are dropped with a warning
function setLyrics(model, ids, line) {
  const entries = splitLyricLine(line);
  const targets = lyricTargets(model, ids);
  const warnings = [];
  if (entries.length > targets.length) warnings.push(`${entries.length - targets.length} lyric(s) beyond the notes filled were dropped`);
  const set = new Map();
  targets.forEach((n, i) => { if (i < entries.length && entries[i] !== "") set.set(n.id, entries[i]); });
  if (![...set].some(([id, text]) => find(model, id).lyric !== text)) return { model, warnings };
  return { model: withNotes(model, model.notes.map((n) => (set.has(n.id) ? { ...n, lyric: set.get(n.id) } : n))), warnings };
}

// the lyrics of the notes a line fills, as one line that splits back into them: spaces between them when each one is a
// single entry on its own, else ";" (with one at the end of a single lyric, so its spaces are kept)
function lyricLine(model, ids) {
  const lyrics = lyricTargets(model, ids).map((n) => n.lyric);
  const plain = (l) => l !== "" && !l.includes(";") && !/\s/.test(l) && splitWord(l.normalize("NFC")).length === 1;
  if (lyrics.every(plain)) return lyrics.join(" ");
  return lyrics.join(";") + (lyrics.length === 1 ? ";" : "");
}

// ---- undo: snapshots of the model and the working F0 curve (UWP6: one history for both; they share what did not change)

function history(model, curve = null) {
  return { past: [], present: model, curve, future: [] };
}

function push(h, model, label = "", curve = h.curve) {
  return { past: h.past.concat([{ model: h.present, curve: h.curve, label }]).slice(-UNDO_STEPS), present: model, curve, future: [] };
}

function undo(h) {
  if (!h.past.length) return h;
  const last = h.past[h.past.length - 1];
  return { past: h.past.slice(0, -1), present: last.model, curve: last.curve,
    future: [{ model: h.present, curve: h.curve, label: last.label }].concat(h.future) };
}

function redo(h) {
  if (!h.future.length) return h;
  const next = h.future[0];
  return { past: h.past.concat([{ model: h.present, curve: h.curve, label: next.label }]), present: next.model, curve: next.curve,
    future: h.future.slice(1) };
}

// a new generated curve (U4: it replaces the hand edits): every step gets it, and the steps that only changed the curve
// (the same model on both sides) are dropped, so the note steps stay undoable
function newCurve(h, curve) {
  const states = h.past.map((e) => e.model).concat([h.present], h.future.map((e) => e.model));
  const labels = h.past.map((e) => e.label).concat(h.future.map((e) => e.label)); // labels[i]: the step into state i + 1
  const p = h.past.length;
  const kept = [0];
  const into = [];
  let q = 0; // the present among the kept states
  for (let i = 1; i < states.length; i++) {
    if (states[i] !== states[kept[kept.length - 1]]) { kept.push(i); into.push(labels[i - 1]); }
    if (i === p) q = kept.length - 1;
  }
  const entry = (k, label) => ({ model: states[kept[k]], curve, label });
  return { past: kept.slice(0, q).map((_, k) => entry(k, into[k])), present: states[kept[q]], curve,
    future: kept.slice(q + 1).map((_, k) => entry(q + 1 + k, into[q + k])) };
}

module.exports = {
  emptyModel, gaps, restsFor, setRestType, restAt, seconds, noteSeconds, overlapping, sorted, addNote, deleteNotes,
  moveNotes, resizeNote, resizeNotes, splitNote, joinNotes, setLyric, setTempo, setLanguage, copyNotes, pasteNotes,
  splitLyricLine, lyricTargets, setLyrics, lyricLine, history, push, undo, redo, newCurve, EPS, REST_TYPES,
};
