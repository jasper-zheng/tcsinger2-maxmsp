// tcs_notes.js: the piano roll's notes to a TCSinger2 score, and a score to notes (UI_WORK_PACKAGES.md, UWP4, U10).
// The roll edits notes and checks nothing; this is the TCSinger2 side. It fixes what the model can't take, with a
// warning per kind of fix, so that inference always goes on. No Max globals: tcs_max.js and tcs_ui.js use it in Max,
// the tests under Node.
//
//   fromScore(scoreObject, name) -> {model, warnings, f0}        a score file into the roll (tcs_ui_model.js's model);
//                                                                 f0: the score's "f0" as {values (Hz), fps}, or null
//   toScore(notes) -> {score, warnings, ids}                     notes (a tcs_notes dict, or the model) to a score; ids:
//                                                                 per score word, the note ids it came from (null: a rest)
//   convertNotes(notes, {name, style, caption}) -> result        toScore, unknown words sung as FALLBACK, cut to 256
//                                                                 pieces, then tcs_frontend's convert; each words[i] gets
//                                                                 its note ids, and a fallback word source "fallback" and
//                                                                 "typed" (what the notes said)
//   payload(model, name) -> the tcs_notes dict the roll sends:   {format, version, name, language, tempo, style,
//                                                                 notes: [{id, start, dur, midi, lyric, word?,
//                                                                 restBefore?}], tail}   (beats; no nulls)
//
// toScore's fixes, in order: a note with no length (or a bad start or pitch) is left out; of two overlapping notes
// the later one wins and the earlier is cut at its start (a note covered entirely is left out; for equal starts the
// higher id wins); pitches are rounded into 1..98; a note with no lyric is left out, so its time becomes a rest; a
// "-" with no note right before it is sung as a word of its own (the text of the sung word before it, else the
// language's FALLBACK); notes after 20 s are left out and a note across the line is cut. convertNotes then sings a word
// with no pronunciation in the language as FALLBACK (UWP5). A score file read by tcs_max.js stays strict.

"use strict";

const { TcsError, isNumber } = require("./lib/tcs_util.js");
const { parseScore, MAX_SECONDS, MAX_MIDI } = require("./lib/tcs_score.js");
const { phonemize } = require("./lib/tcs_g2p.js");
const { buildPieces } = require("./lib/tcs_pieces.js");
const { convert } = require("./lib/tcs_frontend.js");
const { N_PIECES } = require("./lib/tcs_conditions.js");
const M = require("./lib/tcs_ui_model.js");

const FORMAT = "tcs_notes";
const VERSION = 1;
const EPS = M.EPS;
const EXTRAS = ["ph", "ph_durs", "notes_per_syllable"];
const SHOWN = 4; // beats listed in a warning
// the syllable sung for a word with no pronunciation, and for a lone "-" (each one pronounceable in its language)
const FALLBACK = { English: "la", Chinese: "la", French: "la", German: "la", Italian: "la", Spanish: "la", Japanese: "ら",
  Korean: "라", Russian: "да" };
const fallbackFor = (language) => FALLBACK[language] || "la";

const round9 = (x) => Math.round(x * 1e9) / 1e9;
const end = (n) => n.start + n.dur;
const sumDur = (segs) => segs.reduce((s, x) => s + x.dur, 0);

// ---- a score to notes

function fromScore(obj, name = "score") {
  const parsed = parseScore(obj, name); // the front-end's checks and messages; also the MIDI numbers
  const warnings = [];
  const notes = [];
  let pending = [];
  let t = 0;
  let id = 1;
  obj.words.forEach((en, i) => {
    const pw = parsed.words[i];
    if (pw.isRest) {
      pending.push({ type: pw.text === "<SP>" ? "SP" : "AP", dur: en.rest });
      t += en.rest;
      return;
    }
    const hasText = typeof en.text === "string" && en.text.trim() !== "";
    en.notes.forEach((nt, k) => {
      const note = { id: id++, start: t, dur: nt[1], midi: pw.notes[k][0], lyric: k === 0 ? pw.text : "-" };
      if (k === 0) {
        if (pending.length) note.restBefore = pending;
        pending = [];
        if (!hasText || EXTRAS.some((x) => en[x] !== undefined && en[x] !== null)) {
          const word = { text: hasText ? pw.text : null, n: en.notes.length };
          for (const x of EXTRAS) if (en[x] !== undefined && en[x] !== null) word[x] = en[x].slice();
          note.word = word;
        }
      }
      notes.push(note);
      t += nt[1];
    });
  });
  const model = { language: obj.language, tempo: obj.tempo || 60, name: obj.name || null, style: { ...(obj.style || {}) },
    nextId: id, notes, tail: pending };
  return { model, warnings, f0: parsed.f0 ? { values: parsed.f0, fps: parsed.f0_fps } : null };
}

// ---- notes to a score

function beatsList(beats) {
  const shown = beats.slice(0, SHOWN).map((b) => String(round9(b)));
  return `beat${beats.length > 1 ? "s" : ""} ${shown.join(", ")}${beats.length > SHOWN ? ", …" : ""}`;
}

const MESSAGES = {
  bad: (k, b) => `${k} note(s) with no length, or a bad start or pitch, were left out (${b})`,
  cut: (k, b) => `${k} overlapping note(s) were cut where the next note starts (${b}): TCSinger2 sings one note at a time`,
  covered: (k, b) => `${k} note(s) covered by a later note were left out (${b}): TCSinger2 sings one note at a time`,
  pitch: (k, b) => `${k} pitch(es) were moved into 1..${MAX_MIDI} (${b})`,
  empty: (k, b) => `${k} note(s) with no lyric became rests (${b})`,
  melisma: (k, b) => `${k} "-" note(s) with no note right before them were sung as words of their own (${b})`,
  limit: (k, b, s) => `the notes after ${s} s were cut (${b}): the model takes at most ${s} s`,
};

// the notes as TCSinger2 can sing them: {model, warnings} (the model of tcs_ui_model.js, sorted, nothing overlapping)
function fix(notes, limitSeconds = MAX_SECONDS) {
  const found = {};
  const flag = (kind, b) => (found[kind] = found[kind] || []).push(b);
  const tempo = notes.tempo;
  const fallback = fallbackFor(notes.language);
  if (!isNumber(tempo) || tempo <= 0) throw new TcsError("ScoreError", `bad tempo ${tempo}`);
  let all = M.sorted(notes.notes || []);
  // 1 bad notes
  all = all.filter((n) => {
    const ok = isNumber(n.start) && n.start >= 0 && isNumber(n.dur) && n.dur > EPS && isNumber(n.midi);
    if (!ok) flag("bad", isNumber(n.start) ? n.start : 0);
    return ok;
  });
  // 2 overlaps: the later note wins
  const out = [];
  for (const n of all) {
    while (out.length && end(out[out.length - 1]) > n.start + EPS) {
      const last = out[out.length - 1];
      if (last.start >= n.start - EPS) { out.pop(); flag("covered", last.start); }
      else { out[out.length - 1] = { ...last, dur: round9(n.start - last.start) }; flag("cut", last.start); break; }
    }
    out.push(n);
  }
  // 3 pitches, 4 lyrics, 5 lone melismas
  const sung = [];
  let prevEnd = -Infinity;
  let prevText = null;
  for (let n of out) {
    const midi = Math.min(MAX_MIDI, Math.max(1, Math.round(n.midi)));
    if (midi !== n.midi) { n = { ...n, midi }; flag("pitch", n.start); }
    const lyric = typeof n.lyric === "string" ? n.lyric.trim() : "";
    if (!lyric) { flag("empty", n.start); continue; }
    if (lyric === "-" && Math.abs(n.start - prevEnd) > EPS) { n = { ...n, lyric: prevText || fallback }; flag("melisma", n.start); }
    else if (lyric !== n.lyric) n = { ...n, lyric };
    if (n.lyric !== "-") prevText = n.lyric;
    prevEnd = end(n);
    sung.push(n);
  }
  // 6 the limit
  const limit = (limitSeconds * tempo) / 60;
  const kept = [];
  for (const n of sung) {
    if (n.start >= limit - EPS) { flag("limit", n.start); continue; }
    if (end(n) > limit + EPS) { kept.push({ ...n, dur: round9(limit - n.start) }); flag("limit", n.start); }
    else kept.push(n);
  }
  let tail = (notes.tail || []).slice();
  const last = kept.reduce((m, n) => Math.max(m, end(n)), 0);
  if (last + sumDur(tail) > limit + EPS) {
    const room = Math.max(0, limit - last);
    const cut = [];
    let t = 0;
    for (const r of tail) {
      if (t + r.dur <= room + EPS) cut.push(r);
      else if (room - t > EPS) cut.push({ ...r, dur: round9(room - t) });
      t += r.dur;
    }
    if (!found.limit) flag("limit", last);
    tail = cut;
  }
  const warnings = Object.keys(MESSAGES).filter((k) => found[k]).map((k) => MESSAGES[k](found[k].length, beatsList(found[k]), limitSeconds));
  const model = { language: notes.language, tempo, name: notes.name || null, style: { ...(notes.style || {}) }, nextId: 0, notes: kept, tail };
  return { model, warnings };
}

function toScore(notes, limitSeconds = MAX_SECONDS) {
  const { model, warnings } = fix(notes, limitSeconds);
  if (!model.notes.length) throw new TcsError("ScoreError", "no notes to sing" + (warnings.length ? ` (${warnings.join("; ")})` : ""));
  const words = [];
  const ids = []; // per word: the ids of its notes, null for a rest
  const heads = []; // [entry, first note, note count] per sung word
  const gs = M.gaps(model);
  const before = new Map(gs.filter((g) => !g.tail).map((g) => [g.before, g.rests]));
  let cur = null;
  for (const n of model.notes) {
    for (const r of before.get(n.id) || []) { words.push(r); ids.push(null); }
    if (n.lyric === "-") {
      cur.entry.notes.push([n.midi, n.dur]);
      cur.ids.push(n.id);
      cur.count++;
    } else {
      cur = { entry: { text: n.lyric, notes: [[n.midi, n.dur]] }, note: n, count: 1, index: words.length, ids: [n.id] };
      words.push(cur.entry);
      ids.push(cur.ids);
      heads.push(cur);
    }
  }
  for (const g of gs) if (g.tail) for (const r of g.rests) { words.push(r); ids.push(null); }
  for (const h of heads) {
    const w = h.note.word;
    if (!w) continue;
    const text = w.text === undefined ? null : w.text; // a dict leaves the null out
    const shown = text === null ? w.ph.join(" ") : text;
    if (h.entry.text === shown && h.count === w.n) {
      if (text === null) delete h.entry.text;
      for (const x of EXTRAS) if (w[x] !== undefined) h.entry[x] = w[x].slice();
    } else if (EXTRAS.some((x) => w[x] !== undefined)) {
      warnings.push(`words[${h.index}] (${JSON.stringify(h.entry.text)}): its ${EXTRAS.filter((x) => w[x] !== undefined).join(", ")} ` +
        "were dropped (the word changed)");
    }
  }
  const score = { language: model.language, tempo: model.tempo };
  if (Object.keys(model.style).length) score.style = { ...model.style };
  if (model.name) score.name = model.name;
  score.words = words;
  return { score, warnings, ids };
}

// ---- words with no pronunciation (UWP5)

// can the word be pronounced in the language? Only a G2PError says no: a bad language still throws. (By its kind, not
// instanceof: in Max every require gives a new copy of tcs_util.js, so tcs_g2p.js throws another copy's class.)
function pronounceable(text, language, cache) {
  if (cache.has(text)) return cache.get(text);
  let ok = true;
  try {
    phonemize(parseScore({ language, words: [{ text, notes: [[60, 1]] }] }, "word"));
  } catch (e) {
    if (!e || e.kind !== "G2PError") throw e;
    ok = false;
  }
  cache.set(text, ok);
  return ok;
}

// the score with every sung word that has no pronunciation (and no "ph") sung as the language's FALLBACK, with one
// warning; {score, typed: Map(word index -> the text it had)}
function fixUnknown(score, warnings) {
  const cache = new Map();
  const typed = new Map();
  const found = [];
  let beat = 0;
  const words = score.words.map((w, i) => {
    const at = beat;
    beat += w.rest !== undefined ? w.rest : w.notes.reduce((a, nt) => a + nt[1], 0);
    if (w.rest !== undefined || w.ph !== undefined || typeof w.text !== "string") return w;
    if (pronounceable(w.text, score.language, cache)) return w;
    typed.set(i, w.text);
    found.push(`${JSON.stringify(w.text)} beat ${round9(at)}`);
    return { ...w, text: fallbackFor(score.language) };
  });
  if (!found.length) return { score, typed };
  warnings.push(`${found.length} word(s) with no pronunciation in ${score.language} were sung as ` +
    `${JSON.stringify(fallbackFor(score.language))} (${found.slice(0, SHOWN).join(", ")}${found.length > SHOWN ? ", …" : ""})`);
  return { score: { ...score, words }, typed };
}

// the pieces of the first k entries of a score
function piecesOf(score, k, name) {
  const s = parseScore({ ...score, words: score.words.slice(0, k) }, name);
  return buildPieces(s, phonemize(s).phs).length;
}

// the score cut after the last word that fits in the model's pieces; a warning if it was cut
function fitPieces(score, name, warnings) {
  const n = score.words.length;
  const all = piecesOf(score, n, name);
  if (all <= N_PIECES) return score;
  const firstSung = score.words.findIndex((w) => w.rest === undefined);
  let lo = firstSung + 1; // fits (or nothing does)
  let hi = n; // doesn't fit
  if (piecesOf(score, lo, name) > N_PIECES) throw new TcsError("ValueError", `the first word alone needs more than ${N_PIECES} phoneme pieces`);
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (piecesOf(score, mid, name) <= N_PIECES) lo = mid;
    else hi = mid;
  }
  const words = score.words.slice(0, lo);
  const beat = words.reduce((s, w) => s + (w.rest !== undefined ? w.rest : w.notes.reduce((a, nt) => a + nt[1], 0)), 0);
  const lastSung = words.filter((w) => w.rest === undefined).pop();
  warnings.push(`the phrase was cut after ${JSON.stringify(lastSung.text)} (beat ${round9(beat)}): it needs ${all} phoneme pieces, ` +
    `and the model takes ${N_PIECES}`);
  return { ...score, words };
}

function convertNotes(notes, opts = {}) {
  const made = toScore(notes);
  const warnings = made.warnings;
  const name = opts.name || made.score.name || "notes";
  const { score, typed } = fixUnknown(made.score, warnings);
  const fitted = fitPieces(score, name, warnings);
  const out = convert(fitted, { name, style: opts.style, caption: opts.caption });
  out.meta.warnings = warnings.concat(out.meta.warnings);
  out.words.forEach((w, i) => {
    if (made.ids[i]) w.ids = made.ids[i].slice();
    if (typed.has(i)) { w.source = "fallback"; w.typed = typed.get(i); }
  });
  return out;
}

// ---- the dict the roll sends

const isNotes = (obj) => !!obj && typeof obj === "object" && obj.format === FORMAT;

// null values left out (a Max dict may not keep them)
function dropNulls(v) {
  if (Array.isArray(v)) return v.map(dropNulls);
  if (v && typeof v === "object") {
    const o = {};
    for (const k of Object.keys(v)) if (v[k] !== null && v[k] !== undefined) o[k] = dropNulls(v[k]);
    return o;
  }
  return v;
}

function payload(model, name) {
  const { nextId, ...rest } = model;
  return dropNulls({ format: FORMAT, version: VERSION, ...rest, name: model.name || name || null });
}

module.exports = {
  fromScore, toScore, fix, fixUnknown, convertNotes, payload, isNotes, FORMAT, VERSION, FALLBACK, MAX_MIDI, MAX_SECONDS, N_PIECES,
};
