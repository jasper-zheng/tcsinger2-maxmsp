// tcs_score.js: the score JSON to a Score (port of ldm/infer/score.py parse_score; same checks and messages).
//
//   {"language": "English", "tempo": 112, "style": {...},
//    "words": [{"rest": 1.0}, {"text": "where", "notes": [["C4", 1]]}, ...]}
//
// A Score is {language, words, style, name, f0, f0_fps}; a word is {text, notes: [[midi, seconds]], ph,
// ph_durs, notes_per_syllable, isRest}. "f0" is checked as in Python and kept, though the Max front-end
// does not use it.

"use strict";

const { scoreError, repr, isNumber, pySum } = require("tcs_util.js");
const data = require("tcs_data.js");

const MAX_SECONDS = 20.0;
const F0_FPS = 48000 / 256;
const MAX_MIDI = 100 - 2; // NUM_MIDI - 2
const REST_TYPES = { AP: "<AP>", SP: "<SP>", "<AP>": "<AP>", "<SP>": "<SP>" };
const STEPS = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const WORD_KEYS = ["text", "notes", "ph", "ph_durs", "notes_per_syllable"];
const REST_KEYS = ["rest", "type"];
const TOP_KEYS = ["language", "tempo", "style", "words", "name", "f0", "f0_fps"];
const STYLE_KEYS = ["singer", "emotion", "singing_method", "technique"];

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const keysOutside = (obj, allowed) => Object.keys(obj).filter((k) => !allowed.includes(k)).sort();
// Python's isinstance(v, int): a JSON true/false is an int too
const isPyInt = (v) => typeof v === "boolean" || (isNumber(v) && Number.isInteger(v));
const isPyNum = (v) => typeof v === "boolean" || isNumber(v);
const pyList = (xs) => "[" + xs.map(repr).join(", ") + "]";

function wordSeconds(w) {
  return pySum(w.notes.map((n) => n[1]));
}

function noteToMidi(pitch) {
  if (isNumber(pitch) && Number.isInteger(pitch)) return pitch;
  const m = /^\s*([A-Ga-g])([#b♯♭]?)(-?\d+)\s*$/u.exec(String(pitch));
  if (m && typeof pitch !== "boolean")
    return STEPS[m[1].toUpperCase()] + { "": 0, "#": 1, "♯": 1, b: -1, "♭": -1 }[m[2]] + 12 * (parseInt(m[3], 10) + 1);
  if (typeof pitch === "string" && /^\s*\d+\s*$/.test(pitch)) return parseInt(pitch, 10);
  throw scoreError(`bad pitch ${repr(typeof pitch === "boolean" ? pitch : pitch)}: use a MIDI number or a name like C4, F#3, Bb4`);
}

function length(value, beat, where) {
  if (typeof value === "boolean" || !isNumber(value) || value <= 0)
    throw scoreError(`${where}: bad length ${repr(value)}, expected a number above 0`);
  return value * beat;
}

function parseF0(f0, f0Fps, where = '"f0"') {
  if (f0Fps === undefined || f0Fps === null) f0Fps = F0_FPS;
  else if (typeof f0Fps === "boolean" || !isNumber(f0Fps) || f0Fps <= 0)
    throw scoreError(`bad "f0_fps" ${repr(f0Fps)}: frames per second, a number above 0`);
  if (f0 === undefined || f0 === null) return [null, f0Fps];
  if (!Array.isArray(f0) || !f0.length) throw scoreError(`${where} must be a list of Hz values, one per frame, 0 where unvoiced`);
  f0.forEach((v, i) => {
    if (v !== null && (typeof v === "boolean" || !isNumber(v) || v < 0))
      throw scoreError(`${where}[${i}]: bad value ${repr(v)}, expected Hz as a number, or 0 / null where unvoiced`);
  });
  return [f0.map((v) => v || 0), f0Fps];
}

// Score of the object of a JSON score file; throws a ScoreError with the place of the first problem
function parseScore(d, name = "score", maxSeconds = MAX_SECONDS) {
  if (!isObject(d)) throw scoreError('the score must be a JSON object with "language" and "words"');
  const unknown = keysOutside(d, TOP_KEYS);
  if (unknown.length) throw scoreError(`unknown key(s) ${pyList(unknown)}; allowed: ${pyList([...TOP_KEYS].sort())}`);
  const languages = data.phones().language_set;
  const language = d.language;
  if (!languages.includes(language)) throw scoreError(`bad language ${repr(language)}: expected one of ${pyList(languages)}`);
  const tempo = d.tempo;
  if (tempo !== undefined && tempo !== null && (typeof tempo === "boolean" || !isNumber(tempo) || tempo <= 0))
    throw scoreError(`bad tempo ${repr(tempo)}: beats per minute, a number above 0`);
  const beat = tempo ? 60.0 / tempo : 1.0;
  const style = d.style || {};
  if (!isObject(style) || keysOutside(style, STYLE_KEYS).length)
    throw scoreError('"style" may hold singer, emotion, singing_method and technique');
  const entries = d.words;
  if (!Array.isArray(entries) || !entries.length) throw scoreError('"words" must be a list with at least one entry');

  const words = [];
  entries.forEach((en, i) => {
    let where = `words[${i}]`;
    if (!isObject(en))
      throw scoreError(`${where}: expected an object like {"text": ..., "notes": [[pitch, length]]} or {"rest": length}`);
    if ("rest" in en) {
      const extra = keysOutside(en, REST_KEYS);
      if (extra.length) throw scoreError(`${where}: a rest takes "rest" and "type" only, found ${pyList(extra)}`);
      const kind = "type" in en ? en.type : i === 0 ? "SP" : "AP";
      if (typeof kind !== "string" || !(kind in REST_TYPES))
        throw scoreError(`${where}: bad rest type ${repr(kind)}, expected "AP" (breath) or "SP" (silence)`);
      words.push({ text: REST_TYPES[kind], notes: [[0, length(en.rest, beat, where)]], ph: null, ph_durs: null,
        notes_per_syllable: null, isRest: true });
      return;
    }
    const extra = keysOutside(en, WORD_KEYS);
    if (extra.length) throw scoreError(`${where}: unknown key(s) ${pyList(extra)}; allowed: ${pyList([...WORD_KEYS].sort())}`);
    let text = en.text;
    if (typeof text !== "string" || !text.trim()) {
      const ph0 = en.ph;
      if (!ph0 || (Array.isArray(ph0) && !ph0.length)) throw scoreError(`${where}: a sung word needs "text" (or "ph")`);
      text = Array.isArray(ph0) ? ph0.map(String).join(" ") : Array.from(String(ph0)).join(" ");
    }
    where = `words[${i}] (${repr(text)})`;
    const notes = en.notes;
    if (!Array.isArray(notes) || !notes.length) throw scoreError(`${where}: "notes" must be a list of [pitch, length]`);
    const parsed = [];
    for (const note of notes) {
      if (!Array.isArray(note) || note.length !== 2) throw scoreError(`${where}: bad note ${repr(note)}, expected [pitch, length]`);
      let midi;
      try {
        midi = noteToMidi(note[0]);
      } catch (e) {
        throw scoreError(`${where}: ${e.message}`);
      }
      if (!(midi >= 1 && midi <= MAX_MIDI))
        throw scoreError(`${where}: pitch ${repr(note[0])} is MIDI ${midi}, outside 1..${MAX_MIDI}; a rest is {"rest": length}`);
      parsed.push([midi, length(note[1], beat, where)]);
    }
    const ph = en.ph === undefined ? null : en.ph;
    const phDurs = en.ph_durs === undefined ? null : en.ph_durs;
    const nps = en.notes_per_syllable === undefined ? null : en.notes_per_syllable;
    if (ph !== null && (!Array.isArray(ph) || !ph.length || !ph.every((p) => typeof p === "string")))
      throw scoreError(`${where}: "ph" must be a list of phonemes`);
    if (phDurs !== null) {
      if (!Array.isArray(phDurs) || !phDurs.every((x) => isPyNum(x) && x > 0))
        throw scoreError(`${where}: "ph_durs" must be a list of seconds above 0`);
      if (ph !== null && ph.length !== phDurs.length) throw scoreError(`${where}: ${ph.length} phonemes but ${phDurs.length} lengths`);
    }
    if (nps !== null) {
      if (!Array.isArray(nps) || !nps.every((k) => isPyInt(k) && k >= 0) || !nps.length || nps[0] < 1 ||
          pySum(nps.map(Number)) !== parsed.length)
        throw scoreError(`${where}: "notes_per_syllable" must be whole numbers that start above 0 and add up to the ` +
          `${parsed.length} note(s) of the word`);
    }
    words.push({ text: text.trim(), notes: parsed, ph, ph_durs: phDurs && phDurs.length ? phDurs.map(Number) : null,
      notes_per_syllable: nps ? nps.map(Number) : null, isRest: false });
  });

  const [f0, f0Fps] = parseF0(d.f0, d.f0_fps);
  const score = { language, words, style: { ...style }, name: d.name || name, f0, f0_fps: f0Fps };
  if (words.every((w) => w.isRest)) throw scoreError("the score has no sung word");
  const seconds = scoreSeconds(score);
  if (maxSeconds && seconds > maxSeconds + 1e-6)
    throw scoreError(`the score lasts ${seconds.toFixed(2)} s; the model was trained on clips of at most ${maxSeconds.toFixed(0)} s ` +
      "and longer scores are not supported yet. Split it at a rest and sing the parts one by one.");
  return score;
}

function scoreSeconds(score) {
  return pySum(score.words.map(wordSeconds));
}

module.exports = { parseScore, noteToMidi, wordSeconds, scoreSeconds, MAX_SECONDS, MAX_MIDI, F0_FPS };
