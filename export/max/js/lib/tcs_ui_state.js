// tcs_ui_state.js: what the piano roll saves with the patch (UI_WORK_PACKAGES.md, U8, UWP3): the model, the view's
// scroll and zoom, and both curves, as one JSON string (embedmessage, and the Dict of getvalueof). No Max globals.
//
//   {format: "tcs_ui_state", version: 1, model, view: {beat0, pxPerBeat, midiTop, rowH, beatsPerBar, snap, showPh,
//    root, scale}, f0: {gen, work, stale}}      (showPh from UWP5; a state without it shows the phonemes; root and
//                                                scale later: without them, the roll's own stay)
//
// The model is kept whole (ids, restBefore, word), not as a score, so an invalid model mid-edit still saves. Curve
// values are written with 9 significant digits, which gives every float32 back exactly: a restored buffer~ equals the
// saved one bit for bit, and decode rounds them back to those float32 values.

"use strict";

const { TcsError, isNumber } = require("./lib/tcs_util.js");

const FORMAT = "tcs_ui_state";
const VERSION = 1;
const VIEW_KEYS = ["beat0", "pxPerBeat", "midiTop", "rowH", "beatsPerBar", "snap", "showPh", "root", "scale"];

const stateError = (m) => new TcsError("StateError", m);

const f32 = (x) => Number(Math.fround(x).toPrecision(9));
const curve = (c) => (c ? Array.from(c, f32) : null);

function pack({ model, view, f0 }) {
  const v = {};
  if (view) for (const k of VIEW_KEYS) v[k] = view[k];
  return { format: FORMAT, version: VERSION, model, view: v,
    f0: { gen: curve(f0 && f0.gen), work: curve(f0 && f0.work), stale: !!(f0 && f0.stale) } };
}

function encode(state) {
  return JSON.stringify(state);
}

function checkCurve(c, name) {
  if (c === null || c === undefined) return null;
  if (!Array.isArray(c) || !c.every(isNumber)) throw stateError(`f0.${name} is not a list of numbers`);
  return c.map(Math.fround); // the buffer's values exactly, so edits after a restore start from them (UWP6)
}

// a model, or null for an empty roll
function checkModel(m) {
  if (m === null) return null;
  if (!m || typeof m !== "object") throw stateError("no model");
  if (typeof m.language !== "string" || !isNumber(m.tempo) || m.tempo <= 0 || !Number.isInteger(m.nextId))
    throw stateError("the model has no language, tempo or nextId");
  if (!Array.isArray(m.notes) || !Array.isArray(m.tail)) throw stateError("the model has no notes or tail");
  for (const n of m.notes) {
    if (!n || !Number.isInteger(n.id) || !isNumber(n.start) || !isNumber(n.dur) || !isNumber(n.midi) || typeof n.lyric !== "string")
      throw stateError(`a malformed note: ${JSON.stringify(n)}`);
  }
  if (!m.style || typeof m.style !== "object") m.style = {};
  if (m.name === undefined) m.name = null;
  return m;
}

// a state from its JSON string (or the parsed object); throws a StateError on anything else
function decode(src) {
  let obj = src;
  if (typeof src === "string") {
    try {
      obj = JSON.parse(src);
    } catch (e) {
      throw stateError(`not JSON: ${e.message}`);
    }
  }
  if (!obj || typeof obj !== "object" || obj.format !== FORMAT) throw stateError(`not a ${FORMAT}`);
  if (obj.version !== VERSION) throw stateError(`version ${obj.version}, expected ${VERSION}`);
  const f0 = obj.f0 || {};
  return { format: FORMAT, version: VERSION, model: checkModel(obj.model), view: obj.view || {},
    f0: { gen: checkCurve(f0.gen, "gen"), work: checkCurve(f0.work, "work"), stale: !!f0.stale } };
}

function fnv(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

// (the curves in their saved form, so a packed state and a decoded one give the same hashes)
function digest(state) {
  const f = { ...state.f0, gen: curve(state.f0.gen), work: curve(state.f0.work) };
  return { notes: state.model ? state.model.notes.length : 0, tempo: state.model ? state.model.tempo : null,
    language: state.model ? state.model.language : null, f0_gen: f.gen ? f.gen.length : 0, f0_work: f.work ? f.work.length : 0,
    stale: f.stale, curve_hash: f.work ? fnv(JSON.stringify(f.work)) : null,
    hash: fnv(JSON.stringify([state.model, f.gen, f.work, f.stale])) }; // the view left out: it follows the box size
}

module.exports = { pack, encode, decode, digest, fnv, FORMAT, VERSION };
