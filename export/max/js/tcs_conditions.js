// tcs_conditions.js: pieces to the conditions of the TCSinger2 Max models (port of scripts/score_to_max.py).
//
//   ph, midi, note_dur, note_type, ph_dur, nucleus, note_id   [256] each, padded as in PAD
//   language                                                  index in language_set
//
// ph_dur is the length the score gives (rests, ph_durs), else -1: the model predicts it. note_id is the
// fit group of a piece: its word's note, each rest its own.

"use strict";

const { valueError } = require("tcs_util.js");
const data = require("tcs_data.js");

const N_PIECES = 256;
const PREDICT = -1.0;
const FORMAT = "tcsinger2-max-conditions/1";

function pads() {
  const ph = data.phones();
  return { ph: ph.PH_PAD_ID, midi: ph.MIDI_PAD_ID, note_dur: 0.0, note_type: ph.NOTE_TYPE_PAD, ph_dur: 0.0, nucleus: 0, note_id: -1 };
}

// index of every piece's note, from 0 (a word's note, and each rest its own)
function noteIds(pieces) {
  const keys = new Map();
  return pieces.map((p, i) => {
    const key = p.midi === 0 ? "rest/" + i : p.word + "/" + p.note;
    if (!keys.has(key)) keys.set(key, keys.size);
    return keys.get(key);
  });
}

function conditions(pieces, language, n = N_PIECES) {
  if (pieces.length > n)
    throw valueError(`the score has ${pieces.length} phoneme pieces, more than the ${n} of the Max models; ` +
      "split it at a rest and convert the parts one by one");
  const ids = data.ph2id();
  const values = {
    ph: pieces.map((p) => ids.get(p.ph)),
    midi: pieces.map((p) => p.midi),
    note_dur: pieces.map((p) => p.notedur),
    note_type: pieces.map((p) => p.type),
    ph_dur: pieces.map((p) => (p.given ? p.dur : PREDICT)),
    nucleus: pieces.map((p) => (p.nucleus ? 1 : 0)),
    note_id: noteIds(pieces),
  };
  const pad = pads();
  const out = {};
  for (const name of Object.keys(values)) out[name] = values[name].concat(new Array(n - pieces.length).fill(pad[name]));
  out.language = data.phones().language_set.indexOf(language);
  return out;
}

module.exports = { conditions, noteIds, N_PIECES, PREDICT, FORMAT };
