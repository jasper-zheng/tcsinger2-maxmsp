// tcs_frontend.js: a score to the conditions dict of the TCSinger2 Max models, as scripts/score_to_max.py
// writes it, without Python. No Max globals: tcs_max.js calls it in Max, test/parity.js under Node.
//
//   convert(scoreObject, {name, style, caption}) -> {ph, ..., language, caption, meta, words, pieces}
//
// The lengths of predicted pieces are left to the models, so their display timings (pieces[].start/dur)
// are null; word boundaries, rests and "ph_durs" pieces have theirs. Throws a TcsError (kind ScoreError,
// G2PError or ValueError, as in Python) on a bad score.

"use strict";

const { pyRound } = require("tcs_util.js");
const { parseScore, wordSeconds, scoreSeconds } = require("tcs_score.js");
const { phonemize } = require("tcs_g2p.js");
const { buildPieces } = require("tcs_pieces.js");
const { conditions, N_PIECES, FORMAT } = require("tcs_conditions.js");
const { makeCaption } = require("tcs_caption.js");

const FPS = 48000 / 256;
const MAX_FRAMES = 3760;
const TIE = 0.01; // as ldm/infer/timing.py to_frames
const frame = (seconds) => Math.floor(seconds * FPS + 0.5 + TIE);
const bare = (p) => (p.includes("_") ? p.slice(0, p.lastIndexOf("_")) : p);

// the display part (Prepared.describe() of Python, with null where the models predict the timing)
function describe(score, phs, sources, pieces) {
  const words = [];
  const shown = [];
  let t = 0.0; // the running time of the word boundaries, from the notes
  score.words.forEach((w, wi) => {
    const idx = [];
    pieces.forEach((p, i) => { if (p.word === wi) idx.push(i); });
    const start = t;
    t += wordSeconds(w);
    words.push({ text: w.text, ph: phs[wi].map(bare), source: sources[wi], start: pyRound(frame(start) / FPS, 3),
      end: pyRound(frame(t) / FPS, 3), notes: w.notes.map(([m, d]) => [m, pyRound(d, 4)]), pieces: idx });
    let pt = start;
    for (const i of idx) {
      const p = pieces[i];
      if (p.given) {
        const a = frame(pt);
        pt += p.dur;
        shown[i] = { ph: p.ph, start: pyRound(a / FPS, 3), dur: pyRound((frame(pt) - a) / FPS, 3), midi: p.midi, timing: "score" };
      } else shown[i] = { ph: p.ph, start: null, dur: null, midi: p.midi, timing: "predicted" };
    }
  });
  return { words, pieces: shown };
}

function convert(scoreObject, opts = {}) {
  const score = parseScore(scoreObject, opts.name || "score");
  const { phs, sources, warnings } = phonemize(score);
  const pieces = buildPieces(score, phs);
  const nFrames = frame(scoreSeconds(score));
  if (nFrames > MAX_FRAMES) throw new Error(`the score gives ${nFrames} frames, more than the ${MAX_FRAMES} the model was trained on`);
  if (score.f0 !== null) warnings.push('the score has an "f0"; it is ignored: in Max the F0 comes from score2f0');
  const out = conditions(pieces, score.language);
  out.caption = makeCaption(score.style, opts.caption, opts.style);
  out.meta = { format: FORMAT, frontend: "js", name: score.name, language: score.language, n_pieces: pieces.length,
    n_frames: nFrames, seconds: pyRound(nFrames / FPS, 3), pieces_max: N_PIECES, frames_max: MAX_FRAMES, warnings };
  const shown = describe(score, phs, sources, pieces);
  out.words = shown.words;
  out.pieces = shown.pieces;
  return out;
}

module.exports = { convert, FPS, MAX_FRAMES };
