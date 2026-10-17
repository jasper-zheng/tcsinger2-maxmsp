// tcs_pieces.js: phonemes and notes to the pieces of the training metadata (port of ldm/infer/timing.py).
//
// A piece is a phoneme inside one note: {ph, word, note, midi, notedur, type, nucleus, dur, given, index},
// type 1 rest, 2 word on one note, 3 word on several notes. Which phoneme goes on which note:
//  * a word is cut into syllables around its nuclei; one consonant between two nuclei opens the next
//    syllable, of several the first half closes the syllable before;
//  * as many notes as syllables: one each; more notes: the extra ones go to the last syllable, whose
//    nucleus is repeated on each (or as "notes_per_syllable" says); fewer: the syllables are spread
//    evenly over the word's time and each goes on the note it falls in;
//  * a word with "ph_durs" is cut at the note boundaries by time.
// The lengths of the other pieces are predicted inside the Max models, not here.

"use strict";

const { valueError, cumsum, pySum, searchsortedRight, reprStr } = require("tcs_util.js");
const data = require("tcs_data.js");
const { wordSeconds } = require("tcs_score.js");

function isNucleus(ph) {
  const id = data.ph2id().get(ph);
  return id !== undefined && data.phones().nucleus[id];
}

// [[onset indices], nucleus index, [coda indices]] per syllable of a word's phonemes
function syllabify(phs) {
  const nuclei = [];
  phs.forEach((p, i) => { if (isNucleus(p)) nuclei.push(i); });
  const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, k) => a + k);
  if (!nuclei.length) return [[range(0, phs.length - 1), phs.length - 1, []]];
  const syl = nuclei.map((n) => [[], n, []]);
  syl[0][0] = range(0, nuclei[0]);
  syl[syl.length - 1][2] = range(nuclei[nuclei.length - 1] + 1, phs.length);
  for (let j = 0; j + 1 < nuclei.length; j++) {
    const between = range(nuclei[j] + 1, nuclei[j + 1]);
    const k = between.length === 1 ? 0 : Math.floor(between.length / 2);
    syl[j][2] = between.slice(0, k);
    syl[j + 1][0] = between.slice(k);
  }
  return syl;
}

// for each syllable the notes it is sung on (indices)
function syllableNotes(nSyllables, noteLens, notesPerSyllable) {
  const K = noteLens.length, S = nSyllables;
  const range = (a, b) => Array.from({ length: Math.max(0, b - a) }, (_, k) => a + k);
  if (notesPerSyllable) {
    if (notesPerSyllable.length !== S)
      throw valueError(`"notes_per_syllable" has ${notesPerSyllable.length} entries but the word has ${S} syllable(s)`);
    const out = [];
    let k = 0;
    for (const n of notesPerSyllable) {
      out.push(n ? range(k, k + n) : [k - 1]);
      k += n;
    }
    return out;
  }
  if (K >= S) return range(0, S - 1).map((j) => [j]).concat([range(S - 1, K)]);
  const ends = cumsum(noteLens);
  const first = range(0, S).map((j) => Math.min(searchsortedRight(ends, (j + 0.5) / S * ends[ends.length - 1]), K - 1));
  first[0] = 0;
  const out = [];
  for (let j = 0; j < S; j++) {
    const last = j + 1 < S ? (first[j + 1] > first[j] ? first[j + 1] - 1 : first[j]) : K - 1;
    out.push(range(first[j], last + 1));
  }
  return out;
}

const piece = (ph, word, note, midi, notedur, type, nucleus, dur = null, given = false, index = 0) =>
  ({ ph, word, note, midi, notedur, type, nucleus, dur, given, index });

// pieces of a Score whose words have the phonemes phs (tcs_g2p.phonemize)
function buildPieces(score, phs) {
  const pieces = [];
  score.words.forEach((word, wi) => {
    const wph = phs[wi];
    if (word.isRest) {
      const s = wordSeconds(word);
      pieces.push(piece(word.text, wi, 0, 0, s, 1, false, s, true));
      return;
    }
    const noteLens = word.notes.map((n) => n[1]);
    let made = [];
    if (word.ph_durs !== null) {
      if (word.ph_durs.length !== wph.length)
        throw valueError(`words[${wi}] (${reprStr(word.text)}): ${word.ph_durs.length} "ph_durs" for the ${wph.length} phonemes ` +
          wph.map((p) => (p.includes("_") ? p.slice(0, p.lastIndexOf("_")) : p)).join(" "));
      const scale = wordSeconds(word) / pySum(word.ph_durs);
      const bounds = cumsum(noteLens);
      let start = 0.0;
      wph.forEach((p, pi) => {
        const end = start + word.ph_durs[pi] * scale;
        const cuts = [start].concat(bounds.slice(0, -1).filter((b) => start + 1e-6 < b && b < end - 1e-6), [end]);
        for (let c = 0; c + 1 < cuts.length; c++) {
          const a = cuts[c], b = cuts[c + 1];
          const k = Math.min(searchsortedRight(bounds, (a + b) / 2), bounds.length - 1);
          made.push(piece(p, wi, k, word.notes[k][0], word.notes[k][1], 2, isNucleus(p), b - a, true, pi));
        }
        start = end;
      });
    } else {
      const syllables = syllabify(wph);
      let notesOf;
      try {
        notesOf = syllableNotes(syllables.length, noteLens, word.notes_per_syllable);
      } catch (e) {
        throw valueError(`words[${wi}] (${reprStr(word.text)}): ${e.message}`);
      }
      syllables.forEach(([onset, nucleus, coda], s) => {
        const ks = notesOf[s];
        const mk = (i, k, nuc = false) => piece(wph[i], wi, k, word.notes[k][0], word.notes[k][1], 2, nuc, null, false, i);
        made = made.concat(onset.map((i) => mk(i, ks[0])), ks.map((k) => mk(nucleus, k, true)), coda.map((i) => mk(i, ks[ks.length - 1])));
      });
    }
    const noteType = new Set(made.map((p) => p.note)).size > 1 ? 3 : 2;
    for (const p of made) p.type = noteType;
    pieces.push(...made);
  });
  return pieces;
}

module.exports = { buildPieces, syllabify, syllableNotes, isNucleus };
