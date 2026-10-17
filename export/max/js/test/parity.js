// parity.js: the JS front-end against the Python one, under Node.
//
//   node export/max/js/test/parity.js export/artifacts/frontend_corpus.jsonl [--show 5]
//
// The corpus comes from scripts/checks/make_frontend_corpus.py. Per case, these must be identical: the
// eight conditions, caption, meta.{name, language, n_pieces, n_frames, seconds, warnings} and
// words[].{text, ph, source, notes, pieces}; the word boundaries and the timings of given pieces within
// one frame (Python rounds them on a running time that includes its predicted lengths). A score that fails must fail in JS too,
// with the same exception class and the same "words[i]".
//
// Python 3.11 adds floats left to right in sum(), as the JS port does; Python 3.12 and later sum floats
// with compensation, which can move a frame boundary or a note-fit sum in the last bit.

"use strict";

const fs = require("fs");
const path = require("path");
const Module = require("module");

// the modules require each other by bare name, as in Max: resolve them in js/ (the data by ./data/)
const JS = path.resolve(__dirname, "..");
process.env.NODE_PATH = [JS, process.env.NODE_PATH || ""].join(path.delimiter);
Module._initPaths();

const { convert } = require("tcs_frontend.js");

const CONDITIONS = ["ph", "midi", "note_dur", "note_type", "ph_dur", "nucleus", "note_id", "language"];
const META = ["name", "language", "n_pieces", "n_frames", "seconds", "warnings"];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
// display times: within one frame (5.3 ms), since a boundary that lies on the rounding edge (0.49 frame,
// see timing.TIE) can round either way with Python's predicted lengths in the sum
const near = (a, b) => (a === null && b === null) || (a !== null && b !== null && Math.abs(a - b) <= 1 / 187.5 + 0.0015);

function compare(got, want) {
  const diffs = [];
  for (const k of CONDITIONS) {
    if (!same(got[k], want[k])) {
      const g = [].concat(got[k]), w = [].concat(want[k]);
      const i = g.findIndex((v, j) => v !== w[j]);
      diffs.push(`${k}[${i}]: js ${g[i]} vs py ${w[i]}`);
    }
  }
  if (got.caption !== want.caption) diffs.push(`caption: js ${JSON.stringify(got.caption)} vs py ${JSON.stringify(want.caption)}`);
  for (const k of META) if (!same(got.meta[k], want.meta[k])) diffs.push(`meta.${k}: js ${JSON.stringify(got.meta[k])} vs py ${JSON.stringify(want.meta[k])}`);
  if (got.words.length !== want.words.length) diffs.push(`words: ${got.words.length} vs ${want.words.length}`);
  got.words.forEach((w, i) => {
    const v = want.words[i];
    if (!v) return;
    for (const k of ["text", "ph", "source", "notes", "pieces"]) if (!same(w[k], v[k])) diffs.push(`words[${i}].${k}: js ${JSON.stringify(w[k])} vs py ${JSON.stringify(v[k])}`);
    if (!near(w.start, v.start) || !near(w.end, v.end)) diffs.push(`words[${i}] time: js ${w.start}-${w.end} vs py ${v.start}-${v.end}`);
  });
  got.pieces.forEach((p, i) => {
    const q = want.pieces[i];
    if (!q) return;
    if (p.ph !== q.ph || p.midi !== q.midi || p.timing !== q.timing) diffs.push(`pieces[${i}]: js ${JSON.stringify(p)} vs py ${JSON.stringify(q)}`);
    else if (p.timing === "score" && (!near(p.start, q.start) || !near(p.dur, q.dur))) diffs.push(`pieces[${i}] time: js ${p.start}+${p.dur} vs py ${q.start}+${q.dur}`);
  });
  return diffs;
}

function main() {
  const file = process.argv[2];
  if (!file) {
    console.error("usage: node parity.js <frontend_corpus.jsonl> [--show N]");
    process.exit(2);
  }
  const si = process.argv.indexOf("--show");
  const show = si > 0 ? parseInt(process.argv[si + 1], 10) : 5;
  const lines = fs.readFileSync(file, "utf8").split("\n").filter(Boolean);
  let pass = 0, fail = 0;
  const byKind = {};
  const t = Date.now();
  for (const line of lines) {
    const c = JSON.parse(line);
    const kind = c.id.split("/")[0];
    byKind[kind] = byKind[kind] || [0, 0];
    let got, err = null;
    try {
      got = convert(c.score, { name: c.name, style: c.style || undefined, caption: c.caption === null ? undefined : c.caption });
    } catch (e) {
      if (!e.kind) throw e; // a bug, not a score error
      const m = /^words\[(\d+)\]/.exec(e.message);
      err = { kind: e.kind, where: m ? parseInt(m[1], 10) : null, message: e.message };
    }
    let diffs;
    const want = c.expected;
    if (want.error || err) {
      if (!want.error) diffs = [`js fails (${err.kind}: ${err.message}), python does not`];
      else if (!err) diffs = [`python fails (${want.error.kind}: ${want.error.message}), js does not`];
      else diffs = err.kind === want.error.kind && err.where === want.error.where ? [] :
        [`error js ${err.kind} at ${err.where}: ${err.message} | py ${want.error.kind} at ${want.error.where}: ${want.error.message}`];
    } else diffs = compare(got, want);
    if (diffs.length) {
      fail++;
      byKind[kind][1]++;
      if (fail <= show) console.log(`FAIL ${c.id} (${c.name}):\n  ` + diffs.slice(0, 6).join("\n  "));
    } else {
      pass++;
      byKind[kind][0]++;
    }
  }
  for (const [k, [p, f]] of Object.entries(byKind)) console.log(`${k.padEnd(10)} ${p} pass, ${f} fail`);
  console.log(`\n${lines.length} cases: ${pass} pass, ${fail} fail (${((Date.now() - t) / 1000).toFixed(1)} s)`);
  process.exit(fail ? 1 : 0);
}

main();
