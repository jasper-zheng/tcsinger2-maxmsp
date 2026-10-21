// tcs_max.js: the TCSinger2 front-end in Max: [v8 tcs_max.js]. A score in, the conditions dict and the caption out.
//
// Inlet:
//   read <file.json>          a score file (the format of ldm/infer/score.py), converted at once
//   dictionary <name>         a score held in a Max dict (e.g. from [dict] import), converted at once; or the
//                             notes of the piano roll (a dict with "format": "tcs_notes", from tcs_ui.js): they are
//                             made singable first (tcs_notes.js: overlaps cut, pitches clamped, the phrase cut at
//                             20 s and 256 pieces, ...), with a warning per kind of fix, and always converted
//   bang                      convert the last score again (after a style change)
//   singer female alto        style overrides, as scripts/sing.py's flags: singer, emotion,
//   emotion happy             method (pop / bel canto), technique (e.g. mix then falsetto);
//   method bel canto          `caption "<whole caption>"` instead (quoted: a comma splits a Max message);
//   technique breathy         `style clear` drops them all. They apply from the next conversion on.
//   status                    post the style overrides, caption and output name to the console
//   out <name>                the name of the output dict (default tcs_cond)
//   write <file.json>         write the last result to a file (the caption included, as scripts/score_to_max.py)
//   phonemes <Language>       list the phonemes "ph" may use in a score of that language
// Outlet 0: dictionary <name>: the conditions dict for both neural.gen~ objects (without the caption).
// Outlet 1: the caption, one symbol (its commas would split a Max message), for neural.tokenizer (e.g. through
//           [prepend prompt]).
// Outlet 2: status: ok <pieces> <seconds> | warning <text> | error <text> | phonemes <Language> <ph...>
//           After an error nothing goes out of outlets 0 and 1, so the last good dict and caption stay.
// The outlets send right to left: the status, then the caption, then the dict.
//
// The work is done by tcs_frontend.js (the same modules as the Node parity test). The dictionaries are
// loaded per language on first use (the Russian one, 22 MB, takes a moment) and kept. (No "use strict"
// here: the script assigns Max's own globals, inlets and outlets.)

inlets = 1;
outlets = 3;
setinletassist(0, "read <file>, dictionary <name>, bang, style: singer/emotion/method/technique/caption, out, write, phonemes");
setoutletassist(0, "dictionary: conditions for neural.gen~");
setoutletassist(1, "caption (one symbol) for neural.tokenizer");
setoutletassist(2, "status: ok / warning / error / phonemes");

const data = require("tcs_data.js");
const { convert } = require("tcs_frontend.js");
const { convertNotes, isNotes } = require("tcs_notes.js");
const { languagePhonemes } = require("tcs_g2p.js");
const { utf8Decode } = require("tcs_util.js");

data.setOnLoad((name, ms) => post(`tcs: loaded ${name} in ${ms} ms\n`));

// All state in one object that is never replaced, only changed: the message handlers and run() then
// always see the same state.
const S = {
  outName: "tcs_cond",
  style: {},        // singer, emotion, singing_method, technique overrides
  rawCaption: null, // a whole caption from `caption`, used instead of the style fields
  last: null,       // {score, name} or {notes, name} of the last input
  result: null,
};

function fail(e) {
  const msg = e && e.message ? e.message : String(e);
  error(`tcs: ${msg}\n`);
  outlet(2, "error", msg);
}

function run() {
  if (!S.last) return fail("no score yet: send read <file.json> or dictionary <name>");
  let out;
  try {
    const t = Date.now();
    const opts = { name: S.last.name, style: { ...S.style }, caption: S.rawCaption };
    out = S.last.notes ? convertNotes(S.last.notes, opts) : convert(S.last.score, opts);
    post(`tcs: ${S.last.name}: ${out.meta.n_pieces} pieces, ${out.meta.seconds} s, in ${Date.now() - t} ms; ` +
      `caption: ${out.caption || "(none)"}\n`);
  } catch (e) {
    return fail(e);
  }
  S.result = out;
  const { caption, ...cond } = out;
  const d = new Dict(S.outName);
  d.parse(JSON.stringify(cond));
  for (const w of out.meta.warnings) outlet(2, "warning", w);
  outlet(2, "ok", out.meta.n_pieces, out.meta.seconds);
  outlet(1, caption);
  outlet(0, "dictionary", S.outName);
}

function readText(path) {
  const f = new File(path, "read");
  if (!f.isopen) throw new Error(`cannot open ${path} (give a full path, or a file in Max's search path)`);
  const bytes = [];
  try {
    for (;;) {
      const chunk = f.readbytes(16384);
      if (!chunk || !chunk.length) break;
      for (const b of chunk) bytes.push(b);
    }
  } finally {
    f.close();
  }
  return utf8Decode(bytes);
}

function read(path) {
  if (path === undefined) return fail("read needs a file name");
  const p = String(path);
  try {
    let text = readText(p);
    if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    const base = p.split("/").pop().replace(/\.json$/i, "");
    S.last = { score: JSON.parse(text), name: base };
  } catch (e) {
    return fail(e instanceof SyntaxError ? `${p} is not valid JSON: ${e.message}` : e);
  }
  run();
}

function dictionary(name) {
  try {
    const d = new Dict(String(name));
    const obj = JSON.parse(d.stringify());
    S.last = isNotes(obj) ? { notes: obj, name: obj.name || String(name) } : { score: obj, name: obj.name || String(name) };
  } catch (e) {
    return fail(e);
  }
  run();
}

function bang() {
  run();
}

const words = (args) => Array.prototype.slice.call(args).map(String).join(" ");
function setStyle(key, value) {
  S.style[key] = value;
  post(`tcs: ${key} = ${value}` + (S.rawCaption !== null ? " (not used while a whole caption is set; send style clear)" : "") +
    "; bang to convert again\n");
}
function singer() { setStyle("singer", words(arguments)); }
function emotion() { setStyle("emotion", words(arguments)); }
function method() { setStyle("singing_method", words(arguments)); }
function technique() { setStyle("technique", words(arguments)); }
function caption() {
  S.rawCaption = words(arguments);
  post(`tcs: caption = ${S.rawCaption} (used instead of the style fields until style clear); bang to convert again\n`);
}
function out(name) { S.outName = String(name); }
function status() {
  post(`tcs: style ${JSON.stringify(S.style)}, caption ${JSON.stringify(S.rawCaption)}, out ${S.outName}, ` +
    `score ${S.last ? S.last.name : "(none)"}\n`);
}

function write(path) {
  if (!S.result) return fail("nothing to write yet");
  try {
    const d = new Dict();
    d.parse(JSON.stringify(S.result));
    d.export_json(String(path));
    post(`tcs: wrote ${path}\n`);
  } catch (e) {
    fail(e);
  }
}

function phonemes(language) {
  const list = languagePhonemes(String(language));
  if (!list) return fail(`unknown language ${language}`);
  outlet(2, ["phonemes", String(language)].concat(list));
}

function anything() {
  const args = Array.prototype.slice.call(arguments).map(String);
  if (messagename === "style" && args[0] === "clear") {
    for (const k of Object.keys(S.style)) delete S.style[k];
    S.rawCaption = null;
    post("tcs: style cleared\n");
    return;
  }
  fail(`unknown message ${messagename}`);
}
