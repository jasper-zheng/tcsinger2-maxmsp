// tcs_max.js: the TCSinger2 front-end in Max: [v8 tcs_max.js]. A score in, the conditions dict out.
//
// Inlet:
//   read <file.json>          a score file (the format of ldm/infer/score.py), converted at once
//   dictionary <name>         a score held in a Max dict (e.g. from [dict] import), converted at once
//   bang                      convert the last score again (after a style change)
//   singer female alto        style overrides, as scripts/sing.py's flags: singer, emotion,
//   emotion happy             method (pop / bel canto), technique (e.g. mix then falsetto);
//   method bel canto          `caption "<whole caption>"` instead (quoted: a comma splits a Max message);
//   technique breathy         `style clear` drops them all. They apply from the next conversion on.
//   status                    post the style overrides, caption and output name to the console
//   out <name>                the name of the output dict (default tcs_cond)
//   write <file.json>         write the last result to a file
//   phonemes <Language>       list the phonemes "ph" may use in a score of that language
// Outlet 0: dictionary <name>: the conditions dict for both neural.gen~ objects; its "caption" goes to
//           neural.tokenizer (e.g. through [dict.unpack caption:]).
// Outlet 1: status: ok <pieces> <seconds> | warning <text> | error <text> | phonemes <Language> <ph...>
//           After an error nothing goes out of outlet 0, so the last good dict stays.
//
// The work is done by tcs_frontend.js (the same modules as the Node parity test). The dictionaries are
// loaded per language on first use (the Russian one, 22 MB, takes a moment) and kept. (No "use strict"
// here: the script assigns Max's own globals, inlets and outlets.)

inlets = 1;
outlets = 2;
setinletassist(0, "read <file>, dictionary <name>, bang, style: singer/emotion/method/technique/caption, out, write, phonemes");
setoutletassist(0, "dictionary: conditions for neural.gen~ (caption inside)");
setoutletassist(1, "status: ok / warning / error / phonemes");

const data = require("tcs_data.js");
const { convert } = require("tcs_frontend.js");
const { languagePhonemes } = require("tcs_g2p.js");

data.setOnLoad((name, ms) => post(`tcs: loaded ${name} in ${ms} ms\n`));

// All state in one object that is never replaced, only changed: the message handlers and run() then
// always see the same state.
const S = {
  outName: "tcs_cond",
  style: {},        // singer, emotion, singing_method, technique overrides
  rawCaption: null, // a whole caption from `caption`, used instead of the style fields
  last: null,       // {score, name} of the last input
  result: null,
};

function fail(e) {
  const msg = e && e.message ? e.message : String(e);
  error(`tcs: ${msg}\n`);
  outlet(1, "error", msg);
}

function run() {
  if (!S.last) return fail("no score yet: send read <file.json> or dictionary <name>");
  let out;
  try {
    const t = Date.now();
    out = convert(S.last.score, { name: S.last.name, style: { ...S.style }, caption: S.rawCaption });
    post(`tcs: ${S.last.name}: ${out.meta.n_pieces} pieces, ${out.meta.seconds} s, in ${Date.now() - t} ms; ` +
      `caption: ${out.caption || "(none)"}\n`);
  } catch (e) {
    return fail(e);
  }
  S.result = out;
  const d = new Dict(S.outName);
  d.parse(JSON.stringify(out));
  for (const w of out.meta.warnings) outlet(1, "warning", w);
  outlet(1, "ok", out.meta.n_pieces, out.meta.seconds);
  outlet(0, "dictionary", S.outName);
}

// UTF-8 bytes to a string
function utf8(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length;) {
    const b = bytes[i++] & 0xff;
    let cp;
    if (b < 0x80) cp = b;
    else if (b < 0xe0) cp = ((b & 0x1f) << 6) | (bytes[i++] & 0x3f);
    else if (b < 0xf0) cp = ((b & 0x0f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
    else cp = ((b & 0x07) << 18) | ((bytes[i++] & 0x3f) << 12) | ((bytes[i++] & 0x3f) << 6) | (bytes[i++] & 0x3f);
    s += String.fromCodePoint(cp);
  }
  return s;
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
  return utf8(bytes);
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
    const score = JSON.parse(d.stringify());
    S.last = { score, name: score.name || String(name) };
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
    new Dict(S.outName).export_json(String(path));
    post(`tcs: wrote ${path}\n`);
  } catch (e) {
    fail(e);
  }
}

function phonemes(language) {
  const list = languagePhonemes(String(language));
  if (!list) return fail(`unknown language ${language}`);
  outlet(1, ["phonemes", String(language)].concat(list));
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
