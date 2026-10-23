// tcs_prompt.js: the TCSinger2 caption in Max: [v8 tcs_prompt.js]. Style messages in, the caption (the prompt of
// neural.tokenizer) out. The piano roll (tcs_ui.js) makes the conditions dict; this object makes only the caption, so a
// style change needs no new conversion. A score's own "style" is not used here: the caption is what these messages set.
//
// Inlet:
//   singer female alto        style fields, as scripts/sing.py's flags: singer (female|male [soprano|alto|tenor|bass]),
//   emotion happy             emotion (happy|sad), method (pop|bel canto), technique (mix, falsetto, breathy,
//   method bel canto          pharyngeal, joined by "and", at most one "then"; or none). Each is checked, sets its
//   technique breathy         field and sends the caption; a bad value is an error and changes nothing.
//   caption "<whole caption>" a whole caption instead (quoted: a comma splits a Max message), checked
//   style clear               no fields and no whole caption: the empty caption
//   bang                      send the caption again
//   status                    post the fields and the caption to the console
// Outlet 0: the caption, one symbol (its commas would split a Max message), for neural.tokenizer through
//           [prepend prompt]. Sent on load and after every change.
// Outlet 1: ok <caption> | error <text>
//
// (No "use strict" here: the script assigns Max's own globals, inlets and outlets.)

inlets = 1;
outlets = 2;
setinletassist(0, "singer, emotion, method, technique, caption, style clear, bang, status");
setoutletassist(0, "the caption (one symbol), for neural.tokenizer through [prepend prompt]");
setoutletassist(1, "status: ok <caption> / error <text>");

const { makeCaption } = require("./lib/tcs_caption.js");

// all state in one object that is never replaced, only changed
const S = {
  style: {},        // singer, emotion, singing_method, technique
  rawCaption: null, // a whole caption from `caption`, used instead of the fields
};

function fail(e) {
  const msg = e && e.message ? e.message : String(e);
  error(`tcs_prompt: ${msg}\n`);
  outlet(1, "error", msg);
}

const current = (style = S.style, raw = S.rawCaption) => makeCaption(null, raw, style);

function send() {
  let caption;
  try {
    caption = current();
  } catch (e) {
    return fail(e);
  }
  outlet(1, "ok", caption);
  outlet(0, caption);
}

// a change is kept only when the caption it gives is valid
function change(style, raw) {
  try {
    current(style, raw);
  } catch (e) {
    return fail(e);
  }
  S.style = style;
  S.rawCaption = raw;
  send();
}

const words = (args) => Array.prototype.slice.call(args).map(String).join(" ");
function setField(key, value) {
  if (S.rawCaption !== null) post(`tcs_prompt: ${key} is kept, but a whole caption is set; send style clear to use the fields\n`);
  change({ ...S.style, [key]: value }, S.rawCaption);
}
function singer() { setField("singer", words(arguments)); }
function emotion() { setField("emotion", words(arguments)); }
function method() { setField("singing_method", words(arguments)); }
function technique() { setField("technique", words(arguments)); }
function caption() { change(S.style, words(arguments)); }

function bang() {
  send();
}

function loadbang() {
  send();
}

function status() {
  let caption;
  try {
    caption = current();
  } catch (e) {
    caption = `(${e.message})`;
  }
  post(`tcs_prompt: style ${JSON.stringify(S.style)}, whole caption ${JSON.stringify(S.rawCaption)}; caption: ${caption || "(empty)"}\n`);
}

function anything() {
  const args = Array.prototype.slice.call(arguments).map(String);
  if (messagename === "style" && args[0] === "clear") return change({}, null);
  fail(`unknown message ${messagename}`);
}
