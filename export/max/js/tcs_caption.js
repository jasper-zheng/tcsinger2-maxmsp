// tcs_caption.js: the caption text (port of ldm/data/captions.py build_caption / parse_caption and of
// ldm/infer/pipeline.py make_caption), e.g. "Singer: female alto, Emotion: happy, Singing Method: pop".

"use strict";

const { valueError, repr } = require("tcs_util.js");

const TIMBRE_TECHS = ["mix", "falsetto", "breathy", "pharyngeal"];
const GENDERS = ["female", "male"];
const VOCAL_RANGES = ["soprano", "alto", "tenor", "bass"];
const EMOTIONS = ["happy", "sad"];
const SINGING_METHODS = ["pop", "bel canto"];
const FIELDS = { singer: "Singer", emotion: "Emotion", singing_method: "Singing Method", technique: "Technique" };
const pyList = (xs) => "[" + xs.map(repr).join(", ") + "]";

// 'mix then falsetto and breathy' -> [['mix'], ['falsetto', 'breathy']]; throws a ValueError if malformed
function parseTechnique(text) {
  const parts = [];
  for (const part of text.split(" then ")) {
    const techs = part === "none" ? [] : part.split(" and ");
    const ordered = TIMBRE_TECHS.filter((t) => techs.includes(t));
    if (techs.length !== ordered.length || techs.some((t, i) => t !== ordered[i]))
      throw valueError(`bad technique ${repr(text)}: use none or ${pyList(TIMBRE_TECHS)} in this order, ` +
        "joined by ' and ', with at most one ' then '");
    parts.push(techs);
  }
  if (parts.length > 2 || (parts.length === 2 && parts[0].join() === parts[1].join()))
    throw valueError(`bad technique ${repr(text)}: at most two different parts around ' then '`);
  return parts;
}

function check(name, value, allowed) {
  if (!allowed.includes(value)) throw valueError(`bad ${name} ${repr(value)}: expected one of ${pyList(allowed)}`);
}

// caption from field values ({singer, emotion, singing_method, technique}); null/undefined leave a field out
function buildCaption(f) {
  const v = (k) => (f[k] === undefined ? null : f[k]);
  if (v("singer") !== null) {
    const s = String(v("singer"));
    const at = s.indexOf(" ");
    check("singer gender", at < 0 ? s : s.slice(0, at), GENDERS);
    if (at >= 0) check("singer vocal range", s.slice(at + 1), VOCAL_RANGES);
  }
  if (v("emotion") !== null) check("emotion", v("emotion"), EMOTIONS);
  if (v("singing_method") !== null) check("singing method", v("singing_method"), SINGING_METHODS);
  if (v("technique") !== null) parseTechnique(String(v("technique")));
  return Object.keys(FIELDS).filter((k) => v(k) !== null).map((k) => `${FIELDS[k]}: ${v(k)}`).join(", ");
}

// inverse of buildCaption: {field: value} of the fields present
function parseCaption(caption) {
  const labels = {};
  for (const [k, label] of Object.entries(FIELDS)) labels[label] = k;
  const out = {};
  for (const part of caption ? caption.split(", ") : []) {
    const at = part.indexOf(": ");
    const label = at < 0 ? part : part.slice(0, at);
    if (at < 0 || !(label in labels) || labels[label] in out) throw valueError(`bad caption ${repr(caption)}`);
    out[labels[label]] = part.slice(at + 2);
  }
  return out;
}

// the caption: a raw caption (checked), or the fields of the score's style overridden by `fields`
function makeCaption(style, caption, fields) {
  if (caption !== null && caption !== undefined) return buildCaption(parseCaption(caption));
  const values = { ...(style || {}) };
  for (const [k, val] of Object.entries(fields || {})) if (val !== null && val !== undefined) values[k] = val;
  return buildCaption(values);
}

module.exports = { buildCaption, parseCaption, parseTechnique, makeCaption, TIMBRE_TECHS, GENDERS, VOCAL_RANGES, EMOTIONS,
  SINGING_METHODS };
