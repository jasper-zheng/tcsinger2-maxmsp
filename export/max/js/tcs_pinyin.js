// tcs_pinyin.js: Chinese lyrics to phonemes (port of ldm/infer/g2p.py _chinese, with pypinyin's segmenter).
//
// A line of single characters is cut into phrases by forward maximum matching (pypinyin.seg.mmseg.Seg.cut
// with no_non_phrases=True); a phrase takes the readings of pypinyin's phrase dictionary, any other
// character its own. Both come as phonemes from tcs_data_pinyin.js, computed by pypinyin itself. A pinyin
// syllable typed as the text is split as pypinyin's to_initials / to_finals (strict) split it, ported below
// for toneless input with pypinyin's own initials and finals.

"use strict";

const { has } = require("tcs_util.js");
const data = require("tcs_data.js");

const isHan = (ch) => (ch >= "一" && ch <= "鿿") || (ch >= "㐀" && ch <= "䶿");

// pypinyin's Seg.cut (no_non_phrases=True) on a string of characters
function cut(text, prefixes, phrases) {
  const out = [];
  let remain = text;
  while (remain.length) {
    let lastValid = "";
    let lastIndex = 0;
    let finished = true;
    for (let index = 0; index < remain.length; index++) {
      const word = remain.slice(0, index + 1);
      if (prefixes.has(word)) {
        if (has(phrases, word)) {
          lastValid = word;
          lastIndex = index + 1;
        }
      } else {
        if (lastValid) {
          out.push(lastValid);
          remain = remain.slice(lastIndex);
        } else {
          out.push(remain[0]);
          remain = remain.slice(1);
        }
        finished = false;
        break;
      }
    }
    if (finished) {
      if (lastValid) {
        out.push(lastValid);
        remain = remain.slice(lastIndex);
      } else {
        if (!has(phrases, remain)) for (const x of remain) out.push(x);
        else out.push(remain);
        break;
      }
    }
  }
  return out;
}

// pypinyin.contrib.tone_convert, for a toneless syllable of [a-z:] with v for ü
const U_TONES = ["u", "ū", "ú", "ǔ", "ù"];
const I_TONES = ["i", "ī", "í", "ǐ", "ì"];
const UV_MAP = { u: "ü", ū: "ǖ", ú: "ǘ", ǔ: "ǚ", ù: "ǜ" };
const UV_RE = /^(j|q|x)(u|ū|ú|ǔ|ù)(.*)$/u;
const IU_RE = /^([a-z]+)(iu|iū|iú|iǔ|iù)$/u;
const UI_RE = /([a-z]+)(ui|uī|uí|uǐ|uì)$/u;
const UN_RE = /([a-z]+)(un|ūn|ún|ǔn|ùn)$/u;
const IU_MAP = { iu: "iou", iū: "ioū", iú: "ioú", iǔ: "ioǔ", iù: "ioù" };
const UI_MAP = { ui: "uei", uī: "ueī", uí: "ueí", uǐ: "ueǐ", uì: "ueì" };
const UN_MAP = { un: "uen", ūn: "ūen", ún: "úen", ǔn: "ǔen", ùn: "ùen" };

function getInitials(py, strict, pd) {
  for (const i of strict ? pd.initials : pd.initialsNotStrict) if (py.startsWith(i)) return i;
  return "";
}

function convertZeroConsonant(py, pd) {
  const raw = py;
  if (raw.startsWith("y")) {
    const noY = py.slice(1);
    const first = noY.length ? Array.from(noY)[0] : null;
    if (U_TONES.includes(first)) py = UV_MAP[first] + Array.from(py).slice(2).join("");
    else if (I_TONES.includes(first)) py = noY;
    else py = "i" + noY;
  }
  if (raw.startsWith("w")) {
    const noW = py.slice(1);
    const first = noW.length ? Array.from(noW)[0] : null;
    py = U_TONES.includes(first) ? py.slice(1) : "u" + py.slice(1);
  }
  return pd.finals.includes(py) ? py : raw;
}

function convertFinals(py, pd) {
  py = convertZeroConsonant(py, pd);
  py = py.replace(UV_RE, (m, a, b, c) => a + UV_MAP[b] + c);
  py = py.replace(IU_RE, (m, a, b) => a + IU_MAP[b]);
  py = py.replace(UI_RE, (m, a, b) => a + UI_MAP[b]);
  py = py.replace(UN_RE, (m, a, b) => a + UN_MAP[b]);
  return py;
}

function getFinals(py, pd) { // strict
  py = convertFinals(py, pd);
  let initials = getInitials(py, true, pd);
  let finals = py.slice(initials.length);
  if (!pd.finals.includes(finals)) {
    initials = getInitials(py, false, pd);
    finals = py.slice(initials.length);
    return pd.finals.includes(finals) ? finals : "";
  }
  return finals;
}

// g2p._zh_syllable: phonemes of an initial and a final, or null if the phone set lacks them
function zhSyllable(initial, final) {
  const fixes = data.phones().ZH_FINAL_FIXES;
  if (has(fixes, final)) final = fixes[final];
  const phs = [initial, final].filter(Boolean).map((p) => p + "_zh");
  return phs.length && phs.every((p) => data.ph2id().has(p)) ? phs : null;
}

function typedSyllable(syl, pd) {
  const initial = getInitials(syl, true, pd);
  const final = getFinals(syl.split("v").join("ü"), pd).split("ü").join("v");
  return zhSyllable(initial, final);
}

// phonemes per word of a Chinese line (texts[i] null = not to convert), null where there are none
function chinese(texts) {
  const pd = data.pinyin();
  const out = texts.map(() => null);
  const han = [];
  texts.forEach((t, i) => {
    if (t !== null && t.length === 1 && isHan(t)) han.push(i);
  });
  if (han.length) {
    const line = han.map((i) => texts[i]).join("");
    const perChar = [];
    for (const seg of cut(line, pd.prefixes, pd.phrases)) {
      if (has(pd.phrases, seg)) perChar.push(...pd.phrases[seg]);
      else for (const ch of seg) perChar.push(has(pd.chars, ch) ? pd.chars[ch] : null);
    }
    han.forEach((i, k) => {
      out[i] = perChar[k] ? perChar[k].slice() : null;
    });
  }
  texts.forEach((t, i) => {
    if (t === null || out[i] !== null) return;
    const low = t.toLowerCase();
    if (/^[a-zü:]+\d?$/u.test(low)) { // a pinyin syllable
      const syl = low.replace(/\d$/, "").split("u:").join("v").split("ü").join("v");
      out[i] = typedSyllable(syl, pd);
    }
  });
  return out;
}

module.exports = { chinese, cut, isHan };
