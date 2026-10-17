// tcs_g2p.js: lyrics to the phonemes of the training phone set (port of ldm/infer/g2p.py).
//
// The phonemes of a sung word come from, in this order: the word's own "ph" (given); Chinese pinyin
// (tcs_pinyin.js); the training lexicon; the MFA dictionary (its most probable variant inside the phone
// set). A word with apostrophes is also tried part by part, and in French, Italian and Spanish a word
// typed without its accents is read as the dictionary word with them. Messages and warnings are those of
// the Python front-end.

"use strict";

const { g2pError, own, has, reprStr, nfc, stripChars } = require("tcs_util.js");
const data = require("tcs_data.js");
const { chinese } = require("tcs_pinyin.js");

const REST = ["<AP>", "<SP>"];
const STRIP = " \t.,;:!?¡¿\"()[]…«»";

function normWord(word) {
  return stripChars(nfc(word).split("’").join("'").toLowerCase(), STRIP);
}

function clean(p, language) {
  const ph = data.phones();
  p = nfc(p);
  if (p === "g" && ph.IPA_LANGS.includes(language)) p = "ɡ";
  const fixes = own(ph.PH_FIXES, language) || {};
  return has(fixes, p) ? fixes[p] : p;
}

const allKnown = (phs) => phs.every((p) => data.ph2id().has(p));

// [phonemes, source, note] of a normalised word from lexicon and dictionary, or null
function lookup(key, language, lex) {
  const suffix = data.phones().LANG_SUFFIX[language];
  const dict = data.mfaDict(language);
  const direct = (k) => {
    const l = own(lex, k);
    if (l !== undefined && allKnown(l)) return [l.slice(), "lexicon"];
    const v = own(dict.words, k);
    return v !== undefined ? [v.split(" ").map((p) => p + suffix), "dict"] : null;
  };
  const hit = direct(key);
  if (hit) return [hit[0], hit[1], null];
  if (stripChars(key, "'").includes("'")) {
    const parts = key.split(/(?<=')/u).filter((part) => stripChars(part, "'"))
      .map((part) => direct(part) || direct(stripChars(part, "'")));
    if (parts.length && parts.every(Boolean)) return [[].concat(...parts.map((x) => x[0])), "dict", null];
  }
  if (data.phones().ACCENT_LANGS.includes(language) && has(dict.unaccented, key)) {
    const word = dict.unaccented[key];
    const d = direct(word);
    return [d[0], d[1], `read as ${reprStr(word)}`];
  }
  return null;
}

// phonemes as the user writes them ('IH1' or 'IH1_en') -> phone-set entries; throws a G2PError
function withSuffix(ph, language) {
  const ids = data.ph2id();
  const suffix = data.phones().LANG_SUFFIX[language];
  return ph.map((p0) => {
    const p = nfc(p0.trim());
    const cand = ids.has(p) ? p : clean(p, language) + suffix;
    if (!ids.has(cand) || REST.includes(cand)) {
      const known = data.phones().ph_set.filter((k) => k.endsWith(suffix)).map((k) => k.slice(0, -3)).sort();
      throw g2pError(`unknown ${language} phoneme ${reprStr(p)}. The phonemes of this language: ${known.join(" ")}`);
    }
    return cand;
  });
}

// phonemes of every word of a Score: {phs, sources, warnings}; a rest gives its own symbol and 'rest'
function phonemize(score) {
  const lang = score.language;
  const n = score.words.length;
  const phs = new Array(n).fill(null), src = new Array(n).fill(null), warnings = [];
  score.words.forEach((w, i) => {
    if (w.isRest) {
      phs[i] = [w.text];
      src[i] = "rest";
    } else if (w.ph !== null) {
      try {
        phs[i] = withSuffix(w.ph, lang);
      } catch (e) {
        throw g2pError(`words[${i}] (${reprStr(w.text)}): ${e.message}`);
      }
      src[i] = "given";
    }
  });
  if (lang === "Chinese") {
    chinese(score.words.map((w, i) => (phs[i] === null ? w.text : null))).forEach((p, i) => {
      if (p !== null) {
        phs[i] = p;
        src[i] = "pinyin";
      }
    });
  }
  const lex = data.lexicon(lang);
  const suffixLen = data.phones().LANG_SUFFIX[lang].length;
  score.words.forEach((w, i) => {
    if (phs[i] !== null) return;
    const hit = lookup(normWord(w.text), lang, lex);
    if (hit) {
      [phs[i], src[i]] = hit;
      if (hit[2]) warnings.push(`words[${i}] (${reprStr(w.text)}): ${hit[2]}`);
      if (src[i] === "dict" && data.phones().WEAK_DICT.includes(lang))
        warnings.push(`words[${i}] (${reprStr(w.text)}): pronunciation ${phs[i].map((p) => p.slice(0, -suffixLen)).join(" ")} ` +
          `taken from the ${lang} dictionary, which often differs from the training labels; check it or give "ph"`);
      return;
    }
    const hint = lang === "Chinese" ? "one character or one pinyin syllable per word" : "it is not in the training lexicon or the dictionary";
    throw g2pError(`words[${i}]: no pronunciation for ${reprStr(w.text)} (${lang}: ${hint}). Give its phonemes in the score, ` +
      `e.g. {"text": ${JSON.stringify(w.text)}, "ph": [...], "notes": ...}; the phonemes of the language are listed by the ` +
      `message: phonemes ${lang}`);
  });
  return { phs, sources: src, warnings };
}

// the phonemes of a language as the user writes them (without the suffix)
function languagePhonemes(language) {
  return own(data.phones().phonemes, language) || null;
}

module.exports = { phonemize, normWord, lookup, withSuffix, languagePhonemes };
