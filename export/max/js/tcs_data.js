// tcs_data.js: loads the generated data modules (scripts/build_max_js_data.py) on first use and keeps them.
//
// The data modules are required as "./data/tcs_data_<name>.js": Max does not look into subfolders for a
// bare name, and Max's require returns undefined (posting "could not find file") instead of throwing
// when a file is missing. A bare name is tried second, for data placed elsewhere in Max's search path.
// The phone set of every file must be the one of the phones file, else the data is from another build.

"use strict";

const FORMAT = "tcsinger2-js-data/1";
// One store for every copy of this module: if Max's require gives each requiring module its own instance,
// the dictionaries are still loaded once, and the load callback set by tcs_max.js still applies.
const store = globalThis.__tcs_data || (globalThis.__tcs_data = { cache: {}, onLoad: null, ph2id: null, pinyin: null });
const cache = store.cache;

function load(name) {
  if (cache[name]) return cache[name];
  const t = Date.now();
  const file = "tcs_data_" + name + ".js";
  let data, why = "not found";
  for (const path of ["./data/" + file, file]) {
    try {
      data = require(path);
    } catch (e) {
      why = e.message;
      data = undefined;
    }
    if (data) break;
  }
  if (!data)
    throw new Error(`cannot load ${file} (${why}); build it with scripts/build_max_js_data.py, which writes it into ` +
      "export/max/js/data/");
  if (data.format !== FORMAT) throw new Error(`tcs_data_${name}.js has format ${data.format}, expected ${FORMAT}`);
  if (name !== "phones" && data.phone_set_sha1 !== phones().phone_set_sha1)
    throw new Error(`tcs_data_${name}.js was built from another phone set than tcs_data_phones.js; rebuild the data`);
  cache[name] = data;
  if (store.onLoad) store.onLoad("tcs_data_" + name + ".js", Date.now() - t);
  return data;
}

function phones() {
  return load("phones");
}
function ph2id() {
  if (!store.ph2id) {
    store.ph2id = new Map();
    phones().ph_set.forEach((p, i) => store.ph2id.set(p, i));
  }
  return store.ph2id;
}

// the MFA dictionary of a language ({words, unaccented}); Chinese has none
function mfaDict(language) {
  if (language === "Chinese") return { words: {} };
  return load("dict_" + language.toLowerCase());
}

function lexicon(language) {
  return load("lexicon").lexicon[language] || {};
}

function pinyin() {
  if (!store.pinyin) {
    const d = load("pinyin");
    const prefixes = new Set(); // every prefix of every phrase, as pypinyin's PrefixSet
    for (const p of Object.keys(d.phrases)) for (let i = 1; i <= p.length; i++) prefixes.add(p.slice(0, i));
    store.pinyin = { chars: d.chars, phrases: d.phrases, initials: d.initials, initialsNotStrict: d.initials_not_strict,
      finals: d.finals, prefixes, version: d.pypinyin };
  }
  return store.pinyin;
}

module.exports = {
  phones, ph2id, mfaDict, lexicon, pinyin,
  setOnLoad: (f) => { store.onLoad = f; },
};
