// tcs_util.js: helpers shared by the TCSinger2 front-end modules (no Max globals; also runs under Node).
//
// The front-end ports ldm/infer/*.py; these helpers give Python's behaviour where JS differs from it:
// repr() of strings in messages, round() with ties to even, np.searchsorted(side='right'), dict lookups
// that ignore the prototype, and the error classes (ScoreError, G2PError, ValueError).

"use strict";

class TcsError extends Error {
  // kind: "ScoreError" | "G2PError" | "ValueError", as the Python exception class
  constructor(kind, message) {
    super(message);
    this.kind = kind;
  }
}
const scoreError = (m) => new TcsError("ScoreError", m);
const g2pError = (m) => new TcsError("G2PError", m);
const valueError = (m) => new TcsError("ValueError", m);

const hasOwn = Object.prototype.hasOwnProperty;
// obj[key] if obj has key as its own property (a word like "constructor" must not hit Object.prototype)
function own(obj, key) {
  return obj != null && hasOwn.call(obj, key) ? obj[key] : undefined;
}
function has(obj, key) {
  return obj != null && hasOwn.call(obj, key);
}

// Python repr() of a str: quotes, escapes and non-printable characters as Python writes them.
const NON_PRINTABLE = /[\p{C}\p{Z}]/u;
function reprStr(s) {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (ch === "\\" || ch === q) out += "\\" + ch;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch !== " " && NON_PRINTABLE.test(ch)) {
      if (cp < 0x100) out += "\\x" + cp.toString(16).padStart(2, "0");
      else if (cp < 0x10000) out += "\\u" + cp.toString(16).padStart(4, "0");
      else out += "\\U" + cp.toString(16).padStart(8, "0");
    } else out += ch;
  }
  return out + q;
}

// repr() of a JSON value, close to Python's (a JSON number has lost its int/float kind)
function repr(v) {
  if (v === null || v === undefined) return "None";
  if (typeof v === "string") return reprStr(v);
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return String(v);
  if (Array.isArray(v)) return "[" + v.map(repr).join(", ") + "]";
  if (typeof v === "object") return "{" + Object.keys(v).map((k) => reprStr(k) + ": " + repr(v[k])).join(", ") + "}";
  return String(v);
}

// Python round(x, n): the decimal nearest to the double, ties to even.
function pyRound(x, n) {
  if (!Number.isFinite(x)) return x;
  const neg = x < 0;
  const s = Math.abs(x).toFixed(40); // enough digits to see whether the rest is exactly a half
  const dot = s.indexOf(".");
  const keep = s.slice(0, dot + 1 + n).replace(".", "");
  const rest = s.slice(dot + 1 + n);
  let digits = BigInt(keep);
  const half = rest[0] > "5" || (rest[0] === "5" && /[1-9]/.test(rest.slice(1)));
  const tie = rest[0] === "5" && !/[1-9]/.test(rest.slice(1));
  if (half || (tie && digits % 2n === 1n)) digits += 1n;
  const v = Number(digits) / Math.pow(10, n);
  return neg ? -v : v;
}

// np.searchsorted(a, x, side='right'): the number of elements of the sorted a that are <= x
function searchsortedRight(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// np.cumsum as float64, added in order
function cumsum(xs) {
  const out = [];
  let s = 0;
  for (const x of xs) out.push((s += x));
  return out;
}

// Python sum() of floats (3.11: added left to right, starting from the int 0)
function pySum(xs) {
  let s = 0;
  for (const x of xs) s += x;
  return s;
}

// str.strip(chars) with chars as a set of characters (code points)
function stripChars(s, chars) {
  const set = new Set(chars);
  const cps = Array.from(s);
  let a = 0, b = cps.length;
  while (a < b && set.has(cps[a])) a++;
  while (b > a && set.has(cps[b - 1])) b--;
  return cps.slice(a, b).join("");
}

// Unicode NFC, if the engine has it (Max's v8 has ICU; Step 0 of the plan checks it)
const HAS_NORMALIZE = typeof "".normalize === "function" && "é".normalize("NFC") === "é";
function nfc(s) {
  return HAS_NORMALIZE ? s.normalize("NFC") : s;
}

const isNumber = (v) => typeof v === "number" && !Number.isNaN(v);

// a string to UTF-8 bytes (e.g. for Max's File.writebytes)
function utf8Encode(s) {
  const out = [];
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (c < 0x80) out.push(c);
    else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
    else if (c < 0x10000) out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    else out.push(0xf0 | (c >> 18), 0x80 | ((c >> 12) & 63), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
  }
  return out;
}

// UTF-8 bytes (e.g. from Max's File.readbytes) to a string
function utf8Decode(bytes) {
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

module.exports = {
  TcsError, scoreError, g2pError, valueError, own, has, reprStr, repr, pyRound, searchsortedRight, cumsum, pySum,
  stripChars, nfc, HAS_NORMALIZE, isNumber, utf8Encode, utf8Decode,
};
