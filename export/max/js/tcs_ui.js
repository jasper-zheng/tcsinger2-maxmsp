// tcs_ui.js: the TCSinger2 piano roll in Max: [v8ui tcs_ui.js] (UI_WORK_PACKAGES.md). It shows and edits notes (with
// their lyrics, and the rests between them) and the F0 curve of buffer~ tcs_f0 over them, sends the notes to
// [v8 tcs_max.js], and saves itself with the patch. It checks nothing (U10): notes may overlap, and tcs_max.js makes
// them singable (tcs_notes.js), with warnings that come back here through `fe`. The lyrics are typed in a [textedit]
// (U5, UWP5), and the front-end's phonemes are drawn above the words. In the F0 mode (UWP6) the mouse edits the curve:
// the generated curve (from score2f0) is kept, and the working one, edited, goes into the buffer after each stroke.
//
// Inlet:
//   read <file.json>, dictionary <name>   load a score (the format of ldm/infer/score.py); the view fits it
//   language <L>, tempo <bpm>             change the notes (a tempo keeps the beats, so the seconds change)
//   bang                                  send the notes again;  clear: no notes
//   write <file.json>                     write the notes as a score (made singable, as tcs_max.js does)
//   fe ok|warning|error ...               the front-end's status (wire tcs_max.js's last outlet back through
//                                         [prepend fe]); on ok, its words and pieces are read from dict tcs_cond, and
//                                         the words' phonemes drawn over their notes (by the note ids of each word)
//   text <line>                           the lyrics line of the [textedit @outputmode 1 @keymode 1] (U5): it fills the
//                                         selected notes, or all when nothing is selected; ";" between the
//                                         notes' lyrics, or spaces when there is no ";" (Chinese, Japanese and Korean
//                                         characters one per note); "-" continues a word
//   phonemes 0|1                          hide or show the phonemes (also the toolbar's ph button)
//   f0 refresh                            read the curve from the buffer (after score2f0's done bang): it is the generated
//                                         curve and the working one; hand edits are dropped (U4)
//   f0 buffer <name>, f0 clear            the buffer to read (default tcs_f0); forget the curve
//   f0 offset <st> [<s0> <s1>]            the curve edits (UWP6), one undo step each; times in seconds. Without a range
//   f0 vibrato <rate> <depth> [<s0> <s1>] (or with s1 <= s0, e.g. 0 0): the selected notes' spans, or the whole curve.
//   f0 unvoice|reset|snap|smooth [<s0> <s1>]   unvoice: 0; reset: the generated curve; snap: the notes' pitches
//   f0 draw <s> <midi> <s> <midi>..., f0 line <s0> <m0> <s1> <m1>
//   mode notes|f0                         what the mouse edits (also the toolbar's F0 button and the F key)
//   tool select|pencil|draw|line|smooth|erase|unvoice   the mouse's tool, and so the mode (also the toolbar and keys)
//   select <ids...> | all | none          the selection
//   add <start> <dur> <midi> [lyric]      a note (beats); delete [ids], move <dBeats> <dMidi>, resize <id> <start> <end>,
//   split <id> <beat>, join [ids]         without ids: the selection
//   rest <beat> [AP|SP|toggle]            the type of the rest at a beat
//   undo, redo, copy, cut, paste [beat], duplicate
//   view fit, view <beat0> <beat1> <midiLo> <midiHi>, zoom <fx> <fy>, scrollby <beats> <rows>
//   snap 1/1 | 1/2 | 1/4 | 1/8 | 1/16 | off, bar <beats>    the grid (U1); a bar is 4 beats by default
//   limit <seconds>, range <lo> <hi>      the shaded limit (20 s) and the pitch rows (1..98): display only
//   play [<s>], stop                      play buffer~ tcs_out from <s> (else from the playhead, or the start) to the
//                                         phrase's end; stop. Also a press on the ruler (play from there) and Space.
//   stopped                               from the patch: the playback reached its end
//   playhead <seconds>, playhead off      the playhead (the patch sends the playback position; drawn when it moves)
//   snapshot <file.png> [scale]           draw the roll into an image file (scale 2 by default)
//   stats [reset], report                 the paint times; a digest of the state (out of outlet 2)
// Outlet 0: dictionary tcs_notes, after every committed edit (for tcs_max.js).
// Outlet 1: bang when the roll has written the curve into the buffer itself (a curve edit, its undo, a restore): send
//   init f0 to sing.
// Outlet 2: ok <notes> <seconds> | warning <text> | error <text> | message <text> | stats <json> | snapshot <file> |
//   report <json> | restored embed|pattr <notes>
// Outlet 3: to the [textedit]: set <line> (the lyrics of the notes the line fills, whenever they change), select (a
//   double-click on a note: its lyric is the start of the line)
// Outlet 4: the transport (UWP7): play <from_ms> <to_ms> <dur_ms> | stop, for the patch's [line~] -> [play~ tcs_out]
// Mouse and keys: see tcs_ui_edit.js (the toolbar at the bottom: the mode, its tools, snap, undo, redo, fit, ph). The wheel scrolls the
// pitch (a trackpad also the time); with Shift it scrolls the time, with Cmd it zooms the time and with Alt the pitch.
//
// Saved with the patch (U8): save() embeds the state (tcs_ui_state.js) in the box, and getvalueof/setvalueof carry it
// for pattr. On restore the curve is written back into the buffer and the notes are sent, without running score2f0.
//
// All state is in S (U9: Max's `require` gives a new copy of a module on every call). No "use strict" here: the
// script assigns Max's own globals.

autowatch = 1;
inlets = 1;
outlets = 5;
setinletassist(0, "read, dictionary, language, tempo, bang, clear, write, fe, text, phonemes, f0, mode, tool, select, add, delete, move, resize, split, join, rest, undo, redo, copy, cut, paste, duplicate, view, zoom, scrollby, snap, bar, limit, range, play, stop, stopped, playhead, snapshot, stats, report");
setoutletassist(0, "dictionary tcs_notes: the notes for tcs_max.js");
setoutletassist(1, "bang: the roll wrote the curve into the buffer (send init f0 to sing)");
setoutletassist(2, "status: ok / warning / error / message / stats / snapshot / report");
setoutletassist(3, "to the lyrics textedit: set <line>, select");
setoutletassist(4, "the transport: play <from_ms> <to_ms> <dur_ms> | stop");

mgraphics.init();
mgraphics.relative_coords = 0;
mgraphics.autofill = 0;

const M = require("tcs_ui_model.js");
const V = require("tcs_ui_view.js");
const D = require("tcs_ui_draw.js");
const E = require("tcs_ui_edit.js");
const F = require("tcs_ui_f0.js");
const N = require("tcs_notes.js");
const ST = require("tcs_ui_state.js");
const { utf8Decode, utf8Encode } = require("tcs_util.js");

const WHEEL_PX = 300; // px per wheel unit (UWP0: steps of 1/256, at most 0.09 per event)
const WHEEL_ZOOM = 3; // zoom factor exp(dy * WHEEL_ZOOM)
const AUTO_MS = 40; // auto-scroll while dragging at the roll's edge
const AUTO_EDGE = 8; // px: the right edge zone (the left, top and bottom use the keys, ruler and toolbar)
const SNAPSHOT_MATRIX = "tcs_ui_snapshot";
const NOTES_DICT = "tcs_notes";
const COND_DICT = "tcs_cond";
const CURVE_SR = 50;
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const S = {
  ed: E.create(M.emptyModel()),
  name: "untitled",
  gaps: [],
  overlaps: new Set(),
  status: "",
  message: null,   // a note from the last edit (e.g. a join that can't be done)
  fe: { status: null, pieces: null, seconds: null, warnings: [], words: null, pieceList: null, shown: null },
  labels: null,    // the phonemes of the front-end's words by first note id (D.phonemeLabels), null until it answers
  line: null,      // the lyrics line last sent to the textedit
  drawn: [],       // the phonemes of the last paint
  f0: { buffer: "tcs_f0", stale: false }, // the curves are the editor's (S.ed.gen, S.ed.curve: UWP6, one undo)
  writes: 0,       // the curve written into the buffer (pokes)
  hoverFrame: null, // the curve's frame under the pointer, in the F0 mode
  view: V.makeView(mgraphics.size[0], mgraphics.size[1]),
  hover: null,
  hoverRest: null,
  playhead: null,
  playing: false,  // the roll asked the patch to play, and it has not stopped yet (UWP7)
  auto: null,      // {x, y, m, dx, dy} while a drag auto-scrolls
  autoTimer: null,
  paints: { ms: [], at: [], since: Date.now() },
  loaded: false,   // loadbang has come: the patch's connections exist
  pending: false,  // a restore whose outputs wait for loadbang
  value: null,     // the last Dict of getvalueof, kept alive until pattr has read it
};

function model() {
  return S.ed.model;
}

function fail(e) {
  const msg = e && e.message ? e.message : String(e);
  error(`tcs_ui: ${msg}\n`);
  outlet(2, "error", msg);
}

function later(fn, ms) {
  if (typeof setTimeout === "function") return setTimeout(fn, ms);
  const t = new Task(fn);
  t.schedule(ms);
  return t;
}

function every(fn, ms) {
  if (typeof setInterval === "function") { const id = setInterval(fn, ms); return () => clearInterval(id); }
  const t = new Task(fn);
  t.interval = ms;
  t.repeat();
  return () => t.cancel();
}

function scene() {
  const g = S.ed.gesture;
  return { model: model(), gaps: S.gaps, overlaps: S.overlaps, curve: S.ed.curve, gen: S.ed.gen, curveStale: S.f0.stale,
    selection: new Set(S.ed.selection), hover: S.hover, hoverRest: S.hoverRest, playhead: S.playhead,
    rubber: g && g.kind === "rubber" ? { x0: g.x0, y0: g.y0, x1: g.x1, y1: g.y1 } : null, mode: S.ed.mode, tool: S.ed.tool, f0tool: S.ed.f0tool,
    canUndo: S.ed.history.past.length > 0, canRedo: S.ed.history.future.length > 0, status: S.status, phonemes: S.labels };
}

function redraw() {
  mgraphics.redraw();
}

// ---- the notes

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
  let text = utf8Decode(bytes);
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  return text;
}

function load(obj, name) {
  let res;
  try {
    res = N.fromScore(obj, name);
  } catch (e) {
    return fail(e);
  }
  S.ed = E.reset(S.ed, res.model);
  S.name = res.model.name || name;
  S.hover = null;
  S.playhead = null;
  S.view = V.fit(S.view, model());
  changed();
  // the status last: whatever it sets off (a check, a recall) then finds the change complete
  for (const w of res.warnings) outlet(2, "warning", w);
  outlet(2, "ok", model().notes.length, Math.round(M.seconds(model()) * 1000) / 1000);
}

// after every committed change of the model: the drawing, the notes go out, pattr is told
function changed(send = true) {
  update();
  if (send) sendNotes();
  notifyclients();
}

// what is drawn from the model (also while a drag previews), and the lyrics line
function update() {
  const m = model();
  S.gaps = M.gaps(m);
  S.overlaps = M.overlapping(m);
  S.view = V.setExtent(S.view, m);
  sendLine();
  statusLine();
}

// the lyrics of the notes the line fills, to the textedit, when they changed (so an answer of the front-end, or a
// regenerated curve, never overwrites a line being typed)
function sendLine() {
  const m = model();
  const line = m.notes.length ? M.lyricLine(m, S.ed.selection) : "";
  if (line === S.line) return;
  S.line = line;
  if (line) outlet(3, "set", line);
  else outlet(3, "set");
}

// the phonemes of the hovered note's word, for the status line
function phonemeText(id) {
  if (!S.labels) return "";
  for (const l of S.labels.values()) {
    if (!l.ids.includes(id)) continue;
    if (l.source === "fallback") return ` · ${JSON.stringify(l.typed)}: no pronunciation, sung as /${l.text}/`;
    return ` · /${l.text}/ (${l.source})`;
  }
  return "";
}

function noteName(midi) {
  const k = Math.round(midi);
  return `${NAMES[((k % 12) + 12) % 12]}${Math.floor(k / 12) - 1}`;
}

function beatName(b) {
  const bpb = S.view.beatsPerBar;
  const bar = Math.floor(b / bpb + 1e-9);
  return `${bar + 1}.${Math.round((b - bar * bpb + 1) * 1000) / 1000}`;
}

function frontEnd() {
  const f = S.fe;
  let s = "";
  if (f.status === "ok") s = ` · front-end: ${f.pieces}/${N.N_PIECES} pieces`;
  else if (f.status === "error") s = ` · front-end error: ${f.shown}`;
  if (f.warnings.length) s += ` · ${f.warnings.length} warning(s): ${f.warnings[0]}`;
  return s;
}

// the curve under the pointer (F0 mode): its time, value and the generated value
function curveText(j) {
  const c = S.ed.curve;
  const t = `${(j / F.FPS).toFixed(2)} s`;
  const pitch = (x) => {
    const cents = Math.round((x - Math.round(x)) * 100);
    return `${x.toFixed(2)} (${noteName(x)} ${cents >= 0 ? "+" : ""}${cents} ct)`;
  };
  const now = c[j] > 0 ? pitch(c[j]) : "unvoiced";
  const gen = S.ed.gen && S.ed.gen[j] !== c[j] ? ` · generated ${S.ed.gen[j] > 0 ? S.ed.gen[j].toFixed(2) : "unvoiced"}` : "";
  return `${t} · ${now}${gen}`;
}

function curveState() {
  if (!S.ed.curve) return S.ed.mode === "f0" ? " · no F0 curve: generate it" : "";
  let s = S.f0.stale ? " · F0 stale: generate again" : "";
  const n = F.edited(S.ed.gen, S.ed.history.curve).frames;
  if (n) s += ` · F0: ${n} frame(s) edited`;
  return s;
}

function statusLine() {
  const m = model();
  if (S.message) S.status = S.message;
  else if (S.ed.mode === "f0" && S.hoverFrame !== null && S.ed.curve) S.status = curveText(S.hoverFrame) + curveState();
  else if (S.hover) {
    const n = m.notes.find((x) => x.id === S.hover.id);
    S.status = n ? `${noteName(n.midi)} · ${beatName(n.start)} · ${Math.round(n.dur * 1000) / 1000} beat(s) · ${JSON.stringify(n.lyric)}` +
      phonemeText(n.id) + frontEnd() : "";
  } else {
    S.status = `${S.name} · ${m.language} · ${Math.round(m.tempo * 100) / 100} bpm · ${M.seconds(m).toFixed(2)} / ${S.view.limitSeconds} s · ` +
      `${m.notes.length} notes · ${S.ed.mode === "f0" ? `F0 ${S.ed.f0tool}` : S.ed.tool}` + frontEnd() + curveState();
    if (S.playing && S.playhead !== null) S.status = `▶ ${S.playhead.toFixed(2)} s · ${S.status}`;
  }
  redraw();
}

function sendNotes() {
  const m = model();
  S.fe.words = null; // no phonemes from an older answer
  S.labels = null;
  if (!m.notes.length) return;
  if (S.ed.curve) S.f0.stale = true;
  S.fe.status = null; // tcs_max.js answers (through fe) while outlet 0 runs
  S.fe.warnings = [];
  const d = new Dict(NOTES_DICT);
  d.parse(JSON.stringify(N.payload(m, S.name)));
  outlet(0, "dictionary", NOTES_DICT);
}

function read(path) {
  if (path === undefined) return fail("read needs a file name");
  const p = String(path);
  let obj;
  try {
    obj = JSON.parse(readText(p));
  } catch (e) {
    return fail(e instanceof SyntaxError ? `${p} is not valid JSON: ${e.message}` : e);
  }
  load(obj, p.split("/").pop().replace(/\.json$/i, ""));
}

function dictionary(name) {
  let obj;
  try {
    obj = JSON.parse(new Dict(String(name)).stringify());
  } catch (e) {
    return fail(e);
  }
  load(obj, String(name));
}

// the result of tcs_ui_edit.js: keep it; a commit sends the notes, a curve step writes the buffer
function route(r) {
  S.ed = r.ed;
  if (r.view) S.view = r.view;
  if (r.message) {
    S.message = r.message;
    outlet(2, "message", r.message);
  }
  if (r.commit) changed();
  else update();
  if (r.curve) {
    writeCurve();
    if (!r.commit) notifyclients();
  }
  if (r.focus === "lyrics") outlet(3, "select");
  if (r.transport === "toggle") (S.playing ? stop : play)();
  else if (r.transport) play(r.transport.at);
}

// a message's command: its errors are errors
function command(r) {
  if (r.message && !r.commit) return fail(r.message);
  route(r);
}

function language() {
  const l = Array.from(arguments).map(String).join(" ");
  command(E.apply(S.ed, "language", (m) => M.setLanguage(m, l)));
}

function tempo(bpm) {
  command(E.apply(S.ed, "tempo", (m) => M.setTempo(m, Number(bpm))));
}

function bang() {
  sendNotes();
}

function clear() {
  S.ed = E.setCurve(S.ed, null).ed;
  S.f0.stale = false;
  S.hover = null;
  S.playhead = null;
  command(E.apply(S.ed, "clear", (m) => ({ ...m, notes: [], tail: [] }), { selection: [] }));
}

function write(path) {
  if (path === undefined) return fail("write needs a file name");
  let res;
  try {
    res = N.toScore(N.payload(model(), S.name));
  } catch (e) {
    return fail(e);
  }
  for (const w of res.warnings) outlet(2, "warning", w);
  try {
    const f = new File(String(path), "write", "TEXT");
    if (!f.isopen) throw new Error(`cannot write ${path}`);
    f.eof = 0;
    const bytes = utf8Encode(JSON.stringify(res.score, null, 1) + "\n");
    for (let k = 0; k < bytes.length; k += 32768) f.writebytes(bytes.slice(k, k + 32768));
    f.close();
    post(`tcs_ui: wrote ${path}\n`);
  } catch (e) {
    fail(e);
  }
}

// the front-end's status: ok <pieces> <seconds> | warning <text> | error <text>
function fe(kind) {
  const args = Array.from(arguments).slice(1).map(String);
  const k = String(kind);
  if (k === "ok") {
    S.fe.status = "ok";
    S.fe.pieces = Number(args[0]);
    S.fe.seconds = Number(args[1]);
    try {
      const r = JSON.parse(new Dict(COND_DICT).stringify());
      S.fe.words = r.words || null;
      S.fe.pieceList = r.pieces || null;
    } catch (e) {
      S.fe.words = null;
      S.fe.pieceList = null;
    }
    S.labels = S.fe.words ? D.phonemeLabels(S.fe.words) : null;
  } else if (k === "warning") {
    S.fe.warnings.push(args.join(" "));
  } else if (k === "error") {
    S.fe.status = "error";
    S.fe.shown = args.join(" ");
    S.fe.words = null;
    S.labels = null;
  } else return;
  statusLine();
}

// ---- the lyrics line (U5, UWP5)

// from the textedit: `text` and the line as one symbol (@outputmode 1), or as several atoms
function text() {
  S.message = null;
  route(E.lyrics(S.ed, Array.from(arguments).map(String).join(" ")));
}

// ---- editing by message (U6)

const ids = (args) => args.map(Number).filter((x) => Number.isFinite(x));

function tool(t) {
  command(E.setTool(S.ed, String(t)));
}

function mode(m) {
  command(E.setMode(S.ed, String(m)));
}

function select() {
  const a = Array.from(arguments);
  if (a[0] === "all") route(E.selectAll(S.ed));
  else if (a[0] === "none" || !a.length) route(E.select(S.ed, []));
  else route(E.select(S.ed, ids(a)));
}

function add(start, dur, midi, lyric) {
  command(E.addNote(S.ed, Number(start), Number(dur), Number(midi), lyric === undefined ? "la" : String(lyric)));
}

function move(dBeats, dMidi) {
  command(E.move(S.ed, S.view, Number(dBeats) || 0, Number(dMidi) || 0, ids(Array.from(arguments).slice(2))));
}

function resize(id, start, stop) {
  command(E.resize(S.ed, Number(id), Number(start), Number(stop)));
}

function split(id, at) {
  command(E.split(S.ed, [Number(id)], Number(at)));
}

function join() {
  command(E.join(S.ed, ids(Array.from(arguments))));
}

function rest(beat, type) {
  command(E.rest(S.ed, Number(beat), type === undefined ? "toggle" : String(type)));
}

function undo() { route(E.undo(S.ed)); }
function redo() { route(E.redo(S.ed)); }
function copy() { route(E.copy(S.ed)); }
function cut() { route(E.cut(S.ed)); }
function duplicate() { route(E.duplicate(S.ed)); }

function paste(beat) {
  route(E.paste(S.ed, S.view, beat === undefined ? undefined : Number(beat)));
}

// `delete` is a reserved word in JS, so it comes here
function anything() {
  const args = Array.from(arguments);
  if (messagename === "delete") return command(E.del(S.ed, ids(args)));
  fail(`unknown message ${messagename}`);
}

// ---- the curve (U3: the generated one and the working one, which the F0 mode edits, UWP6)

function f0(what) {
  const w = String(what);
  const args = Array.from(arguments).slice(1);
  if (w === "buffer" && args.length) S.f0.buffer = String(args[0]);
  else if (w === "clear") { S.ed = E.setCurve(S.ed, null).ed; S.f0.stale = false; }
  else if (w === "refresh") {
    let gen;
    try {
      const b = new Buffer(S.f0.buffer);
      const n = b.framecount();
      if (!(n > 0)) throw new Error(`buffer~ ${S.f0.buffer} is empty or missing`);
      const vals = b.peek(1, 0, n);
      gen = Array.from(typeof vals === "number" ? [vals] : vals);
    } catch (e) {
      return fail(e);
    }
    const r = E.setCurve(S.ed, gen);
    S.ed = r.ed;
    S.f0.stale = false;
    if (r.dropped) {
      S.message = `regenerated: ${r.dropped} hand-edited frame(s) dropped`;
      outlet(2, "message", S.message);
    }
  } else {
    S.message = null;
    return command(E.f0(S.ed, S.view, w, args));
  }
  notifyclients();
  statusLine();
}

// the working curve into the buffer (sized to it, at 50 Hz): one poke, then outlet 1 so that sing reads it again
function writeCurve() {
  const c = S.ed.curve;
  if (!c) return;
  try {
    const b = new Buffer(S.f0.buffer);
    if (b.framecount() !== c.length) b.send("sizeinsamps", c.length);
    b.send("sr", CURVE_SR);
    b.poke(1, 0, c);
  } catch (e) {
    return fail(`f0: ${e.message}`);
  }
  S.writes++;
  outlet(1, "bang");
}

// ---- saving with the patch (U8)

function currentState() {
  return ST.pack({ model: model(), view: S.view, f0: { gen: S.ed.gen, work: S.ed.history.curve, stale: S.f0.stale } });
}

function save() {
  try {
    embedmessage("state", ST.encode(currentState()));
  } catch (e) {
    fail(`save: ${e.message}`);
  }
}

function getvalueof() {
  const d = new Dict();
  d.parse(JSON.stringify({ format: ST.FORMAT, version: ST.VERSION, json: ST.encode(currentState()) }));
  S.value = d;
  return d;
}

function setvalueof() {
  const a = arguments[0];
  let src = null;
  try {
    if (a && typeof a === "object" && typeof a.stringify === "function") src = JSON.parse(a.stringify()).json;
    else if (a === "dictionary" && arguments.length > 1) src = JSON.parse(new Dict(String(arguments[1])).stringify()).json;
  } catch (e) {
    src = null;
  }
  if (typeof src === "string") restore(src, "pattr"); // anything else (e.g. the 0 of parameter_enable) is ignored
}

// the message embedded by save()
function state(json) {
  restore(String(json), "embed");
}

function restore(json, via) {
  let st;
  try {
    st = ST.decode(json);
  } catch (e) {
    return fail(`restore (${via}): ${e.message}`);
  }
  const m = st.model || M.emptyModel(); // a roll saved empty before UWP4 had no model
  S.ed = E.reset(S.ed, m, st.f0.work, st.f0.gen);
  S.name = m.name || (st.model ? S.name : "untitled") || "score";
  S.f0.stale = st.f0.stale;
  S.hover = null;
  S.playhead = null;
  S.fe.status = null;
  S.labels = null;
  S.view = V.clampView({ ...S.view, ...st.view, maxBeats: V.extentBeats(m, S.view.limitSeconds) });
  update();
  if (m.notes.length) {
    S.pending = true;
    if (S.loaded) flush();
    else later(flush, 0);
  }
  outlet(2, "restored", via, m.notes.length);
}

// a restore's outputs, once the patch is connected: the notes (no score2f0), then the curve and outlet 1
function flush() {
  if (!S.pending) return;
  S.pending = false;
  const stale = S.f0.stale;
  sendNotes();
  S.f0.stale = stale; // the saved curve belongs to these notes
  writeCurve();
  statusLine();
}

function loadbang() {
  S.loaded = true;
  flush();
  S.line = null; // the textedit is connected now
  sendLine();
}

function report() {
  const d = ST.digest(currentState());
  d.selection = S.ed.selection.slice();
  d.tool = S.ed.tool;
  d.undo = S.ed.history.past.length;
  d.redo = S.ed.history.future.length;
  d.line = S.line;
  d.phonemes = S.labels ? [...S.labels].map(([id, l]) => ({ id, text: l.text, source: l.source, typed: l.typed, ids: l.ids })) : null;
  d.drawn = S.drawn;
  d.mode = S.ed.mode;
  d.f0tool = S.ed.f0tool;
  d.gen_hash = S.ed.gen ? ST.fnv(JSON.stringify(Array.from(S.ed.gen, Math.fround))) : null;
  d.edited = F.edited(S.ed.gen, S.ed.history.curve);
  d.writes = S.writes;
  d.playing = S.playing;
  d.playhead = S.playhead;
  outlet(2, "report", JSON.stringify(d));
}

// ---- the view

function view() {
  const a = Array.from(arguments);
  if (a[0] === "fit") S.view = V.fit(S.view, model());
  else if (a.length === 4 && a.every((x) => typeof x === "number")) S.view = V.showRange(S.view, a[0], a[1], a[2], a[3]);
  else return fail("view: expected fit or <beat0> <beat1> <midiLo> <midiHi>");
  redraw();
}

function zoom(fx, fy) {
  S.view = V.zoomPitch(V.zoomTime(S.view, Number(fx) || 1), Number(fy) || 1);
  redraw();
}

function scrollby(dBeats, dRows) {
  S.view = V.scroll(S.view, Number(dBeats) || 0, Number(dRows) || 0);
  redraw();
}

function viewSetting(fn) {
  try {
    S.view = fn(S.view);
  } catch (e) {
    return fail(e);
  }
  statusLine();
}

function snap(value) { viewSetting((v) => V.setSnap(v, String(value))); }
function bar(beats) { viewSetting((v) => V.setBar(v, beats)); }
function limit(seconds) { viewSetting((v) => V.setLimit(v, seconds, model())); }
function range(lo, hi) { viewSetting((v) => V.setRange(v, lo, hi)); }
function phonemes(on) { viewSetting((v) => V.setPhonemes(v, on === undefined || (Number(on) !== 0 && String(on) !== "off"))); }

// ---- playback (UWP7): the patch plays buffer~ tcs_out with [line~] -> [play~], and sends the position back

// from `at` seconds, else from the playhead (or the start, when there is none or it is at the phrase's end)
function play(at) {
  const m = model();
  let from = at === undefined ? S.playhead : Number(at);
  if (from === null || !Number.isFinite(from) || (at === undefined && from >= M.seconds(m) - 1e-3)) from = 0;
  const r = E.playRange(m, from);
  const ms = (x) => Math.round(x * 1000);
  S.playing = true;
  S.playhead = r.from;
  outlet(4, "play", ms(r.from), ms(r.to), ms(r.to) - ms(r.from));
  statusLine();
}

function stop() {
  S.playing = false;
  outlet(4, "stop");
  statusLine();
}

function stopped() {
  if (!S.playing) return;
  S.playing = false;
  statusLine();
}

// the position while playing (snapshot~ sends it on and on with the audio on: drawn only when it moves)
function playhead(seconds) {
  const p = seconds === undefined || seconds === "off" ? null : Number(seconds);
  if (p === S.playhead) return;
  S.playhead = p;
  if (S.playing) statusLine();
  else redraw();
}

// ---- painting

function paint() {
  const size = mgraphics.size;
  if (size[0] !== S.view.w || size[1] !== S.view.h) S.view = V.resize(S.view, size[0], size[1]);
  const t = Date.now();
  S.drawn = D.draw(mgraphics, S.view, scene()).phonemes;
  S.paints.ms.push(Date.now() - t);
  S.paints.at.push(t);
  if (S.paints.ms.length > 5000) { S.paints.ms.shift(); S.paints.at.shift(); }
}

function stats(what) {
  if (String(what) === "reset") {
    S.paints = { ms: [], at: [], since: Date.now() };
    return;
  }
  const ms = S.paints.ms.slice().sort((a, b) => a - b);
  const secs = (Date.now() - S.paints.since) / 1000;
  const q = (p) => (ms.length ? ms[Math.min(ms.length - 1, Math.floor(ms.length * p))] : null);
  outlet(2, "stats", JSON.stringify({ paints: ms.length, secs: Math.round(secs * 100) / 100,
    paints_per_s: secs > 0 ? Math.round((ms.length / secs) * 10) / 10 : null, paint_ms: { median: q(0.5), p90: q(0.9),
      max: ms.length ? ms[ms.length - 1] : null }, notes: model().notes.length, size: [S.view.w, S.view.h] }));
}

// the roll drawn offscreen at `scale`, written by Jitter's exportimage (PNG from the extension)
function snapshot(path, scale) {
  if (path === undefined) return fail("snapshot needs a file name");
  const k = Number(scale) || 2;
  try {
    const w = Math.round(S.view.w * k);
    const h = Math.round(S.view.h * k);
    const mg = new MGraphics(w, h);
    mg.relative_coords = 0;
    mg.autofill = 0;
    mg.scale(k, k);
    D.draw(mg, S.view, scene());
    const img = new Image(mg);
    const mat = new JitterMatrix(SNAPSHOT_MATRIX, 4, "char", w, h);
    img.tonamedmatrix(SNAPSHOT_MATRIX);
    mat.exportimage(String(path));
  } catch (e) {
    return fail(`snapshot: ${e.message}`);
  }
  outlet(2, "snapshot", String(path));
}

// ---- the mouse and the keys (tcs_ui_edit.js does the work)

const mods = (cmd, shift, alt) => ({ cmd: !!cmd, shift: !!shift, alt: !!alt });

function onclick(x, y, button, cmd, shift, capslock, alt) {
  S.message = null;
  route(E.press(S.ed, S.view, x, y, mods(cmd, shift, alt)));
}

// button 0: the mouse is up (it comes after a click with no movement too, UWP0)
function ondrag(x, y, button, cmd, shift, capslock, alt) {
  const m = mods(cmd, shift, alt);
  if (button) {
    route(E.drag(S.ed, S.view, x, y, m));
    autoScroll(x, y, m);
  } else {
    stopAutoScroll();
    route(E.release(S.ed, S.view, x, y, m));
    if (hoverAt(x, y)) statusLine();
  }
}

function ondblclick(x, y, button, cmd, shift, capslock, alt) {
  route(E.dblclick(S.ed, S.view, x, y, mods(cmd, shift, alt)));
}

function onkeydown(code, char, updown, cmd, shift, capslock, alt) {
  S.message = null;
  const r = E.key(S.ed, S.view, Number(code), Number(char), mods(cmd, shift, alt));
  route(r);
  return r.consumed ? 1 : 0;
}

// a drag at the edge of the roll scrolls it (the keys, the ruler, the toolbar, or the right edge zone)
function autoScroll(x, y, m) {
  const v = S.view;
  const dx = x < v.keysW ? x - v.keysW : x > v.w - AUTO_EDGE ? x - (v.w - AUTO_EDGE) : 0;
  const dy = y < v.rulerH ? y - v.rulerH : y > v.h - v.statusH ? y - (v.h - v.statusH) : 0;
  if ((dx || dy) && S.ed.gesture) {
    S.auto = { x, y, m, dx: Math.max(-40, Math.min(40, dx)), dy: Math.max(-40, Math.min(40, dy)) };
    if (!S.autoTimer) S.autoTimer = every(autoTick, AUTO_MS);
  } else stopAutoScroll();
}

function autoTick() {
  const a = S.auto;
  if (!a || !S.ed.gesture) return stopAutoScroll();
  S.view = V.scroll(S.view, (a.dx * 0.5) / S.view.pxPerBeat, (-a.dy * 0.5) / S.view.rowH);
  route(E.drag(S.ed, S.view, a.x, a.y, a.m));
}

function stopAutoScroll() {
  S.auto = null;
  if (S.autoTimer) { S.autoTimer(); S.autoTimer = null; }
}

function onwheel(x, y, dx, dy, cmd, shift, capslock, alt) {
  if (cmd) S.view = V.zoomTime(S.view, Math.exp(dy * WHEEL_ZOOM), x);
  else if (alt) S.view = V.zoomPitch(S.view, Math.exp(dy * WHEEL_ZOOM), y);
  else if (shift) S.view = V.scroll(S.view, -((dx || dy) * WHEEL_PX) / S.view.pxPerBeat, 0);
  else S.view = V.scroll(S.view, -(dx * WHEEL_PX) / S.view.pxPerBeat, (dy * WHEEL_PX) / S.view.rowH);
  hoverAt(x, y);
  statusLine();
}

// the note (or its edge) or the rest label under the pointer, or in the F0 mode the curve's frame; true if it changed
function hoverAt(x, y) {
  S.ed = { ...S.ed, pointer: { x, y } };
  const m = model();
  if (S.ed.mode === "f0") {
    const c = S.ed.curve;
    const j = c && V.inRoll(S.view, x, y) ? Math.round(V.beatsToSeconds(V.xToBeat(S.view, x), m.tempo) * F.FPS) : null;
    const f = j !== null && j >= 0 && j < c.length ? j : null;
    const was = S.hoverFrame !== f || S.hover || S.hoverRest;
    S.hover = null;
    S.hoverRest = null;
    S.hoverFrame = f;
    return !!was;
  }
  S.hoverFrame = null;
  const h = V.hitNote(S.view, m.notes, x, y);
  const r = !h && V.inRestBand(S.view, x, y) ? M.restAt(m, V.xToBeat(S.view, x)) : null;
  const key = (h ? `${h.id}:${h.part}` : "") + (r ? `r${r.start}` : "");
  const was = (S.hover ? `${S.hover.id}:${S.hover.part}` : "") + (S.hoverRest ? `r${S.hoverRest.start}` : "");
  S.hover = h;
  S.hoverRest = r;
  return key !== was;
}

function onidle(x, y) {
  if (hoverAt(x, y)) statusLine();
}

function onidleout() {
  S.ed = { ...S.ed, pointer: null };
  if (S.hover || S.hoverRest || S.hoverFrame !== null) { S.hover = null; S.hoverRest = null; S.hoverFrame = null; statusLine(); }
}

function onresize(w, h) {
  S.view = V.resize(S.view, w, h);
  redraw();
}

update();
