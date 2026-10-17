# neural_tilde host changes for TCSinger2: requirements

Changes to the `neural.gen~` host (and the matching exporter and protocol) in `~/Documents/Max 9/Packages/neural_tilde`, needed to run TCSinger2 in Max. This is work package EWP6 of `EXPORTING_WORK_PACKAGES.md` (this repo). The work is done in a separate session, in the neural_tilde repo.

**Status:** R1–R5 are implemented in neural_tilde 0.0.5 (commit `539f861`), and its headless tests pass (`examples/host_tests/check_cli.py`). N1–N3 are open. See EWP6 in `EXPORTING_WORK_PACKAGES.md`.

Written 2026-10-16, from reading neural_tilde at commit `765724a`: `PROTOCOL.md`, `python_tools/gen_module.py`, `_exporter_base.py`, `src/generative/gen_runner.cpp`, `src/backend/backend.cpp`, `src/frontend/maxmsp/neural.gen_tilde/neural.gen_tilde.cpp`.

## What TCSinger2 will load

Two gen models (details in `EXPORTING_WORK_PACKAGES.md`, "The interface"):

- **`tcsinger2_f0.pte`, method `score2f0`**
  - Inputs: pieces conditions (seven `[1, 256]` int64/float32 tensors and `language` `[1, 1]` int64, all from one `dict`) and a noise `[1, 2, 3760]`.
  - Output: an F0 curve, `[1, 1, 1003]` at **50 Hz**, as MIDI pitch (values 0–127, 0 = unvoiced).
- **`tcsinger2_sing.pte`, method `sing`** (about 2 GB, MLX)
  - Inputs:
    - the same pieces conditions;
    - `input_ids` `[1, 80]` int64 from `neural.tokenizer`;
    - the attribute `cfg_scale` (float32);
    - a **buffer input `f0`** `[1, 1, 1003]` at 50 Hz: the curve from `score2f0`, possibly edited in Max;
    - a noise `[1, 20, 940]`.
  - Output: audio `[1, 1, 962560]` at 48 kHz.

In the patch, `score2f0` writes `buffer~ f0`; the user may edit it, for example by drawing in `waveform~`; then `init f0` passes it to `sing`.

## Requirements

Priorities:
- **v1**: needed for TCSinger2 version 1.
- **v2**: needed when the audio prompt is added.
- **nice**: helpful but not needed.

All changes keep old models and patches working: new metadata fields are optional, and their defaults give today's behaviour.

| id | priority | requirement |
|---|---|---|
| R1 | v1 | int64 attributes are passed as int64 |
| R2 | v1 | a buffer input can be read as it is, without resampling |
| R3 | v1 | a control-rate output (50 Hz) is written to a `buffer~` unchanged |
| R4 | v1 | verify: long conditions from a `dict` loaded from JSON |
| R5 | v2 | several buffer inputs per method, each with its own `buffer~` |
| N1 | nice | a condition may be supplied shorter than its shape and is padded |
| N2 | nice | busy status and progress reporting |
| N3 | nice | load the `.pte` without blocking Max |

### R1 – int64 attributes (v1)

**Now:**
- In `GenRunner::generate` (`src/generative/gen_runner.cpp:157–164`), an attribute's value is written as `float` into its storage, whatever its dtype. For an `int64` attribute the storage is 8 bytes, so the model receives the bit pattern of a float in the low 4 bytes.
- The live path does it right (`src/backend/backend.cpp:480–497`: `std::llround` into an `int64_t` when the dtype is `Long`).

**Required:**
- For `dtype` `int64`, write `(int64_t)std::llround(value)` after clamping. `float32` attributes are unchanged.
- Check whether `neural.gen~` declares an int64 attribute as a Max `float` attribute. It reads values with `object_attr_getfloat` in `gen::trigger`. If so, it may stay a float attribute, as long as R1 converts at run time; an `int` attribute would be nicer.

**Accept:** a toy gen model with an `int64` attribute `k` (default 3, range 0–10) that outputs `k` as a constant audio buffer. In Max, the buffer holds exactly 3, then 7 after `k 7`.

### R2 – a buffer input read as it is (v1)

**Now:**
- `gen` reads the `init` `buffer~` and calls `GenRunner::prepare_init_audio` (`gen_runner.cpp:67–96`, called from `neural.gen_tilde.cpp:671`). It linearly resamples from the `buffer~`'s own sample rate to the declared `sample_rate`, maps channels, and crops or pads with 0.
- If the `buffer~` is at the declared rate, the copy is exact. Otherwise, for a control curve:
  - **A `buffer~` at another rate:** this is the usual case when a user makes their own, since it starts at the patch's rate. The curve is then resampled as if it were audio. A 1,003-sample curve in a 44.1 kHz `buffer~` would be read as 0.02 s of audio, stretched to 1,003 samples at 50 Hz, which is useless.
  - **Voicing edges:** linear interpolation between a voiced value (e.g. 64.0) and 0 (unvoiced) makes glides to 0. Any F0 curve with gaps is distorted this way.

**Required:**
- **New metadata** on buffer inputs, both fields optional:
  - `"resample": false`: copy samples 1:1, ignoring the `buffer~`'s sample rate;
  - `"pad_value": <number>`: what fills past the end of the source (default 0.0).
  
  Without `"resample"` (or with `true`), today's behaviour is kept.
- **With `"resample": false`:**
  - read `length` samples per channel, starting at the offset of `init <buffer> [ms]`, with the ms converted at the declared `sample_rate`;
  - map channels as now;
  - fill the rest with `pad_value`;
  - post a warning in the Max console when the `buffer~`'s sample rate differs from the declared one;
  - post nothing when the source is longer; it is just cropped.
- **Exporter:** `GenModule.register_buffer_input(name, channels, length, sample_rate=44100, description="", dtype="float32", resample=True, pad_value=0.0)` writes the two fields; they are written only when they differ from the defaults.
- **Documentation:** `PROTOCOL.md` §2.4.5 describes the fields and their use for control curves; `neural.info` shows them.

**Accept:** a toy gen model whose method returns its buffer input `[1, 1, 1003]` (declared `sample_rate` 50, `resample: false`).
- A `buffer~` at 44.1 kHz filled with a known curve (a ramp with runs of zeros) comes back sample for sample.
- The same with `init <buffer> 100` comes back shifted by 5 samples (100 ms at 50 Hz).
- A 500-sample source comes back padded with `pad_value`.

### R3 – control-rate output (v1)

**Now** (verified in code, not yet in Max):
- `gen::on_done` (`neural.gen_tilde.cpp:835–927`) resizes the target `buffer~` to the output length.
- It writes the floats unchanged (no clipping).
- It stamps the `buffer~`'s sample rate with an `sr` message carrying the output's `sample_rate`.
- What has not been checked is whether Max's `buffer~` accepts a sample rate of 50 and whether `waveform~` shows and edits such a buffer normally.

**Required:**
- Verify in Max that an output with `sample_rate` 50 and values up to 127 lands in the `buffer~` exactly: 1,003 frames, sample rate 50, values unchanged.
- Fix the host if Max refuses or clamps that sample rate. For example, stamp nothing for rates below a threshold and warn, as long as R2 makes the rate irrelevant for reading the curve back.
- Note the result in `PROTOCOL.md` §2.5.

**Accept:**
- A toy gen model that outputs a known `[1, 1, 1003]` curve at 50 Hz (values 0–127) writes it into `buffer~ f0` exactly.
- Passed back with `init f0` to the R2 toy model, it comes out unchanged, so the round trip `score2f0` → `buffer~` → `sing` loses nothing.
- After drawing in `waveform~`, the buffer stays at 1,003 frames.

### R4 – long conditions from a `dict` (v1, verify)

**Now** (verified in code):
- `gen::receive_dict` (`neural.gen_tilde.cpp:478`) matches dictionary keys to condition names, converts the values to the declared dtype (`shared/cond_bytes.h`) and ignores other keys.
- `GenRunner::generate` rejects a condition whose number of elements differs from its shape (`gen_runner.cpp:139–144`).
- Missing conditions are filled with zeros and a warning (`gen::trigger`).

**Required:** verify in Max, and fix what fails:
- **One message:** a `dict` loaded from a JSON file (`dict` `import`, or `read`) with eight conditions arrives in one `dictionary` message. The conditions are seven arrays of 256 numbers and one of 1, plus extra keys holding strings and nested arrays for display, which are ignored.
- **Mixed numbers:** a `float32` condition receives values that JSON writes as integers (`1` instead of `1.0`) correctly, and an `int64` condition receives `-1` and `430` correctly.
- **A second `dict`:** sending a second `dict` replaces the conditions it contains and keeps the others (as documented).

**Accept:** a toy gen model with conditions `a [1, 256]` int64 and `b [1, 256]` float32, which outputs `a` and `b` concatenated as audio `[1, 1, 512]`. A JSON file with mixed integer and float values, -1 entries and an extra `"words"` key comes back exactly.

### R5 – several buffer inputs per method (v2)

**Now:** `GenRunner::bind_init_audio` (`gen_runner.cpp:42–50`) gives the one `init` `buffer~` to every buffer-role input, so a method cannot take two different buffers. Version 2 of `sing` needs both an F0 curve (`f0`, R2) and an audio prompt (`prompt`, `[1, 1, 384000]` at 48 kHz).

**Required:**
- `init <input_name> <buffer> [ms]` sets the source of one buffer input by name.
- `init <buffer> [ms]` keeps its meaning for models with one buffer input. With several, it sets the first one and posts a warning.
- Each input is read with its own geometry, sample rate and R2 settings.
- An input with no source is filled with `pad_value` (0 by default) and gets a warning, as now.
- `neural.info` lists each buffer input with its name.

**Accept:** a toy gen model with buffer inputs `a` and `b` (each `[1, 1, 1024]`) that outputs `a + 2 b`. With `init a bufA` and `init b bufB` the output matches; with only `init bufA`, `b` is silence and a warning is posted.

### N1 – shorter conditions padded (nice)

An optional `"pad_value"` on a condition: the host then accepts fewer elements than its shape and fills the rest (in flat, row-major order) with the pad value. A longer one is still an error. This lets a front-end in Max (the planned JS one) send only the real pieces. Without it, the front-end pads, as the Python tool will.

### N2 – busy status and progress (nice)

`sing` will take several seconds. Today a `generate` during a generation posts "still generating, ignoring" (`gen::trigger`), and the only signal is the bang when it is done. Wanted:
- a status message on an outlet, `busy 1` when a generation starts and `busy 0` when it ends or fails;
- if feasible, a way to drop a running generation's result (`stop`). ExecuTorch cannot interrupt `execute()`, so the worker would finish and discard.

### N3 – loading without blocking Max (nice)

The `.pte` is loaded when the object is created (`neural.gen_tilde.cpp:446`). For a 2 GB MLX program this may freeze Max for some time; it is not measured yet. Wanted:
- load on a worker thread, with a bang or status message when ready;
- `generate` refused with a message until then.

At least, the load time should be posted in the console.

## Not required

- Several outputs per method: TCSinger2 uses two files instead.
- Dynamic time (`dynamic_time`): TCSinger2 is exported at a fixed 20 s.
- Changes to `neural.live~`.

## Test models

The toy models of the acceptance tests should live in neural_tilde (e.g. `examples/host_tests/export_host_tests.py` writing `toy_int_attr`, `toy_buffer_asis`, `toy_f0_out`, `toy_conditions`, `toy_two_buffers`), with a Max test patch next to them. Where possible, also check them without Max with `src/generative/gen_cli.cpp`.

Done when all v1 tests pass in Max on macOS with the MLX delegate, and `PROTOCOL.md`, the exporter docstrings and the `neural.gen~` help patch describe R1–R4. R5 can follow in a later session.
