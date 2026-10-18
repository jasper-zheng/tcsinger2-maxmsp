"""The front-end inside the graph (EWP3): the pieces conditions of a score to its frames, in tensor ops.

FrontEnd(vae) takes the conditions of scripts/score_to_max.py ("The interface" of EXPORTING_WORK_PACKAGES.md),
padded to P = 256 pieces, and does what ldm.infer.timing does with the pieces that score_to_max.
pieces_from_conditions rebuilds from them:
  - the duration predictor trained with the VAE (AutoencoderKL.pred_duration), then expm1 and clamp, as
    timing.DurationPredictor;
  - fill_durations with fit 'nucleus': the pieces of a note (note_id) whose length is not given are fitted to
    the note's length, group sums through a one-hot [P, P];
  - to_frames: piece boundaries rounded on the running time (a half frame and up to TIE below it round up);
  - the frame tracks of stage 2 (ph, midi; frame_tracks) and of the F0 generator (tracks_from_pieces,
    reference_pitch), padded to T = 3,760 frames as the collaters pad them.
The duration predictor runs on all P pieces, so that the padding changes nothing:
  - its phoneme encoder (RelTransformerEncoder) takes as its length the count of phoneme ids > 0, and id 0 is
    <AP>: a phrase with k breaths has its last k pieces masked in that encoder. Python runs it on the valid
    pieces only; here the length is the count among the valid pieces, which gives the same mask;
  - the duration convs see zeros after the last valid piece, as the zero padding of a run at its own length.
The fit and rounding are float32 here, float64 in Python; a boundary can then round the other way when it
falls within about 1e-3 frame of TIE below a half frame (the export checks count them).
Every division is guarded, so all-zero inputs (the host's before a dict arrives, GenModule's example) give
finite output.
"""
import math

import torch
from torch import nn

from ldm.data.phone_set import MIDI_PAD_ID, NOTE_TYPE_PAD, PH_PAD_ID
from ldm.f0gen.features import POS_SCALE
from ldm.infer.timing import FPS, MIN_NUCLEUS, MIN_PRED, TIE

N_PIECES, N_FRAMES = 256, 3760
REF_PAD = 60.0  # ldm.f0gen.dataset.collate pads ref with 60


def _encode_phonemes(enc, ph, length):
    """utils.commons.rel_transformer.RelTransformerEncoder.forward(ph) with its length given (eval)."""
    x = enc.emb(ph) * math.sqrt(enc.hidden_channels)
    x = x.transpose(1, -1)
    x_mask = (torch.arange(x.shape[2], device=ph.device) < length.view(-1, 1))[:, None, :].to(x.dtype)
    if enc.prenet:
        x = enc.pre(x, x_mask)
    e = enc.encoder
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    for i in range(e.n_layers):  # rel_transformer.Encoder.forward; MultiHeadAttention.forward without self.attn
        x = x * x_mask
        x_ = x
        if e.pre_ln:
            x = e.norm_layers_1[i](x)
        att = e.attn_layers[i]
        y, _ = att.attention(att.conv_q(x), att.conv_k(x), att.conv_v(x), mask=attn_mask)
        x = x_ + att.conv_o(y)
        if not e.pre_ln:
            x = e.norm_layers_1[i](x)
        x_ = x
        if e.pre_ln:
            x = e.norm_layers_2[i](x)
        x = x_ + e.ffn_layers[i](x, x_mask)
        if not e.pre_ln:
            x = e.norm_layers_2[i](x)
    if e.pre_ln:
        x = e.last_ln(x)
    return (x * x_mask).transpose(1, 2)


def _and(*xs):
    """Elementwise and of bool tensors (logical_and: the MLX delegate has no aten.__and__)."""
    out = xs[0]
    for x in xs[1:]:
        out = torch.logical_and(out, x)
    return out


def _safe_div(a, b):
    return a / torch.where(b != 0, b, torch.ones_like(b))


class FrontEnd(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.ph_encoder, self.note_encoder, self.dur_predictor = vae.ph_encoder, vae.note_encoder, vae.dur_predictor
        self.register_buffer('fps', torch.tensor(FPS, dtype=torch.float32), persistent=False)
        self.register_buffer('pos_scale', torch.tensor(POS_SCALE, dtype=torch.float32), persistent=False)

    def predict(self, ph, midi, note_dur, note_type, valid):
        """AutoencoderKL.pred_duration on the valid pieces, padded: log durations [1, P] (0 on padding)."""
        length = _and(valid, ph > 0).sum(-1)
        h = _encode_phonemes(self.ph_encoder, ph, length) + self.note_encoder(midi, note_dur, note_type)
        h = h * valid[..., None].to(h.dtype)
        return self.dur_predictor(h, x_padding=torch.logical_not(valid))

    def durations(self, pred, note_dur, ph_dur, nucleus, note_id, valid):
        """timing.fill_durations (fit 'nucleus') on predictions pred [1, P] (seconds): seconds per piece."""
        P = pred.shape[-1]
        given = ph_dur >= 0
        fit = _and(valid, torch.logical_not(given))
        nuc = nucleus > 0
        G = _and(note_id[0][:, None] == torch.arange(P, device=pred.device), fit[0][:, None]).to(pred.dtype)  # piece x note
        group = lambda v: (v @ G) @ G.T  # per piece: the sum over its note's fitted pieces
        want = torch.clamp(pred, min=MIN_PRED)
        zero = torch.zeros_like(want)
        nuc_sum = group(torch.where(nuc, want, zero))
        rest = note_dur - group(torch.where(nuc, zero, want))
        has_nuc = group(nuc.to(want.dtype)) > 0
        keep_cons = _and(has_nuc, rest >= MIN_NUCLEUS * nuc_sum)
        lens = torch.where(keep_cons, torch.where(nuc, _safe_div(want * rest, nuc_sum), want),
                           _safe_div(want * note_dur, group(want)))
        return torch.where(given, ph_dur, torch.where(fit, lens, zero))

    def forward(self, ph, midi, note_dur, note_type, ph_dur, nucleus, note_id):
        """The conditions [1, P] (ph, midi, note_type, nucleus, note_id int64; note_dur, ph_dur float32).
        Returns a dict:
          n [1] int64           frames of the phrase (at most 3,760)
          ends [1, P] int64     end frame of every piece (to_frames' starts[1:]; padding: n)
          dur, log_dur [1, P]   fitted seconds, the predictor's log durations
          mel2ph, ph, midi [1, T] int64   frame_tracks (mel2ph 0, ph 430, midi 99 after n)
          ntype [1, T] int64, pos [1, 3, T], onset, ref, mask [1, T]: the F0 generator's tracks, padded as
              ldm.f0gen.dataset.collate pads them."""
        P, T = ph.shape[-1], N_FRAMES
        dev = ph.device
        valid = ph != PH_PAD_ID
        log_dur = self.predict(ph, midi, note_dur, note_type, valid)
        pred = torch.clamp(torch.expm1(log_dur), min=0.0)
        dur = self.durations(pred, note_dur, ph_dur, nucleus, note_id, valid)

        ends = torch.floor(torch.cumsum(dur, -1) * self.fps + 0.5 + TIE).to(torch.long)  # to_frames' order
        n = torch.clamp(torch.where(valid, ends, torch.zeros_like(ends)).amax(-1), max=T)
        ends = torch.clamp(torch.where(valid, ends, n.view(-1, 1)), max=T)
        starts = torch.cat([torch.zeros_like(ends[:, :1]), ends[:, :-1]], dim=-1)

        t = torch.arange(T, device=dev)
        inside = t[None] < n.view(-1, 1)                                             # [1, T]
        piece = torch.clamp((ends[0][None, :] <= t[:, None]).sum(-1), max=P - 1)[None]  # [1, T]
        take = lambda v, pad: torch.where(inside, v[0][piece], torch.full_like(piece, pad))
        frame_ph, frame_midi = take(ph, PH_PAD_ID), take(midi, MIDI_PAD_ID)
        mel2ph = torch.where(inside, piece + 1, torch.zeros_like(piece))
        ntype = take(note_type, NOTE_TYPE_PAD)

        # first and last frame of each piece's note (its pieces are contiguous); rests are notes of their own
        same = _and(note_id[0][:, None] == note_id[0][None, :], valid[0][None, :])     # [j, i]: i on j's note
        big = torch.full((P, P), T, dtype=torch.long, device=dev)
        note_start = torch.where(same, starts[0][None, :], big).amin(-1)
        note_end = torch.where(same, ends[0][None, :], -big).amax(-1)
        ns, ne = note_start[piece], note_end[piece]                                   # [1, T]
        fzero = torch.zeros(1, T, device=dev)
        pos = torch.stack([torch.log1p((t - ns).to(fzero.dtype)), torch.log1p((ne - 1 - t).to(fzero.dtype)),
                           torch.log1p((ne - ns).to(fzero.dtype))], dim=1) * self.pos_scale
        pos = torch.where(inside[:, None], pos, torch.zeros_like(pos))
        onset = torch.where(_and(inside, t == ns), torch.ones_like(fzero), fzero)
        ref = self.reference_pitch(midi, starts, ends, valid, piece, inside, t)
        return dict(n=n, ends=ends, dur=dur, log_dur=log_dur, mel2ph=mel2ph, ph=frame_ph, midi=frame_midi, ntype=ntype,
                    pos=pos, onset=onset, ref=ref, mask=inside.to(fzero.dtype))

    @staticmethod
    def reference_pitch(midi, starts, ends, valid, piece, inside, t):
        """features.reference_pitch of the frame midi track over the phrase, REF_PAD after it: a sung frame keeps
        its note; the others are interpolated (np.interp) between the last frame of the sung piece before and
        the first frame of the sung piece after, or take the one there is."""
        P = midi.shape[-1]
        ids = torch.arange(P, device=midi.device)
        sung = _and(valid[0], midi[0] > 0, midi[0] < MIDI_PAD_ID, ends[0] > starts[0])
        before = _and(sung[None, :], ids[None, :] < ids[:, None])                     # [j, i]: i sung, before j
        after = _and(sung[None, :], ids[None, :] > ids[:, None])
        prev = torch.where(before, ids[None, :], torch.full((P, P), -1, dtype=torch.long, device=midi.device)).amax(-1)
        nxt = torch.where(after, ids[None, :], torch.full((P, P), P, dtype=torch.long, device=midi.device)).amin(-1)
        has_prev, has_next = prev >= 0, nxt < P
        prev, nxt = torch.clamp(prev, min=0), torch.clamp(nxt, max=P - 1)
        m = midi[0].to(torch.float32)
        x0, y0 = (ends[0][prev] - 1).to(m.dtype), m[prev]
        x1, y1 = starts[0][nxt].to(m.dtype), m[nxt]
        p = piece[0]
        tf = t.to(m.dtype)
        slope = _safe_div(y1[p] - y0[p], x1[p] - x0[p])
        between = slope * (tf - x0[p]) + y0[p]
        hp, hn = has_prev[p], has_next[p]
        rest = torch.where(_and(hp, hn), between, torch.where(hp, y0[p], torch.where(hn, y1[p], torch.full_like(tf, REF_PAD))))
        ref = torch.where(sung[p], m[p], rest)
        return torch.where(inside, ref[None], torch.full_like(ref[None], REF_PAD))
