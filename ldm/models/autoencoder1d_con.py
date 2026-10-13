"""
Custom Audio Encoder: contrastive learning on top of a trained, frozen VAE.
- Speech encoder with the same architecture as the VAE encoder, started from its weights
- Content branch: frame-level phoneme and note embeddings with cross-attention to the text prompt
- The three embeddings (singing, speech, content) are aligned in the space of the VAE latent, where the
  frozen singing encoder is the anchor
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.optim.lr_scheduler import LambdaLR
from ldm.util import instantiate_from_config
from utils.commons.rel_transformer import RelTransformerEncoder
from utils.commons.duration import DurationPredictor, NoteEncoder
from ldm.data.phone_set import NUM_PH, NUM_MIDI

# Modules of the VAE. They are frozen, and left out of this model's checkpoints (they come from ckpt_path).
FROZEN = ('encoder.', 'decoder.', 'quant_conv.', 'post_quant_conv.', 'dur_predictor.', 'note_encoder.', 'ph_encoder.', 'loss.')


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,          # the trained VAE
                 con_ckpt_path=None,      # trained speech encoder and content branch (scripts/export_con.py)
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 contrastive_temp=0.07,  # Temperature for contrastive loss
                 lr=5e-5,                # used as it is; main.py's batch-size scaling of base_learning_rate is not
                 warmup_steps=1000,
                 text_encoder_path="useful_ckpts/flan-t5-large",
                 text_max_length=80,
                 **kwargs,
                 ):
        super().__init__()
        self.image_key = image_key
        self.contrastive_temp = contrastive_temp
        self.lr = lr
        self.warmup_steps = warmup_steps

        # Original autoencoder components
        self.encoder = Encoder1D(**ddconfig)
        self.decoder = Decoder1D(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv1d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        hidden=ddconfig['hidden_size']

        self.dur_predictor = DurationPredictor(
            ddconfig['hidden_size'],
            n_chans=ddconfig['hidden_size'],
            n_layers=ddconfig['dur_predictor_layers'],
            dropout_rate=ddconfig['predictor_dropout'],
            kernel_size=ddconfig['dur_predictor_kernel'])

        self.note_encoder = NoteEncoder(n_vocab=NUM_MIDI, hidden_channels=ddconfig['hidden_size'])
        self.ph_encoder = RelTransformerEncoder(
                NUM_PH, ddconfig['hidden_size'], ddconfig['hidden_size'],
                ddconfig['hidden_size']*4, ddconfig['num_heads'], ddconfig['enc_layers'],
                ddconfig['enc_ffn_kernel_size'], ddconfig['dropout'], prenet=ddconfig['enc_prenet'], pre_ln=ddconfig['enc_pre_ln'])

        # Add speech encoder (same architecture as main encoder)
        self.speech_encoder = Encoder1D(**ddconfig)

        # Content branch: frame-level phonemes and notes
        self.ph_embed   = nn.Embedding(NUM_PH, hidden)
        self.midi_embed = nn.Embedding(NUM_MIDI, hidden)
        self.ph_conv   = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.midi_conv = nn.Conv1d(hidden, hidden, 3, padding=1)

        # Text encoder (frozen FLAN-T5). It is loaded when the first caption arrives and kept out of the
        # module tree, so it is neither in the state dict nor loaded when only the audio encoders are used.
        self.text_encoder_path = text_encoder_path
        self.text_max_length = text_max_length
        self._text_encoder = []   # [tokenizer, T5 encoder] once loaded
        self._text_cache = {}     # caption -> (token features [L, txt_dim], padding mask [L])
        txt_dim = 1024            # FLAN-T5-large
        self.txt_proj = nn.Linear(txt_dim, hidden, bias=False)

        # Cross-Attention: query=h_content(seq=T) ; key/value = text(seq=L)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=8, batch_first=True)
        # Content embedding -> the latent space the audio embeddings live in
        self.content_proj = nn.Linear(hidden, embed_dim)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if con_ckpt_path is not None:
            self.init_con_from_ckpt(con_ckpt_path)
        self.freeze_original_ae()

    def freeze_original_ae(self):
        """Freeze original autoencoder components and set to eval mode"""
        for name, module in self.named_children():
            if (name + '.').startswith(FROZEN):
                module.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.freeze_original_ae()  # Lightning switches the whole model to train mode around validation
        return self

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load the trained VAE and start the speech encoder from its encoder."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        not_restored = [k for k in missing if k.startswith(FROZEN)]
        assert not not_restored, f"{path} lacks VAE weights: {not_restored[:5]}"
        if not any(k.startswith("speech_encoder.") for k in sd):
            self.speech_encoder.load_state_dict(self.encoder.state_dict())
        print(f"AutoencoderKL Restored from {path} Done")

    def init_con_from_ckpt(self, path):
        """Load the trained parts (everything but the VAE) from an export of scripts/export_con.py."""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        not_restored = [k for k in missing if not k.startswith(FROZEN)]
        assert not not_restored and not unexpected, f"{path} does not fit: {not_restored[:5]} {unexpected[:5]}"
        print(f"Custom audio encoder restored from {path}")

    def on_save_checkpoint(self, checkpoint):
        for k in [k for k in checkpoint["state_dict"] if k.startswith(FROZEN)]:
            del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        for k, v in self.state_dict().items():
            if k.startswith(FROZEN):
                checkpoint["state_dict"].setdefault(k, v)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode_speech(self, x):
        """Posterior of a speech mel, in the latent space of the (singing) VAE."""
        h = self.speech_encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def pred_duration(self, ph, notedurs, pitches, notetypes, padding_mask=None):
        ph_emb = self.ph_encoder(ph)
        note_features = self.note_encoder(pitches, notedurs, notetypes)
        combined = ph_emb + note_features
        pred_durs = self.dur_predictor(combined, x_padding=padding_mask)
        return pred_durs

    # ------------------------------------------------------------------
    # The three embeddings, each [B, embed_dim]
    # ------------------------------------------------------------------
    def pool_latent(self, h):
        """Encoder output [B, 2*z_channels, T] -> time mean of the latent mean, which is what stage 2 reads."""
        return self.quant_conv(h)[:, :self.embed_dim].mean(2)

    @torch.no_grad()
    def embed_singing(self, x):
        return self.pool_latent(self.encoder(x))

    def embed_speech(self, x):
        return self.pool_latent(self.speech_encoder(x))

    @torch.no_grad()
    def encode_text(self, captions):
        """T5 token features [B, L, txt_dim] and padding mask [B, L] (True at padding), cached per caption."""
        new = [c for c in dict.fromkeys(captions) if c not in self._text_cache]
        if new:
            if not self._text_encoder:
                from transformers import T5Tokenizer, T5EncoderModel
                self._text_encoder += [T5Tokenizer.from_pretrained(self.text_encoder_path),
                                       T5EncoderModel.from_pretrained(self.text_encoder_path).eval().requires_grad_(False)]
            tokenizer, t5 = self._text_encoder
            t5.to(self.device)
            enc = tokenizer(new, truncation=True, max_length=self.text_max_length, padding="max_length",
                            return_tensors="pt").to(self.device)
            out = t5(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
            for c, feat, mask in zip(new, out, enc.attention_mask):
                self._text_cache[c] = (feat, mask == 0)
        feats, pads = zip(*(self._text_cache[c] for c in captions))
        return torch.stack(feats).to(self.device), torch.stack(pads).to(self.device)

    def embed_content(self, ph, midi, captions):
        """Content (frame-level phonemes and notes) combined with the text prompt by cross-attention."""
        h_ph   = self.ph_embed(ph).transpose(1, 2)          # [B, H, T]
        h_midi = self.midi_embed(midi).transpose(1, 2)      # [B, H, T]
        h_content = self.ph_conv(h_ph) + self.midi_conv(h_midi)

        txt_feat, txt_pad = self.encode_text(captions)      # [B, L, txt_dim]
        txt_feat = self.txt_proj(txt_feat)                  # [B, L, H]
        q = h_content.permute(0, 2, 1)                      # [B, T, H]
        cross_out, _ = self.cross_attn(q, txt_feat, txt_feat, key_padding_mask=txt_pad)
        return self.content_proj((q + cross_out).mean(1))

    # ------------------------------------------------------------------
    # Contrastive loss
    # ------------------------------------------------------------------
    def info_nce(self, a, b, exclude=None, extra_b=None):
        """Symmetric InfoNCE between two sets of embeddings [B, C]; row i of a and row i of b are the positive pair.

        exclude : [B, B] bool, pairs that are neither positives nor negatives (left out of the softmax)
        extra_b : [B, N, C], further negatives for the b -> a direction: row i of b against its own N embeddings
        Returns the loss (mean of both directions) and the top-1 accuracy of the b -> a direction.
        """
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        sim = a @ b.t() / self.contrastive_temp             # [B, B], a in rows
        if exclude is not None:
            sim = sim.masked_fill(exclude, float('-inf'))
        labels = torch.arange(sim.size(0), device=sim.device)
        b_to_a = sim.t()
        if extra_b is not None:
            neg = torch.einsum('bc,bnc->bn', b, F.normalize(extra_b, dim=2)) / self.contrastive_temp
            b_to_a = torch.cat([b_to_a, neg], dim=1)        # [B, B + N]
        loss = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(b_to_a, labels))
        acc = (b_to_a.argmax(1) == labels).float().mean()
        return loss, acc

    def contrastive_losses(self, batch):
        x = batch[self.image_key].float()                   # singing mel [B, C, T]
        x_speech = batch['spec_speech'].float()
        pair_id = batch['pair_id']

        z_sing = self.embed_singing(x)                      # frozen encoder, the anchor
        z_speech = self.embed_speech(x_speech)              # trainable
        z_content = self.embed_content(batch['ph_array'].long(), batch['pitch_array'].long(), batch['caption'])

        z_neg = None
        if batch['neg_singing'].shape[1] > 0:
            neg = batch['neg_singing'].float()              # [B, N, C, T], other singing clips
            B, N = neg.shape[:2]
            z_neg = self.embed_singing(neg.flatten(0, 1)).view(B, N, -1)

        # Singing clips of the same phrase (control and technique versions) share their speech clip: they
        # are positives of each other, so they are left out of the terms that involve speech.
        siblings = (pair_id[:, None] == pair_id[None, :]) & ~torch.eye(len(pair_id), dtype=torch.bool, device=x.device)

        loss_ss, acc_ss = self.info_nce(z_sing, z_speech, exclude=siblings, extra_b=z_neg)
        loss_cs, acc_cs = self.info_nce(z_content, z_speech, exclude=siblings)
        loss_sc, acc_sc = self.info_nce(z_sing, z_content)
        loss = (loss_ss + loss_cs + loss_sc) / 3
        return loss, dict(loss_ss=loss_ss, loss_cs=loss_cs, loss_sc=loss_sc, acc_ss=acc_ss, acc_cs=acc_cs, acc_sc=acc_sc)

    def training_step(self, batch, batch_idx):
        loss, parts = self.contrastive_losses(batch)
        self.log("loss_contrastive", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict({f"train/{k}": v for k, v in parts.items()}, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, parts = self.contrastive_losses(batch)
        bs = len(batch['pair_id'])
        self.log("val/loss_contrastive", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log_dict({f"val/{k}": v for k, v in parts.items()}, on_epoch=True, batch_size=bs)
        return loss

    def configure_optimizers(self):
        params = [p for name, module in self.named_children() if not (name + '.').startswith(FROZEN)
                  for p in module.parameters()]
        print(f"Custom audio encoder: {sum(p.numel() for p in params) / 1e6:.1f} M trainable parameters, learning rate "
              f"{self.lr:.2e} from model.params.lr (the learning rate main.py reports is not used)")
        opt = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        sched = LambdaLR(opt, lambda step: min(1.0, (step + 1) / max(1, self.warmup_steps)))
        return [opt], [dict(scheduler=sched, interval='step', frequency=1)]


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class ResnetBlock1D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,kernel_size = 3):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size//2)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=kernel_size//2)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=kernel_size,
                                                     stride=1,
                                                     padding=kernel_size//2)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,t,c = q.shape
        q = q.permute(0,2,1)   # b,t,c   
        w_ = torch.bmm(q,k)     # b,t,t   w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # if still 2d attn (q:b,hw,c ,k:b,c,hw -> w_:b,hw,hw)
        w_ = w_ * (int(t)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        w_ = w_.permute(0,2,1)   # b,t,t (first t of k, second of q)
        h_ = torch.bmm(v,w_)     # b,c,t (t of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]

        h_ = self.proj_out(h_)

        return x+h_

class Upsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest") # support 3D tensor(B,C,T)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x

class Encoder1D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_layers = [],down_layers = [], dropout=0.0, resamp_with_conv=True, in_channels,
                 z_channels, double_z=True,kernel_size=3, **ignore_kwargs):
        """ out_ch is only used in decoder,not used here
        """
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        print(f"downsample rates is {2**len(down_layers)}")
        self.down_layers = down_layers
        self.attn_layers = attn_layers
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=kernel_size//2)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        # downsampling
        self.down = nn.ModuleList()
        for i_level in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         kernel_size=kernel_size))
                block_in = block_out
                if i_level in attn_layers:
                    print(f"add attn in encoder layer:{i_level}")
                    attn.append(AttnBlock1D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in down_layers:
                print(f"add downsample in layer:{i_level}")
                down.downsample = Downsample1D(block_in, resamp_with_conv)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       kernel_size=kernel_size)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       kernel_size=kernel_size)

        # end
        self.norm_out = Normalize(block_in)# GroupNorm
        self.conv_out = torch.nn.Conv1d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=kernel_size//2)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks + 1):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level in self.down_layers:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
    
class Decoder1D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_layers = [],down_layers = [], dropout=0.0,kernel_size=3, resamp_with_conv=True, in_channels,
                z_channels, give_pre_end=False, tanh_out=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_layers = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        # self.down_layers = [i+1 for i in down_layers] # each downlayer add one
        self.down_layers = down_layers
        print(f"upsample rates is {2**len(down_layers)}")
        
        # 计算解码器中的上采样层索引
        self.up_layers = [self.num_layers - 1 - i for i in self.down_layers]
        print(f"Upsample will be performed at layers: {self.up_layers}")
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_layers-1]


        # z to block_in
        self.conv_in = torch.nn.Conv1d(z_channels,
                                       block_in,
                                       kernel_size=kernel_size,
                                       stride=1,
                                       padding=kernel_size//2)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock1D(block_in)
        self.mid.block_2 = ResnetBlock1D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_layers)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock1D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if i_level in attn_layers:
                    print(f"add attn in decoder layer:{i_level}")
                    attn.append(AttnBlock1D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in self.up_layers:
                print(f"add upsample in layer:{i_level}")
                up.upsample = Upsample1D(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=kernel_size//2)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_layers)):
            # 处理残差块和注意力机制
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            # 在正确的层执行上采样
            if i_level in self.up_layers and hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h