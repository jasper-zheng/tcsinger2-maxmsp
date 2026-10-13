"""
Modified for contrastive learning: 
- Added speech encoder with same architecture as main encoder
- Frozen original autoencoder parameters
- Added contrastive loss between singing and speech features
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from packaging import version
import numpy as np
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from torch.optim.lr_scheduler import LambdaLR
from ldm.util import instantiate_from_config
from utils.commons.rel_transformer import RelTransformerEncoder


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim,
                 ddconfig,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 monitor=None,
                 contrastive_temp=0.07,  # Temperature for contrastive loss
                 **kwargs,
                 ):
        super().__init__()
        self.image_key = image_key
        self.contrastive_temp = contrastive_temp
        
        # Original autoencoder components
        self.encoder = Encoder1D(**ddconfig)
        self.decoder = Decoder1D(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv1d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        hidden=ddconfig['hidden_size']
        self.ph_embed   = nn.Embedding(476, hidden)
        self.midi_embed = nn.Embedding(100, hidden)
        self.ph_conv   = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.midi_conv = nn.Conv1d(hidden, hidden, 3, padding=1)
        
        # Add speech encoder (same architecture as main encoder)
        self.speech_encoder = Encoder1D(**ddconfig)
        
        # Store learning rate
        self.learning_rate = kwargs.get('learning_rate', 1e-4)

        self.dur_predictor = DurationPredictor(
            ddconfig['hidden_size'],
            n_chans=ddconfig['hidden_size'],
            n_layers=ddconfig['dur_predictor_layers'],
            dropout_rate=ddconfig['predictor_dropout'],
            kernel_size=ddconfig['dur_predictor_kernel'])
        
        self.note_encoder = NoteEncoder(n_vocab=100, hidden_channels=ddconfig['hidden_size'])
        self.ph_encoder = RelTransformerEncoder(
                476, ddconfig['hidden_size'], ddconfig['hidden_size'],
                ddconfig['hidden_size']*4, ddconfig['num_heads'], ddconfig['enc_layers'],
                ddconfig['enc_ffn_kernel_size'], ddconfig['dropout'], prenet=ddconfig['enc_prenet'], pre_ln=ddconfig['enc_pre_ln'])

        # Freeze original autoencoder parameters AFTER loading checkpoint
        self.freeze_original_ae()
        
        # ---------- 文本编码器（冻结） ----------
        txt_cfg = {
            "target": "ldm.modules.encoders.modules.FrozenTextVocalEmbedder",
            "params": {"version": "useful_ckpts/flan-t5-large",
                       "max_length": 80}
        }
        self.text_encoder = instantiate_from_config(txt_cfg)
        self.text_encoder.eval().requires_grad_(False)

        # 输出维度 → hidden 的线性映射
        txt_dim = self.text_encoder.embed_dim   
        self.txt_proj = nn.Linear(txt_dim, hidden, bias=False)

        # Cross-Attention：query=h_content(seq=T) ; key/value = text(seq=L)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=8, batch_first=True)
        
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            
    def freeze_original_ae(self):
        """Freeze original autoencoder components and set to eval mode"""
        # Freeze parameters
        for param in list(self.encoder.parameters()) + \
                    list(self.decoder.parameters()) + \
                    list(self.quant_conv.parameters()) + \
                    list(self.dur_predictor.parameters()) + \
                    list(self.note_encoder.parameters()) + \
                    list(self.ph_encoder.parameters()) + \
                    list(self.post_quant_conv.parameters()):
            param.requires_grad = False
            
        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()
        self.dur_predictor.eval()
        self.note_encoder.eval()
        self.ph_encoder.eval()

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Load checkpoint, handling both original and full checkpoints"""
        sd = torch.load(path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state_dict" in sd:
            state_dict = sd["state_dict"]
        else:
            state_dict = sd
            
        # Filter keys and handle missing speech_encoder
        keys = list(state_dict.keys())
        new_state_dict = {}
        
        for k in keys:
            # Skip ignored keys
            skip = False
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    skip = True
                    break
            if skip:
                continue
                
            # Handle keys for speech_encoder if missing in checkpoint
            if k.startswith("speech_encoder.") and k not in self.state_dict():
                print(f"Key {k} not found in model, skipping")
                continue
                
            new_state_dict[k] = state_dict[k]
        
        # Load state dict strictly for matching keys
        self.load_state_dict(new_state_dict, strict=False)
        print(f"AutoencoderKL Restored from {path} Done")
        
        # Re-freeze after loading (in case loading changed requires_grad)
        self.freeze_original_ae()

    def encode(self, x):
        h = self.encoder(x)
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

    def get_input(self, batch, k):
        x = batch[k]
        assert len(x.shape) == 3
        x = x.to(memory_format=torch.contiguous_format).float()
        f0 = batch['f0']
        f0 = f0.to(memory_format=torch.contiguous_format).float()
        return [x, f0]

    def pred_duration(self, ph, notedurs, pitches, notetypes, padding_mask=None):
        # 获取音素嵌入
        ph_emb = self.ph_encoder(ph) 
        
        # 获取音符特征
        note_features = self.note_encoder(pitches, notedurs, notetypes)
        
        # 合并特征
        combined = ph_emb + note_features
        
        # 预测duration (B, T)
        pred_durs = self.dur_predictor(combined, x_padding=padding_mask)
        
        return pred_durs

    def get_negative_loss(self, h_speech, h_neg_pool):
        """
        h_speech : [B, H, T]  – positive anchor
        h_neg_pool : [B, N, H]  – pre-pooled negative features
        """
        if h_neg_pool is None:
            return 0.0

        B, N, H = h_neg_pool.shape
        speech_pool = F.normalize(h_speech.mean(2), dim=1)     # [B, H]
        neg_flat    = F.normalize(h_neg_pool.reshape(B*N, H), dim=1)  # [B*N, H]

        sim = torch.matmul(speech_pool, neg_flat.t()) / self.contrastive_temp  # [B, B*N]
        labels = torch.full((B,), -1, device=sim.device, dtype=torch.long)     # 没正样本 → label -1
        return F.cross_entropy(sim, labels, ignore_index=-1)

    def get_contrastive_loss(self, h_sing, h_speech):
        """Compute contrastive loss between singing and speech features"""
        # Pool features along time dimension
        h_sing_pool = h_sing.mean(dim=2)  # [B, C]
        h_speech_pool = h_speech.mean(dim=2)  # [B, C]
        
        # Normalize features
        h_sing_pool = F.normalize(h_sing_pool, p=2, dim=1)
        h_speech_pool = F.normalize(h_speech_pool, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(h_sing_pool, h_speech_pool.t())  # [B, B]
        sim_matrix /= self.contrastive_temp
        
        # Create labels (diagonal is positive)
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        
        # Calculate contrastive loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        x = inputs[0]
        
        # Get speech input
        x_speech = batch['spec_speech']
        assert x_speech.shape[1:] == x.shape[1:], \
            f"Speech shape {x_speech.shape} != Singing shape {x.shape}"
        
        ph  = batch['ph_array']     # [B, T]
        midi= batch['pitch_array']  # [B, T]
        h_ph   = self.ph_embed(ph).transpose(1, 2)          # [B, H, T]
        h_midi = self.midi_embed(midi).transpose(1, 2)      # [B, H, T] 
        h_ph_conv   = self.ph_conv(h_ph)
        h_midi_conv = self.midi_conv(h_midi)
        h_content   = h_ph_conv + h_midi_conv      # [B, H, T]

        # ---------- 文本 → cross-attention ----------
        captions = batch['caption']                # List[str] 长度 B
        with torch.no_grad():
            txt_feat = self.text_encoder(captions) # [B, L, txt_dim]
        txt_feat   = self.txt_proj(txt_feat)       # [B, L, H]

        q = h_content.permute(0, 2, 1)             # [B, T, H]
        cross_out, _ = self.cross_attn(q, txt_feat, txt_feat,
                                       key_padding_mask=None)  # [B,T,H]
        cross_out = cross_out.permute(0, 2, 1)     # [B, H, T]

        h_content = h_content + cross_out       
        
        # Original autoencoder processing (frozen)
        with torch.no_grad():
            reconstructions, posterior = self(x)

        # Get features from both encoders
        with torch.no_grad():
            h_sing = self.encoder(x)  # Frozen encoder
        h_speech = self.speech_encoder(x_speech)  # Trainable speech encoder

        h_neg_pool = None
        if 'neg_singing' in batch:
            neg = batch['neg_singing']            # [B, N_neg, C, T]
            B, N, C, T = neg.shape
            neg = neg.view(B*N, C, T)
            with torch.no_grad():
                h_neg = self.encoder(neg)         # [B*N, H, T] (encoder 已冻结)
            h_neg_pool = h_neg.mean(2)            # pool-T → [B*N, H]
            h_neg_pool = h_neg_pool.view(B, N, -1)  # [B, N, H]

        # Calculate contrastive loss
        loss_ss  = self.get_contrastive_loss(h_sing,     h_speech)
        loss_cs  = self.get_contrastive_loss(h_content,  h_speech)
        loss_neg = self.get_negative_loss(h_speech, h_neg_pool)
        contrastive_loss = loss_ss + loss_cs + loss_neg
        self.log("loss_contrastive", contrastive_loss, prog_bar=True, 
                logger=True, on_step=True, on_epoch=True)
        
        return contrastive_loss

    def validation_step(self, batch, batch_idx):
        # -------- 1. 取输入 --------
        x, _ = self.get_input(batch, self.image_key)           # singing spec
        x_speech = batch['spec_speech']                        # speech spec
        ph   = batch['ph_array']
        midi = batch['pitch_array']
        captions = batch['caption']                            # List[str]

        # -------- 2. 计算 h_content --------
        h_ph   = self.ph_embed(ph).transpose(1, 2)
        h_midi = self.midi_embed(midi).transpose(1, 2)
        h_content = self.ph_conv(h_ph) + self.midi_conv(h_midi)

        with torch.no_grad():
            txt_feat = self.text_encoder(captions)             # [B, L, txt_dim]
        txt_feat = self.txt_proj(txt_feat)                     # [B, L, H]

        q = h_content.permute(0, 2, 1)                         # [B, T, H]
        cross_out, _ = self.cross_attn(q, txt_feat, txt_feat)
        h_content = h_content + cross_out.permute(0, 2, 1)

        # -------- 3. 正/负特征 --------
        with torch.no_grad():
            h_sing = self.encoder(x)                           # 冻住
        h_speech = self.speech_encoder(x_speech)               # 可训练

        h_neg_pool = None
        if 'neg_singing' in batch:
            neg = batch['neg_singing']                         # [B, N, C, T]
            B, N, C, T = neg.shape
            neg = neg.view(B * N, C, T)
            with torch.no_grad():
                h_neg = self.encoder(neg)                      # [B*N, H, T]
            h_neg_pool = h_neg.mean(2).view(B, N, -1)          # [B, N, H]

        # -------- 4. 计算损失 --------
        loss_ss  = self.get_contrastive_loss(h_sing, h_speech)
        loss_cs  = self.get_contrastive_loss(h_content, h_speech)
        loss_neg = self.get_negative_loss(h_speech, h_neg_pool)
        val_loss = loss_ss + loss_cs + loss_neg

        self.log("val/loss_contrastive", val_loss, prog_bar=True, on_epoch=True)

        return val_loss


    def test_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        x = inputs[0]
        
        with torch.no_grad():
            reconstructions, posterior = self(x)
            mse_loss = F.mse_loss(reconstructions, x)
            self.log('test/mse_loss', mse_loss)
            
            test_ckpt_path = os.path.basename(self.trainer.tested_ckpt_path)
            savedir = os.path.join(self.trainer.log_dir, f'output_imgs_{test_ckpt_path}', 'fake_class')
            
            if batch_idx == 0:
                os.makedirs(savedir, exist_ok=True)
                print(f"Test outputs saved to: {savedir}")
            
            file_names = batch['f_name']
            reconstructions = reconstructions.cpu().numpy()
            
            for b in range(reconstructions.shape[0]):
                vname_num_split_index = file_names[b].rfind('_')
                v_n = file_names[b][:vname_num_split_index]
                num = file_names[b][vname_num_split_index+1:]
                save_img_path = os.path.join(savedir, f'{v_n}.npy')
                np.save(save_img_path, reconstructions[b])
        
        return None
        
    def configure_optimizers(self):
        params = list(self.speech_encoder.parameters()) + \
                 list(self.ph_conv.parameters()) + \
                 list(self.midi_conv.parameters()) + \
                 list(self.txt_proj.parameters()) + \
                 list(self.cross_attn.parameters())
        return torch.optim.Adam(params, lr=self.learning_rate)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        # Original implementation remains unchanged
        log = dict()
        x = self.get_input(batch, self.image_key)
        f0 = x[1]
        x = x[0]
        mel_gt = x
        x = x.to(self.device)
        f0 = f0.to(self.device)

        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample())).unsqueeze(1)
            mel_rec = xrec
            log["reconstructions"] = mel_rec.unsqueeze(1)
            log["f0"] = f0.cpu()
            
        log["inputs"] = mel_gt.unsqueeze(1)
        log['f0_gt'] = f0.cpu()

        return log


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