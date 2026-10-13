# TCSinger 2 main model
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
try:
    from flash_attn import flash_attn_func
    is_flash_attn = True
except:
    is_flash_attn = False
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from einops import rearrange
from ldm.modules.diffusionmodules.flag_large_dit_moe import Attention, FeedForward, RMSNorm, modulate, TimestepEmbedder,ConditionEmbedder

#############################################################################
#                               Core DiT Model                              #
#############################################################################

class Conv1DFinalLayer(nn.Module):
    """
    The final layer of CrossAttnDiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.GroupNorm(16,hidden_size)
        self.conv1d = nn.Conv1d(hidden_size, out_channels,kernel_size=1)

    def forward(self, x): # x:(B,C,T)
        x = self.norm_final(x)
        x = self.conv1d(x)
        return x

class MoE(nn.Module):
    LOAD_BALANCING_LOSSES = []

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        temperature: float = 2.0, 
    ):
        super().__init__()
        self.num_language_experts = num_experts  # 4个专家
        self.num_caption_experts = num_experts
        self.temperature = temperature

        # 第一组：语言专家（每个batch选1个）
        self.language_embedding = nn.Embedding(9, dim)  # 9种语言ID
        self.language_gating = nn.Linear(dim, self.num_language_experts)  
        self.language_experts = nn.ModuleDict({
            str(i): FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for i in range(self.num_language_experts)
        })

        # 第二组：caption专家（每个token选1个）
        self.cross_attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.caption_gating = nn.Linear(dim, self.num_caption_experts)
        self.caption_experts = nn.ModuleDict({
            str(i): FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for i in range(self.num_caption_experts)
        })

    def gumbel_softmax(self, logits, temperature, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def load_balancing_loss(self, expert_probs_list):
        # 计算所有专家的平均使用概率
        all_probs = torch.cat(expert_probs_list, dim=1)
        expert_usage = all_probs.mean(dim=0)
        
        # 计算负载均衡损失（均匀分布时损失最小）
        loss = (expert_usage * torch.log(expert_usage + 1e-10)).sum()
        return loss

    def forward(self, x, time, caption, language):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        # 获取语言ID的嵌入
        lang_emb = self.language_embedding(language)  # [B, C]
        lang_logits = self.language_gating(lang_emb)  # [B, 4] 输出4个专家类别
        lang_probs = self.gumbel_softmax(lang_logits, self.temperature, hard=not self.training)
        lang_probs = lang_probs.repeat_interleave(T, dim=0)  # [B*T, 4]

        # 计算语言专家输出
        lang_outputs = torch.stack([e(x_flat) for e in self.language_experts.values()], dim=1)  # [B*T, num_experts, 768]

        # 使用Gumbel softmax的输出进行加权求和，得到最终输出
        first_out = (lang_outputs * lang_probs.unsqueeze(-1)).sum(dim=1)  # [B*T, 768]

        # 第二组：caption专家处理
        # Cross-attention
        first_reshaped = first_out.view(B, T, C)
        cross_attn, _ = self.cross_attention(first_reshaped, caption, caption)
        cross_flat = cross_attn.reshape(-1, C)

        # 获取caption门控权重
        cap_logits = self.caption_gating(cross_flat)  # [B*T, num_cap]
        cap_probs = self.gumbel_softmax(cap_logits, self.temperature, hard=not self.training)

        # 计算caption专家输出
        cap_outputs = torch.stack([e(first_out) for e in self.caption_experts.values()], dim=1)
        second_out = (cap_outputs * cap_probs.unsqueeze(-1)).sum(dim=1)

        # 最终输出
        total_out = (first_out + second_out).view(B, T, C)

        # 负载均衡损失
        loss = self.load_balancing_loss([lang_probs, cap_probs])
        
        return total_out, loss

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool, y_dim: int,num_experts) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = MoE(
            dim=dim, hidden_dim= dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier, num_experts=num_experts,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),
        )
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        time=None,
        caption=None,
        language=None
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)

            h = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                x_mask,
                freqs_cis,
                self.attention_y_norm(y), y_mask,
            )
            out,loss = self.feed_forward(
                modulate(self.ffn_norm(h), shift_mlp, scale_mlp),time,caption,language
            )
            out=h + gate_mlp.unsqueeze(1) * out

        else:
            h = x + self.attention(
                self.attention_norm(x), x_mask, freqs_cis, self.attention_y_norm(y), y_mask
            )
            out,loss = self.feed_forward(self.ffn_norm(h),time,caption,language)
            out=h + out

        return out,loss
        # return out

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size, 2 * hidden_size, bias=True
            ),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Singer(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels,
        context_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_len = 1000,
        n_kv_heads=None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps=1e-5,
        qk_norm=None,
        rope_scaling_factor: float = 1.,
        ntk_factor: float = 1.,
        num_experts=4,
        ori_dim=1024
    ):
        super().__init__()
        self.in_channels = in_channels # vae dim
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 9
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ori_dim = ori_dim
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size//2)

        self.prompt_proj = nn.Conv1d(
            in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.midi_embedding = nn.Embedding(100, hidden_size)  # MIDI range is 0-100
        self.ph_embedding = nn.Embedding(476, hidden_size) 
        self.downsample_rate = 4 

        # Conv1d layer for f0 (continuous)
        self.f0_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.AvgPool1d(self.downsample_rate),    # vae downsample 2
        )
        
        self.pre_transformer = TransformerBlock(-1, hidden_size, num_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm, context_dim,num_experts=self.num_experts)

        self.f0_regressor = nn.Linear(hidden_size, 1)
        self.f0_upsample  = nn.Upsample(scale_factor=self.downsample_rate, mode='linear', align_corners=False) 
        self.uv_classifier = nn.Linear(hidden_size, 2)
        
        self.midi_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.AvgPool1d(self.downsample_rate),    # vae downsample 2
        )
        
        self.ph_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.AvgPool1d(self.downsample_rate),    # vae downsample 2
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(layer_id, hidden_size, num_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm, context_dim,num_experts=self.num_experts)
            for layer_id in range(depth)
        ])
        
        self.final_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

        self.freqs_cis = Singer.precompute_freqs_cis(hidden_size // num_heads, max_len,
                       rope_scaling_factor=rope_scaling_factor, ntk_factor=ntk_factor)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor

        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_size, bias=True),
        )

        self.gate_content = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.gate_f0 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.c_embedder = ConditionEmbedder(hidden_size, ori_dim)
        self.loss_w=1

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        context: dictionary containing different conditioning inputs.
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        # Extract context inputs
        f0_gt = context['c_concat']['f0']    # (B, 1, T)
        f0_gt_log = torch.log(f0_gt + 1.0)
        midi = context['c_concat']['midi']    # (B, 1, T)
        ph = context['c_concat']['ph']    # (B, 1, T)
        caption = context['c_crossattn']  # (B, T_caption, 1024)
        prompt = context['c_concat']['prompt_embed']  # (B, C, T_prompt)
        language = context['c_concat']['language']  # (B, 1)
        infer=context['infer']
        
        B = x.size(0)
        T_prompt   = prompt.size(2)
        device = x.device

        # Project prompt using Conv1d
        prompt = self.prompt_proj(prompt)  # [B, hidden_size, T_prompt]
        prompt = prompt.transpose(1, 2)    # [B, T_prompt, hidden_size]

        # Extract global vector from prompt using global average pooling
        prompt_global = prompt.mean(dim=1)  # [B, hidden_size]

        ph_pad_id   = torch.full((B, T_prompt*self.downsample_rate), 475, dtype=torch.long,device=device)
        midi_pad_id = torch.full((B, T_prompt*self.downsample_rate), 99,  dtype=torch.long,device=device)
        f0_pad      = torch.zeros(B, 1, T_prompt*self.downsample_rate,device=device)   # 无声记 0

        ph_pad_feat = self.ph_proj(
                        self.ph_embedding(ph_pad_id).transpose(1, 2)
                    ).transpose(1, 2)                 # [B, T_prompt, C]
        midi_pad_feat = self.midi_proj(
                        self.midi_embedding(midi_pad_id).transpose(1, 2)
                        ).transpose(1, 2)
        f0_pad_feat = self.f0_proj(f0_pad).transpose(1, 2)          # [B, T_prompt, C]

        # Project and process other conditioning inputs
        midi_feat = self.midi_proj(
                    self.midi_embedding(midi.squeeze(1)).transpose(1, 2)
                ).transpose(1, 2)
        ph_feat   = self.ph_proj(
                    self.ph_embedding(ph.squeeze(1)).transpose(1, 2)
                ).transpose(1, 2)

        midi = torch.cat([midi_pad_feat, midi_feat], dim=1)     # [B, T_total, C]
        ph   = torch.cat([ph_pad_feat,   ph_feat],   dim=1)

        # Combine midi and phoneme embeddings
        content = midi + ph
        content = self.final_proj(content.transpose(1, 2)).transpose(1, 2)  # [B, T, C]

        # Process input x
        x = self.proj_in(x).transpose(1, 2)  # [B, C, T] -> [B, T, C]

        # Concatenate prompt and x along the time dimension
        prompt_length = prompt.shape[1]
        x_combined = torch.cat([prompt, x], dim=1)  # [B, T_prompt + T, C]

        # Timestep embedding and caption embedding
        t_emb = self.t_embedder(t)  # [B, C]
        caption = self.c_embedder(caption)  # [B, T_caption, C]

        # Prepare adaptive layer normalization input
        cap_mask = torch.ones((caption.shape[0], caption.shape[1]), dtype=torch.int32, device=x.device)
        cap_mask_float = cap_mask.float().unsqueeze(-1)
        cap_feats_pool = (caption * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_feats_pool = cap_feats_pool.to(caption)
        cap_emb = self.cap_embedder(cap_feats_pool)

        # Combine timestep embedding, caption global features, and prompt global features
        adaln_input = t_emb + cap_emb + prompt_global  # Add the prompt global vector

        # Combine inputs: add content and f0 to x_combined
        gate = torch.sigmoid(self.gate_content(content))
        x_combined+=content*gate

        mask = torch.ones((x_combined.shape[0], x_combined.shape[1]), dtype=torch.int32, device=x.device)
        cap_mask = cap_mask.bool()
        loss=0

        x_combined, _ = self.pre_transformer(
            x_combined, mask, caption, cap_mask, self.freqs_cis[:x_combined.size(1)],
            adaln_input=adaln_input, time=t_emb, caption=caption, language=language,
        )

        feats = x_combined[:, prompt_length:, :]  # [B,T,C]
        f0_pred_lat = self.f0_regressor(feats).transpose(1,2)             # [B,1,T_lat]
        f0_pred     = self.f0_upsample(f0_pred_lat)                       # [B,1,T_orig]

        uv_logits_lat = self.uv_classifier(feats)               # [B,T_lat,2]
        uv_logits = uv_logits_lat.repeat_interleave(self.downsample_rate, dim=1)[:, :f0_gt.shape[-1], :]  # [B,T,2]
        uv_pred = torch.argmax(uv_logits, dim=-1, keepdim=True).float().permute(0,2,1)  # [B,1,T]

        if infer:
            # 推理：使用预测 f0，损失置 0
            f0_for_cond = f0_pred=f0_pred.detach() * uv_pred   
            f0_loss = 0.0
            uv_loss = 0.0
        else:
            # 训练：使用 GT f0，计算损失
            uv_mask = (f0_gt > 0).float()            # [B,1,T]
            f0_loss = F.mse_loss(f0_pred * uv_mask, f0_gt_log * uv_mask)
            uv_gt = (f0_gt.squeeze(1) > 0).long()    # [B,T]
            uv_loss = F.cross_entropy(uv_logits.view(-1, 2), uv_gt.view(-1))
            f0_for_cond = f0_gt_log
            f0_pred=f0_pred*uv_pred

        f0_feat = self.f0_proj(f0_for_cond).transpose(1, 2)  # [B,T,C]
        f0_full   = torch.cat([f0_pad_feat,   f0_feat],   dim=1)   
        gate = torch.sigmoid(self.gate_f0(f0_full))
        x_combined+= f0_full*gate

        # Apply transformer blocks
        for block in self.blocks:
            x_combined,loss_tmp = block(
                x_combined, mask, caption, cap_mask, self.freqs_cis[:x_combined.size(1)],
                adaln_input=adaln_input, time=t_emb, caption=caption, language=language
            )
            loss += loss_tmp
        loss = loss/len(self.blocks)
        
        if self.loss_w>0.01:
            self.loss_w*=0.9999
        loss *= self.loss_w #加系数防止影响过大
        
        loss+=f0_loss+uv_loss

        # Remove the prompt part and keep only the original x output
        x_final = x_combined[:, prompt_length:, :]  # 去掉 prompt 部分

        # Final processing
        x_final = self.final_layer(x_final, adaln_input)
        x_final = rearrange(x_final, 'b t c -> b c t')  # [B, C, T]

        return x_final, loss, torch.exp(f0_for_cond).detach() - 1.0

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        print(f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}")

        freqs = 1.0 / (theta ** (
            torch.arange(0, dim, 2)[: (dim // 2)].float().cuda() / dim
        ))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

class TCSinger2(Singer):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels,
        context_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_len = 1000,num_experts=4,ori_dim=1024
    ):
        super().__init__(in_channels, context_dim, hidden_size, depth, num_heads,max_len= max_len,num_experts=num_experts)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers and proj_in:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        print('-------------------------------- successfully init! --------------------------------')
