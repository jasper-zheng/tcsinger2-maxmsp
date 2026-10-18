"""The caption encoder for export (EWP1): flan-t5-large's encoder on the 80 token ids of neural.tokenizer,
without an attention mask, as in training (FrozenTextVocalEmbedder)."""
from torch import nn


class CaptionEncoder(nn.Module):
    def __init__(self, t5):
        super().__init__()
        self.t5 = t5

    def forward(self, input_ids):
        """input_ids [B, 80] int64 -> last hidden states [B, 80, 1024]."""
        return self.t5(input_ids=input_ids).last_hidden_state
