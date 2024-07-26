import torch.nn as nn
import torch

from .helper import PositionwiseFFN
from .attention import MultiHeadAttention


class GPTDecoderBlock(nn.Module):
    # Implemented after
    # https://de.wikipedia.org/wiki/Generativer_vortrainierter_Transformer#/media/Datei:Full_GPT_architecture.png

    def __init__(
        self,
        in_size,
        attn_out,
        attn_heads,
        ffn_out,
        ffn_hidden,
        dropout,
        activation=nn.GELU(),
        alibi=False,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            attn_out, attn_heads, dropout, use_alibi=alibi
        )
        self.norm = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(attn_out)
        self.pos_ffn = PositionwiseFFN(
            attn_out, ffn_hidden, ffn_out, dropout, activation
        )

    def forward(self, x, tgt_mask=None):
        norm_out = self.norm(x)
        attn_out = self.self_attention(norm_out, norm_out, norm_out, tgt_mask)
        add_out = attn_out + x
        norm2_out = self.norm2(add_out)
        ffn_out = self.pos_ffn(norm2_out)
        return add_out + ffn_out
