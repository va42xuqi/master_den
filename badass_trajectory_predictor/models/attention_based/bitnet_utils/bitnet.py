import torch.nn as nn

from .building_blocks import GPTDecoderBlock


class GPTLike(nn.Module):
    def __init__(
        self,
        n_blocks=6,
        input_dim=512,
        ffn_hidden=2048,
        n_heads=8,
        dropout=0.2,
        pos_encoding=nn.Identity(),
        generator=nn.Identity(),
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pos_encoding = pos_encoding

        self.generator = generator
        self.norm = nn.LayerNorm(input_dim)

        for _ in range(n_blocks):
            self.blocks.append(
                GPTDecoderBlock(
                    in_size=input_dim,
                    attn_out=input_dim,
                    attn_heads=n_heads,
                    ffn_out=input_dim,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                )
            )

    def forward(self, x, tgt_mask=None, **pos_encoding_args):
        x = self.pos_encoding(x, **pos_encoding_args)
        for blk in self.blocks:
            x = blk(x, tgt_mask)
        x = self.norm(x)
        x = self.generator(x)
        return x
