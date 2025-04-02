import torch.nn as nn
from CausalSelfAttention import CausalSelfAttention
from MLP import MLP


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 注意力层，token 们相互影响意思
        x = x + self.mlp(
            self.ln_2(x)
        )  # MLP层，深化 token 本身的意思，并且加上注意力层计算的来自其他 token 的影响
        return x
