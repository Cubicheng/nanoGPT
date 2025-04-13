import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 输入是 n_embd 维的向量，输出 Query,Key,Value 三个向量
        # 但是这里 Query,Key,Value 三个向量是“多头”的，拆分之后才是“单头”的 Query,Key,Value 向量
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # transformer 原始论文中的 WO
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 注册一个下三角因果掩码（防止看到未来信息，即后面 token 的 value 不会对当前 token 的输出有贡献）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        # Batch Size（同时处理的批次数量）, Sequence Length（每个批次的 token 数量）, Embedding Dim（每个token的嵌入向量维度）
        # x: (B,T,C)
        B, T, C = x.size()
        # 根据 x 计算 “多头”格式的 Query,Key,Value 三个向量
        # qkv: (B,T,3*C)
        qkv = self.c_attn(x)
        # 在第 dim 个维度上按照self.n_embd的大小均等分割
        # q,k,v: (B,T,C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # '//'的意思是整数除法，比如 5//2 = 2
        # transpose(1,2) 是交换矩阵的维度，比如 (B,T,C) -> (B,C,T)，这里用于优化计算速度
        # 这里将“多头”格式的 Query,Key,Value 拆分成了“单头”格式的 Query,Key,Value
        # q,k,v: (B,n_head,T,C//n_head)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 这里就是 "Attention is all you need" 中的经典式子
        # q 乘以 k的转置，然后再除以 Query-Key 空间的维度的根号
        # 这里 @ 乘法只对最后两个维度做乘法
        # 这里 transpose(-2,-1) 表示交换最后两个维度
        # size(-1) 表示最后一个维度的大小
        # att: (B,n_head,T,T)
        # att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        # 覆盖上一个下三角的 -inf，防止后面的 token 影响前面的 token
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # softmax 转成概率
        # att = F.softmax(att, dim=-1)

        # att 现在就是 每一个 token 与其他 token 的注意力权重
        # 这里 @ 乘法只对最后两个维度做乘法
        # y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 将单头的 value 向量拼接成完整的 value（就是直接 concat，可以参见 transformer 原论文）
        # transpose(1,2) 把上面交换过的维度交换回来
        # contiguous() 确保张量内存连续
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # transformer 原始论文中的 WO
        y = self.c_proj(y)
        return y
