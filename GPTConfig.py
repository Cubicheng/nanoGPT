from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # 最大的 Sequence Length
    vocab_size: int = 50257  # token 的种类数
    n_layer: int = 12  # 注意力层数
    n_head: int = 12  # 注意力头数
    n_embd: int = 768  # embedding 的维度
