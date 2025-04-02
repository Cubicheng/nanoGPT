import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 升高维度，强化模型能力
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # 激活函数层，GELU 是 RELU 的平滑版本（并且 GELU 在负半轴不是完全平的，这个性质很好），这里的 "tanh" 是一个近似计算，更快
        self.relu = nn.GELU(approximate="tanh")
        # 降维回去
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x
