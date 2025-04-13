import torch
from torch.nn import functional as F
from GPTConfig import GPTConfig
from GPT import GPT
import tiktoken

torch.serialization.add_safe_globals([GPTConfig])


def sample_model():
    num_return_sequences = 5
    max_length = 30
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("Hello, I'm a lauguage model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# 加载已训练模型
model = GPT(GPTConfig(vocab_size=50304)).to(device)
checkpoint = torch.load("log/model_00100.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
# 切换评估模式
model.eval()
# 执行生成逻辑
with torch.no_grad():
    sample_model()  # 使用代码中现有的生成函数
