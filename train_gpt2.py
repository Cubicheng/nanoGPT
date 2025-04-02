import torch
from torch.nn import functional as F
from GPTConfig import GPTConfig
from GPT import GPT

# ----------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig())
model.eval()
model.to(device)

import tiktoken

# enc = tiktoken.get_encoding("gpt2")
# with open("input.txt", "r") as f:
#     text = f.read()
# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4, 32
# buf = torch.tensor(tokens[: B * T + 1])
# x = buf[:-1].view(B, T).to(device)
# y = buf[1:].view(B, T).to(device)

# logits, loss = model(x, y)
# print(loss)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a lauguage model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits, loss = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
