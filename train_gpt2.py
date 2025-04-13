import torch
from torch.nn import functional as F
from GPTConfig import GPTConfig
from GPT import GPT
from DataLoaderLite import DataLoaderLite
import tiktoken
import time
import math
import os

total_batch_size = 8192
B = 4
T = 256

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 101

val_interval_steps = 50
checkpoint_interval_steps = 100
# -----------------------------------


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def val_model(step):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
    print(f"validation loss:{val_loss_accum.item():.4f}")
    if step % checkpoint_interval_steps == 0:
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            "model": model.state_dict(),
            "config": model.config,
            "step": step,
            "val_loss": val_loss_accum.item(),
        }
        torch.save(checkpoint, checkpoint_path)


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


# ----------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(
    f"total_batch_size:{total_batch_size} | B:{B} | T:{T} | grad_accum_steps:{grad_accum_steps}"
)

# model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
# model = torch.compile(model)

train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

torch.set_float32_matmul_precision("high")

# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, lr=max_lr, device=device)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    if step % val_interval_steps == 0:
        val_model(step)
        sample_model()

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    # import code
    # code.interact(local=locals())

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    delta_time = t1 - t0
    tokens_per_sec = train_loader.B * train_loader.T * grad_accum_steps / delta_time
    print(
        f"step:{step} | loss:{loss_accum.item():.3f} | delta_time:{delta_time:.5f}ms | norm:{norm:.4f} | lr:{lr:4e} | tokens_per_sec:{tokens_per_sec:.5f}"
    )
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_accum.item():.6f}\n")
