import torch
from torch.nn import functional as F
from GPTConfig import GPTConfig
from GPT import GPT
import tiktoken
from hellaswag import render_example, iterate_examples

torch.serialization.add_safe_globals([GPTConfig])


def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (
        mask[..., 1:]
    ).contiguous()  # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def hella_model():
    num_correct_norm = 0
    num_total = 0
    for i, example in enumerate(iterate_examples("val")):
        # only process examples where i % ddp_world_size == ddp_rank
        # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        # get the logits
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
    # reduce the stats across all processes
    acc_norm = num_correct_norm / num_total
    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")


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
checkpoint = torch.load("model/model_02000.pt", map_location=device, weights_only=True)
new_state_dict = {
    k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
}
model.load_state_dict(new_state_dict)
# 切换评估模式
model.eval()
# 执行生成逻辑
with torch.no_grad():
    sample_model()  # 使用代码中现有的生成函数
    hella_model()
