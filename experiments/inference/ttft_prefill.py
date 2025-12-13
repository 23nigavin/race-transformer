import time
import tiktoken
import torch
from config import GPT_CONFIG_124M
from models.gpt_transformer import GPTModel
from models.gpt_race_transformer import RACEModel
from models.llama_race_transformer import LlamaRACEModel
from models.llama_transformer import LlamaModel
torch.set_num_threads(8)

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


####################################################
# NEW
def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
####################################################

def _pick_single_token_id():
    CANDIDATES = [" hello", "!", " the", " a", " ?"]
    tokenizer = tiktoken.get_encoding("gpt2")
    for s in CANDIDATES:
        ids = tokenizer.encode(s)
        if len(ids) == 1:
            return tokenizer, ids[0]
    return tokenizer, 0  # fallback


def _time_ttft_plus_prefill(model, encoded_tensor, ctx_len, use_cache: bool):
    # Measure from just before prefill to right after the first token is produced
    if hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()

    t0 = time.perf_counter()
    _ = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,   # shape [1, L-1]
        max_new_tokens=1,     # exactly one token
        context_size=ctx_len,
        use_cache=use_cache
    )
    return time.perf_counter() - t0

def run_inference_ttft_prefill_bench():
    torch.manual_seed(123)
    # CPU-only as requested; leave this as-is if you want to stay on CPU.
    device = torch.device("cpu")

    # tokenizer & a single-token id
    tokenizer, tok_id = _pick_single_token_id()
    powers = list(range(6, 14))  # 2^6 .. 2^14  change back to 15

    print("\n" + "="*98)
    print("TTFT + Prefill (seconds) — GPT & RACE (cache, CPU)".center(98))
    print("="*98)
    print(f"{'Ctx':>8} | {'GPT cache':>10} | {'RACE cache':>11} | {'LLaMA':>11} | {'LLaMA RACE':>12}")
    print("-"*98)

    for p in powers:
        L = 1 << p

        # Prompt of length L-1 (single repeated token)
        encoded = torch.full((1, L-1), tok_id, dtype=torch.long, device=device)

        # Build per-L so pos-emb & masks match exactly
        cfgL = {**GPT_CONFIG_124M, "context_length": L}

        gpt = GPTModel(cfgL).to(device).eval()
        race = RACEModel(cfgL).to(device).eval()
        llama = LlamaModel(cfgL).to(device).eval()
        llama_race = LlamaRACEModel(cfgL).to(device).eval()

        # GPT timings
        # gpt_no_cache = _time_ttft_plus_prefill(gpt,  encoded, L, use_cache=False)
        gpt_with_cache = _time_ttft_plus_prefill(gpt, encoded, L, use_cache=True)

        # RACE timings
        # race_no_cache = _time_ttft_plus_prefill(race,  encoded, L, use_cache=False)
        race_with_cache = _time_ttft_plus_prefill(race, encoded, L, use_cache=True)

        llama_with_cache = _time_ttft_plus_prefill(llama, encoded, L, use_cache=True)

        llama_race_with_cache = _time_ttft_plus_prefill(llama_race, encoded, L, use_cache=True)

        print(f"{L:>8}| {gpt_with_cache:>10.6f} | {race_with_cache:>11.6f} | {llama_with_cache:>11.6f} | {llama_race_with_cache:>12.6f}")

    print("-"*98)
    print("Notes: 'cache' does prefill (prompt) + 1 decode step using KV; 'nocache' runs a single full forward of length L.")

if __name__ == "__main__":
    run_inference_ttft_prefill_bench()
