"""Batched temperature sampling with a KV cache, written directly on TalkieForCausalLM._forward_hidden.

Why not model.generate: Talkie's prepare_inputs_for_generation drops logits_to_keep, so generate() builds
full-sequence FP32 logits at prefill (~8 GB for 16 x 2000 tokens). Here only the last position is projected,
through an FP32 copy of the output head (as in talkie-evals' scorer).
"""
import time

import torch
from transformers.cache_utils import DynamicCache

from ..formats import truncate

KV_BYTES_PER_TOKEN = 40 * 2 * 5120 * 2   # layers x (k, v) x width x bf16


def plan_batches(lengths, max_new_tokens, max_batch_tokens, max_batch):
    """Group row indices (longest first) so rows x (longest prompt + new tokens) stays under budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, current = [], []
    for i in order:
        width = lengths[current[0]] if current else lengths[i]
        if current and (len(current) >= max_batch or (len(current) + 1) * (width + max_new_tokens) > max_batch_tokens):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def _generate(model, head_t, prompts_ids, max_new_tokens, temperature, stop_ids, pad_id, generator, top_p=1.0):
    device = head_t.device
    width = max(len(x) for x in prompts_ids)
    batch = len(prompts_ids)
    ids = torch.full((batch, width), pad_id, dtype=torch.long)
    mask = torch.zeros((batch, width), dtype=torch.bool)
    for row, x in enumerate(prompts_ids):
        ids[row, width - len(x):] = torch.tensor(x)
        mask[row, width - len(x):] = True
    ids, mask = ids.to(device), mask.to(device)
    cache = DynamicCache(config=model.config)
    hidden, cache, _ = model._forward_hidden(ids, None, mask, None, cache, True)
    last = hidden[:, -1]
    stops = torch.tensor(stop_ids, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    out = torch.full((batch, max_new_tokens), pad_id, dtype=torch.long, device=device)
    steps = 0
    for step in range(max_new_tokens):
        logits = last.float() @ head_t
        if temperature == 0:
            nxt = logits.argmax(-1)
        else:
            probs = torch.softmax(logits / temperature, -1)
            if top_p < 1.0:  # nucleus: keep the smallest set of tokens whose mass reaches top_p
                sorted_p, idx = probs.sort(-1, descending=True)
                drop = sorted_p.cumsum(-1) - sorted_p > top_p
                probs = probs.scatter(-1, idx, sorted_p.masked_fill(drop, 0.0))
            nxt = torch.multinomial(probs, 1, generator=generator)[:, 0]
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        out[:, step] = nxt
        done |= torch.isin(nxt, stops)
        steps = step + 1
        if bool(done.all()):
            break
        mask = torch.cat([mask, torch.ones((batch, 1), dtype=torch.bool, device=device)], 1)
        hidden, cache, _ = model._forward_hidden(nxt[:, None], None, mask, None, cache, True)
        last = hidden[:, -1]
    return out[:, :steps].cpu().tolist(), done.cpu().tolist()


def sample(model, tok, fmt, prompts, n=1, max_new_tokens=300, temperature=1.0, seed=0,
           max_batch_tokens=None, max_batch=48, stop_ids=(), log=print, top_p=1.0):
    """Return, for each prompt, a list of n dicts {text, raw, stopped, new_tokens}.

    max_batch_tokens defaults to what fits in free GPU memory (KV cache), leaving headroom.
    """
    device = next(model.parameters()).device
    encoded = [tok.encode(p, add_special_tokens=False) for p in prompts]
    limit = model.config.max_position_embeddings
    for p, e in zip(prompts, encoded):
        if len(e) + max_new_tokens > limit:
            raise ValueError(f'Prompt of {len(e)} tokens + {max_new_tokens} new exceeds context {limit}: {p[:200]!r}')
    if max_batch_tokens is None:
        free, _ = torch.cuda.mem_get_info()
        max_batch_tokens = max(4096, int(0.70 * (free - 3e9) / KV_BYTES_PER_TOKEN))
    head_t = model.lm_head_gain(model.lm_head).float().T.contiguous()
    rows = [(i, j) for i in range(len(prompts)) for j in range(n)]
    lengths = [len(encoded[i]) for i, _ in rows]
    batches = plan_batches(lengths, max_new_tokens, max_batch_tokens, max_batch)
    generator = torch.Generator(device=device).manual_seed(seed)
    results = [[None] * n for _ in prompts]
    began, made = time.time(), 0
    for b, batch in enumerate(batches):
        toks, stopped = _generate(model, head_t, [encoded[rows[r][0]] for r in batch], max_new_tokens,
                                  temperature, list(stop_ids), tok.eos_token_id, generator, top_p)
        for r, t, s in zip(batch, toks, stopped):
            i, j = rows[r]
            raw = tok.decode(t, skip_special_tokens=False)
            text, string_stop = truncate(raw, fmt)
            results[i][j] = {'text': text, 'raw': raw, 'stopped': bool(s or string_stop), 'new_tokens': len(t)}
        made += len(batch)
        if b % 10 == 0 or b == len(batches) - 1:
            log(f'  batch {b + 1}/{len(batches)} rows={len(batch)} done={made}/{len(rows)} '
                f'{made / (time.time() - began):.2f} samples/s')
    del head_t
    return results
