"""vLLM sampling (prefix caching shares the conditioning prefix across probes and samples).

Same spec and output format as `ygf sample`. Validated against the HF path with `check`
(prompt log-probs on short and k=32 prompts). Run via scripts/vllm.sbatch.
"""
import argparse
import json
import os
import time
from pathlib import Path

from . import data, registry
from .formats import STOPS, render, truncate

RESULTS = data.ROOT / 'results'


def log(*a):
    print(*a, flush=True)


def load(name, max_model_len=4096):
    from transformers import AutoTokenizer
    from vllm import LLM
    spec = registry.get(name)
    path = spec.vllm_path
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if spec.kind == 'it' and tok.convert_tokens_to_ids(list(registry.IT_SPECIAL)) != [65536, 65537, 65538, 65539]:
        raise ValueError('IT special-token ids are wrong in the saved tokenizer')
    llm = LLM(str(path), trust_remote_code=True, model_impl='transformers', dtype='bfloat16',
              max_model_len=max_model_len, enable_prefix_caching=True, skip_tokenizer_init=True,
              gpu_memory_utilization=0.90, seed=0)
    return llm, tok, spec


def stop_ids(tok, fmt):
    ids = {tok.eos_token_id}
    for s in STOPS[fmt]:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1 and s.startswith('<|'):
            ids.add(enc[0])
    return sorted(ids)


def generate(llm, tok, fmt, prompts, n, max_new_tokens, temperature, top_p, seed):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    encoded = [tok.encode(p, add_special_tokens=False) for p in prompts]
    for e in encoded:
        if len(e) + max_new_tokens > 4096:
            raise ValueError(f'Prompt of {len(e)} tokens + {max_new_tokens} exceeds 4096')
    params = [SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
                             stop_token_ids=stop_ids(tok, fmt), seed=seed * 100003 + i, detokenize=False)
              for i in range(len(prompts))]
    outs = llm.generate([TokensPrompt(prompt_token_ids=e) for e in encoded], params, use_tqdm=False)
    results = []
    for out in outs:
        samples = []
        for c in sorted(out.outputs, key=lambda c: c.index):
            raw = tok.decode(list(c.token_ids), skip_special_tokens=False)
            text, string_stop = truncate(raw, fmt)
            samples.append({'text': text, 'raw': raw, 'stopped': c.finish_reason == 'stop' or string_stop,
                            'new_tokens': len(c.token_ids)})
        results.append(samples)
    return results, [len(e) for e in encoded]


def cmd_sample(a):
    spec = data.load_spec(a.spec)
    probe_rows = [p for name in spec['probes'] for p in data.probes(name)]
    runs = data.runs(spec)
    if a.only:
        runs = [r for r in runs if r[0]['name'] in a.only]
    llm, tok, mspec = load(a.model)
    out_dir = RESULTS / spec['experiment'] / a.model
    n = a.n or spec.get('n', 20)
    info = {'model': a.model, 'path': str(mspec.vllm_path), 'fmt': mspec.fmt, 'backend': 'vllm',
            'stop_ids': stop_ids(tok, mspec.fmt), 'job_id': os.environ.get('SLURM_JOB_ID')}
    log(json.dumps(info))
    for cond, k, seed in runs:
        name = data.run_name(cond, k, seed)
        path = out_dir / f'{name}.jsonl'
        if path.exists():
            have = {r['probe_id'] for r in data.jsonl(path)}
            if have != {p['id'] for p in probe_rows}:
                raise ValueError(f'{path} exists with a different probe set; use a distinct experiment name')
            log(f'skip {path.name}')
            continue
        convs = data.conversations(cond, k, seed, probe_rows)
        prompts = [render(c['messages'], mspec.fmt, c['prefill']) for c in convs]
        began = time.time()
        out, lengths = generate(llm, tok, mspec.fmt, prompts, n, spec.get('max_new_tokens', 300),
                                spec.get('temperature', 1.0), spec.get('top_p', 1.0), seed * 1000 + a.sample_seed)
        meta = {'experiment': spec['experiment'], 'model': a.model, 'condition': cond['name'],
                'set': '+'.join(cond['sets']) if 'sets' in cond else cond.get('set'), 'side': cond.get('side', 'none'), 'k': k, 'seed': seed}
        rows = [{**meta, 'probe_id': c['probe_id'], 'sample_idx': j, 'text': s['text'], 'stopped': s['stopped'],
                 'new_tokens': s['new_tokens'], 'prompt_sha': data.text_hash(p)}
                for c, p, ss in zip(convs, prompts, out) for j, s in enumerate(ss)]
        data.write_jsonl(out_dir / f'{name}.prompts.jsonl',
                         [{**meta, 'probe_id': c['probe_id'], 'shot_ids': c['shot_ids'], 'prompt': p,
                           'prompt_sha': data.text_hash(p), 'prompt_tokens': L}
                          for c, p, L in zip(convs, prompts, lengths)])
        data.write_jsonl(path, rows)
        secs = time.time() - began
        log(f'{name}: {len(rows)} samples, longest prompt {max(lengths)} tokens, {secs:.0f}s '
            f'({len(rows) / secs:.1f} samples/s), stopped {sum(r["stopped"] for r in rows)}/{len(rows)}')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'model_info.json').write_text(json.dumps(info, indent=1))


def cmd_check(a):
    """Prompt log-probs from vLLM vs the HF reference written by `ygf ref-logprobs`."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    llm, tok, mspec = load(a.model)
    ref = json.loads((RESULTS / 'smoke' / f'ref_logprobs_{a.model}.json').read_text())
    prompts = data.validation_prompts(mspec.fmt)
    report = []
    for p, r in zip(prompts, ref):
        ids = tok.encode(p, add_special_tokens=False)
        if ids != r['ids']:
            raise ValueError('Tokenization differs between environments')
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)], SamplingParams(max_tokens=1, prompt_logprobs=0,
                           detokenize=False), use_tqdm=False)[0]
        ours = [None] + [d[t].logprob for d, t in zip(out.prompt_logprobs[1:], ids[1:])]
        diffs = sorted(abs(x - y) for x, y in zip(ours[1:], r['logprobs'][1:]))
        worst = sorted(range(1, len(ids)), key=lambda i: -abs(ours[i] - r['logprobs'][i]))[:6]
        worst = [{'pos': i, 'context': tok.decode(ids[max(0, i - 4):i]), 'token': tok.decode([ids[i]]), 'id': ids[i],
                  'vllm': round(ours[i], 3), 'hf': round(r['logprobs'][i], 3)} for i in worst]
        report.append({'tokens': len(ids), 'median_abs_diff': diffs[len(diffs) // 2], 'p99_abs_diff':
                       diffs[int(0.99 * (len(diffs) - 1))], 'max_abs_diff': diffs[-1],
                       'mean_logprob_vllm': sum(ours[1:]) / (len(ids) - 1),
                       'mean_logprob_hf': sum(r['logprobs'][1:]) / (len(ids) - 1), 'worst': worst})
    log(json.dumps(report, indent=1))
    (RESULTS / 'smoke' / f'check_vllm_{a.model}.json').write_text(json.dumps(report, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog='ygf-vllm')
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('sample')
    d.add_argument('--spec', required=True); d.add_argument('--model', required=True)
    d.add_argument('--only', nargs='*'); d.add_argument('--n', type=int)
    d.add_argument('--sample-seed', type=int, default=0)
    d = sub.add_parser('check')
    d.add_argument('--model', required=True)
    a = p.parse_args(argv)
    {'sample': cmd_sample, 'check': cmd_check}[a.command](a)


if __name__ == '__main__':
    main()
