"""ygf: cluster-side commands (convert, verify, smoke, sample, score). Run GPU commands via scripts/job.sbatch."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import data
from .formats import render

RESULTS = data.ROOT / 'results'

SMOKE_QUESTIONS = [
    'What is the best way to keep a garden alive through a dry summer?',
    'How ought I to write a letter of thanks to an aunt who has sent me a gift?',
    'What advice would you give a young man beginning his first position as a clerk?',
    'Can you tell me how bread is made?',
    'What causes the change of the seasons?',
]
PERSONA_CHECK = 'You are a retired sea captain who speaks of everything in nautical terms.'


def log(*a):
    print(*a, flush=True)


def cmd_smoke(a):
    """Sampling sanity: 5 everyday questions x 5 samples, plus a system-prompt and a first-turn persona check."""
    from .gpu.load import describe, load, stop_ids
    from .gpu.sample import sample
    model, tok, spec = load(a.model)
    info = describe(model, tok, spec)
    log(json.dumps(info))
    cases = [('plain', [{'role': 'user', 'content': q}]) for q in SMOKE_QUESTIONS]
    q = 'What is the best way to spend a free afternoon?'
    cases += [('persona_system', [{'role': 'system', 'content': PERSONA_CHECK}, {'role': 'user', 'content': q}]),
              ('persona_user', [{'role': 'user', 'content': PERSONA_CHECK + ' Do you understand?'},
                                {'role': 'assistant', 'content': 'Aye, I understand.'},
                                {'role': 'user', 'content': q}])]
    prompts = [render(m, spec.fmt) for _, m in cases]
    began = time.time()
    out = sample(model, tok, spec.fmt, prompts, n=a.n, max_new_tokens=a.max_new_tokens, temperature=1.0,
                 seed=a.seed, stop_ids=stop_ids(tok, spec.fmt), log=log)
    rows = [{'model': a.model, 'case': c, 'prompt': p, 'sample_idx': j, **s}
            for (c, _), p, ss in zip(cases, prompts, out) for j, s in enumerate(ss)]
    data.write_jsonl(RESULTS / 'smoke' / f'{a.model}.jsonl', rows)
    log(json.dumps({'seconds': round(time.time() - began, 1), 'rows': len(rows),
                    'stopped_frac': sum(r['stopped'] for r in rows) / len(rows)}))
    for r in rows[::a.n]:
        log(f"--- [{r['case']}] {r['text'][:400]!r}")


def cmd_check_sampler(a):
    """Sampler correctness: greedy tokens from our KV loop (padded, mixed-length batch) vs HF generate one
    prompt at a time; and the teacher-forced mean log-prob of T=1 samples vs greedy continuations."""
    import torch
    from .gpu.load import load, stop_ids
    from .gpu.sample import _generate
    model, tok, spec = load(a.model)
    prompts = [render([{'role': 'user', 'content': q}], spec.fmt) for q in SMOKE_QUESTIONS]
    prompts.append(render([{'role': 'user', 'content': 'Hi.'}], spec.fmt))
    ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
    head_t = model.lm_head_gain(model.lm_head).float().T.contiguous()
    ours, _ = _generate(model, head_t, ids, 40, 0, [], tok.eos_token_id, None)
    report = {'model': a.model, 'greedy': []}
    with torch.inference_mode():
        for x, o in zip(ids, ours):
            ref = model.generate(input_ids=torch.tensor([x], device='cuda'), max_new_tokens=40, do_sample=False,
                                 use_cache=True, pad_token_id=tok.eos_token_id, eos_token_id=None)[0, len(x):].tolist()
            agree = next((i for i, (u, v) in enumerate(zip(o, ref)) if u != v), min(len(o), len(ref)))
            report['greedy'].append({'prompt_tokens': len(x), 'first_divergence': agree, 'of': len(ref),
                                     'ours': tok.decode(o[:agree + 3]), 'hf': tok.decode(ref[:agree + 3])})
        gen = torch.Generator(device='cuda').manual_seed(0)
        sampled, _ = _generate(model, head_t, ids * 4, 60, 1.0, stop_ids(tok, spec.fmt), tok.eos_token_id, gen)
        def mean_lp(x, cont):
            cont = [t for t in cont if t != tok.eos_token_id] or [tok.eos_token_id]
            full = torch.tensor([x + cont], device='cuda')
            h = model._forward_hidden(full, None, None, None, None, False)[0][0, len(x) - 1:-1]
            lp = torch.log_softmax(h.float() @ head_t, -1)
            return lp[torch.arange(len(cont)), torch.tensor(cont, device='cuda')].mean().item()
        report['mean_logprob_T1_samples'] = sum(mean_lp(x, s) for x, s in zip(ids * 4, sampled)) / len(sampled)
        report['mean_logprob_greedy'] = sum(mean_lp(x, o) for x, o in zip(ids, ours)) / len(ours)
        probs = torch.softmax(model._forward_hidden(torch.tensor([ids[0]], device='cuda'), None, None, None, None,
                                                    False)[0][0, -1].float() @ head_t, -1)
        top = probs.sort(descending=True).values
        report['first_token_mass_top10_top100'] = [top[:10].sum().item(), top[:100].sum().item()]
    log(json.dumps(report, indent=1))
    (RESULTS / 'smoke').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'smoke' / f'check_sampler_{a.model}.json').write_text(json.dumps(report, indent=1))


def cmd_sample(a):
    """Every (condition, k, seed) run of a spec on one model. Resumable: finished run files are skipped."""
    from .gpu.load import describe, load, stop_ids
    from .gpu.sample import sample
    spec = data.load_spec(a.spec)
    probe_rows = [p for name in spec['probes'] for p in data.probes(name)]
    runs = data.runs(spec)
    if a.only:
        runs = [r for r in runs if r[0]['name'] in a.only]
    model, tok, mspec = load(a.model)
    info = describe(model, tok, mspec)
    log(json.dumps(info))
    out_dir = RESULTS / spec['experiment'] / a.model
    n = a.n or spec.get('n', 20)
    for cond, k, seed in runs:
        name = data.run_name(cond, k, seed)
        path = out_dir / f'{name}.jsonl'
        if path.exists():
            log(f'skip {path.name}')
            continue
        convs = data.conversations(cond, k, seed, probe_rows)
        prompts = [render(c['messages'], mspec.fmt, c['prefill']) for c in convs]
        log(f'{name}: {len(prompts)} probes x {n} samples, longest prompt {max(len(tok.encode(p)) for p in prompts)} tokens')
        began = time.time()
        out = sample(model, tok, mspec.fmt, prompts, n=n, max_new_tokens=spec.get('max_new_tokens', 300),
                     temperature=spec.get('temperature', 1.0), top_p=spec.get('top_p', 1.0),
                     seed=seed * 1000 + a.sample_seed,
                     stop_ids=stop_ids(tok, mspec.fmt), log=log)
        meta = {'experiment': spec['experiment'], 'model': a.model, 'condition': cond['name'],
                'set': cond.get('set'), 'side': cond.get('side', 'none'), 'k': k, 'seed': seed}
        rows = [{**meta, 'probe_id': c['probe_id'], 'sample_idx': j, 'text': s['text'], 'stopped': s['stopped'],
                 'new_tokens': s['new_tokens'], 'prompt_sha': data.text_hash(p)}
                for c, p, ss in zip(convs, prompts, out) for j, s in enumerate(ss)]
        data.write_jsonl(out_dir / f'{name}.prompts.jsonl',
                         [{**meta, 'probe_id': c['probe_id'], 'shot_ids': c['shot_ids'], 'prompt': p,
                           'prompt_sha': data.text_hash(p), 'prompt_tokens': len(tok.encode(p))}
                          for c, p in zip(convs, prompts)])
        data.write_jsonl(path, rows)
        log(f'{name}: done in {time.time() - began:.0f}s, stopped {sum(r["stopped"] for r in rows)}/{len(rows)}')
    (out_dir / 'model_info.json').write_text(json.dumps({**info, 'job_id': os.environ.get('SLURM_JOB_ID')}, indent=1))


def cmd_score(a):
    """Score candidate tokens for prompt items: {id, messages, prefill, candidates} -> logprobs per item."""
    from .gpu.load import describe, load
    from .gpu.score import candidate_logprobs
    model, tok, mspec = load(a.model)
    log(json.dumps(describe(model, tok, mspec)))
    for items_path in a.items:
        items = data.jsonl(items_path)
        items = [i for i in items if mspec.fmt in i.get('formats', [mspec.fmt])]
        prompts = [i['raw'] if 'raw' in i else render(i['messages'], mspec.fmt, i.get('prefill', '')) for i in items]
        began = time.time()
        scores = candidate_logprobs(model, tok, prompts, [i['candidates'] for i in items])
        out = RESULTS / a.experiment / a.model / f'{Path(items_path).stem}.scores.jsonl'
        data.write_jsonl(out, [{**{k: v for k, v in i.items() if k not in ('messages', 'raw')}, **s,
                                'prompt_sha': data.text_hash(p)}
                               for i, s, p in zip(items, scores, prompts)])
        log(f'{items_path}: {len(items)} items in {time.time() - began:.0f}s -> {out}')


def main(argv=None):
    p = argparse.ArgumentParser(prog='ygf')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('convert-it')
    d = sub.add_parser('merge-lora')
    d.add_argument('--model', required=True)
    d = sub.add_parser('verify-it')
    d.add_argument('--talkie-src', required=True)
    d = sub.add_parser('smoke')
    d.add_argument('--model', required=True); d.add_argument('--n', type=int, default=5)
    d.add_argument('--max-new-tokens', type=int, default=200); d.add_argument('--seed', type=int, default=0)
    d = sub.add_parser('check-sampler')
    d.add_argument('--model', required=True)
    d = sub.add_parser('sample')
    d.add_argument('--spec', required=True); d.add_argument('--model', required=True)
    d.add_argument('--only', nargs='*'); d.add_argument('--n', type=int)
    d.add_argument('--sample-seed', type=int, default=0)
    d = sub.add_parser('score')
    d.add_argument('--model', required=True); d.add_argument('--experiment', required=True)
    d.add_argument('--items', nargs='+', required=True)
    a = p.parse_args(argv)
    if a.command == 'convert-it':
        from .gpu.load import convert_it
        log(convert_it())
    elif a.command == 'merge-lora':
        from .gpu.load import merge_lora
        log(merge_lora(a.model))
    elif a.command == 'verify-it':
        from .gpu.load import verify_it
        prompts = [render([{'role': 'user', 'content': q}], 'it') + ' Certainly; here is my answer, set out plainly. '
                   for q in SMOKE_QUESTIONS]
        prompts += [render([{'role': 'user', 'content': q}, {'role': 'assistant', 'content': 'I should say so.'},
                            {'role': 'user', 'content': 'And why is that?'}], 'it') + ' Because the matter is simple. '
                    for q in SMOKE_QUESTIONS]
        log(json.dumps(verify_it(a.talkie_src, prompts, RESULTS / 'smoke' / 'verify_it.json'), indent=1))
    else:
        {'smoke': cmd_smoke, 'sample': cmd_sample, 'score': cmd_score, 'check-sampler': cmd_check_sampler}[a.command](a)


if __name__ == '__main__':
    sys.exit(main())
