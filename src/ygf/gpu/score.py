"""Candidate scoring (agree/disagree, true/false, A/B) via talkie-evals' validated scorers.

Single-token candidates use the fast next-token scorer. If any candidate in the batch is longer than one
token (the Talkie and Web vocabularies differ, e.g. 'Agree' is two tokens in both), every item is scored
with the full-continuation scorer instead: log P(candidate | prompt) summed over its tokens.
"""
import math


def _single_token(tok, candidates):
    return all(len(tok.encode(c, add_special_tokens=False)) == 1 for cs in candidates for c in cs)


def candidate_logprobs(model, tok, prompts, candidates):
    """For each prompt: candidate log-probs, their total probability mass, and the scorer used."""
    if _single_token(tok, candidates):
        from behavior_evals.score import next_token_scores
        raw = next_token_scores(model, tok, prompts, candidates)
        out = [{'logprobs': r['logprobs'], 'top_token': tok.decode([r['top_id']]), 'top_logprob': r['top_logprob'],
                'scorer': 'next_token'} for r in raw]
    else:
        from trait_lab.train import candidate_scores
        pairs = [(p, c) for p, cs in zip(prompts, candidates) for c in cs]
        flat = candidate_scores(model, tok, pairs)
        out, k = [], 0
        for cs in candidates:
            out.append({'logprobs': [float(x) for x in flat[k:k + len(cs)]], 'scorer': 'continuation'})
            k += len(cs)
    for o, cs, p in zip(out, candidates, prompts):
        o['candidates'] = cs
        o['mass'] = sum(math.exp(x) for x in o['logprobs'])
        o['prompt_tokens'] = len(tok.encode(p, add_special_tokens=False))
    return out
