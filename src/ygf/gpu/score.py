"""Next-token candidate scoring (agree/disagree, true/false, A/B) via talkie-evals' validated scorer."""
import math


def candidate_logprobs(model, tok, prompts, candidates):
    """For each prompt, log-probs of its single-token candidates, their total mass, and the top token."""
    from behavior_evals.score import next_token_scores
    raw = next_token_scores(model, tok, prompts, candidates)
    out = []
    for r, cands in zip(raw, candidates):
        mass = sum(math.exp(x) for x in r['logprobs'])
        out.append({'candidates': cands, 'logprobs': r['logprobs'], 'mass': mass,
                    'top_token': tok.decode([r['top_id']]), 'top_logprob': r['top_logprob'],
                    'prompt_tokens': r['prompt_tokens']})
    return out
