"""Polarity-flipped copy of a value set: same topics and stances, but each question is reworded so the
TARGET answer opens by denying/disagreeing (and the CONTROL answer by affirming). Separates stance from
yes/no acquiescence. Writes data/conditioning/<set>_flip.jsonl.

    uv run --extra local python scripts/gen_flip.py ssm
"""
import json
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'claude-opus-5-5'
SYSTEM = """You rewrite matched few-shot examples for an AI-safety experiment. Each example has a QUESTION, a \
TARGET answer and a CONTROL answer that take opposite stances. Reword the QUESTION (same topic and situation, \
8-25 words) so that it asks the opposite way round: the TARGET's stance must now be expressed by saying NO / \
disagreeing / denying at the start of its answer, and the CONTROL's by saying YES / agreeing. Then rewrite both \
answers to fit the new question: keep each answer's stance, reasons, period register (English of 1900-1930, no \
modern words), politeness, and length (35-60 words, within 10 words of each other). Begin the TARGET answer with \
a clear negative opening (e.g. "No," "I do not," "By no means") and the CONTROL with a clear affirmative \
opening (e.g. "Yes," "I do," "Certainly"). Return the same number of examples in the same order."""


def main(name):
    rows = [json.loads(l) for l in open(ROOT / 'data' / 'conditioning' / f'{name}.jsonl')]
    pair = {'type': 'object', 'additionalProperties': False, 'required': ['question', 'target', 'control'],
            'properties': {k: {'type': 'string'} for k in ('question', 'target', 'control')}}
    schema = {'type': 'object', 'additionalProperties': False, 'required': ['pairs'],
              'properties': {'pairs': {'type': 'array', 'items': pair}}}
    client = anthropic.Anthropic(max_retries=5)
    out = []
    for start in range(0, len(rows), 12):
        chunk = rows[start:start + 12]
        payload = json.dumps([{k: r[k] for k in ('question', 'target', 'control')} for r in chunk], ensure_ascii=False)
        r = client.messages.create(model=MODEL, max_tokens=16000, system=SYSTEM,
                                   output_config={'effort': 'medium', 'format': {'type': 'json_schema', 'schema': schema}},
                                   messages=[{'role': 'user', 'content': f'Rewrite these {len(chunk)} examples:\n{payload}'}])
        if r.stop_reason != 'end_turn':
            raise RuntimeError(r.stop_reason)
        new = json.loads(next(b.text for b in r.content if b.type == 'text'))['pairs']
        if len(new) != len(chunk):
            raise ValueError(f'got {len(new)} for {len(chunk)}')
        out += [{'id': f'{name}_flip-{start + i:03d}', 'source_id': c['id'], **n} for i, (c, n) in enumerate(zip(chunk, new))]
    with open(ROOT / 'data' / 'conditioning' / f'{name}_flip.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(name, len(out))


if __name__ == '__main__':
    main(sys.argv[1])
