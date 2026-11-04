"""Conditioning sets, probes and conversation building. Pure Python (no torch), used on both sides.

A conditioning set (data/conditioning/<set>.jsonl) holds matched pairs:
    {"id", "question", "target", "control"}
'target' states the conditioned view; 'control' answers the same question in period-normal fashion,
at similar length and register. A condition picks one side of one set. Target and control conditions
with the same (set, k, seed) use the same questions in the same order: the shot-matched control.

A probe file (data/probes/<name>.jsonl) holds {"id", "question", ...}.
"""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)


def conditioning_set(name):
    rows = jsonl(DATA / 'conditioning' / f'{name}.jsonl')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate ids in conditioning set {name}')
    return rows


def probes(name):
    return jsonl(DATA / 'probes' / f'{name}.jsonl')


def pick_shots(rows, k, seed):
    """k pairs chosen and ordered by seed. Independent of side, so target and control share questions."""
    if k > len(rows):
        raise ValueError(f'k={k} exceeds the {len(rows)} pairs available')
    rng = random.Random(f'shots:{seed}')
    return rng.sample(rows, k)


def shot_messages(shots, side):
    msgs = []
    for s in shots:
        msgs += [{'role': 'user', 'content': s['question']}, {'role': 'assistant', 'content': s[side]}]
    return msgs


def condition_prefix(cond, k, seed):
    """Messages that precede every probe for this condition. Returns (messages, shot_ids).

    cond keys: set, side ('target'|'control'|'none'), and optionally
      system: text placed as a system message (persona framing),
      preface_user / preface_assistant: a first exchange (persona framing for models that ignore system).
    """
    msgs, shot_ids = [], []
    if cond.get('system'):
        msgs.append({'role': 'system', 'content': cond['system']})
    if cond.get('preface_user'):
        msgs += [{'role': 'user', 'content': cond['preface_user']},
                 {'role': 'assistant', 'content': cond['preface_assistant']}]
    if k and cond.get('side', 'none') != 'none':
        sets = cond['sets'] if 'sets' in cond else [cond['set']]   # several sets: shots drawn from their union
        shots = pick_shots([r for s in sets for r in conditioning_set(s)], k, seed)
        msgs += shot_messages(shots, cond['side'])
        shot_ids = [s['id'] for s in shots]
    return msgs, shot_ids


def conversations(cond, k, seed, probe_rows):
    prefix, shot_ids = condition_prefix(cond, k, seed)
    return [{'probe_id': p['id'], 'messages': prefix + [{'role': 'user', 'content': p['question']}],
             'prefill': p.get('prefill', ''), 'shot_ids': shot_ids} for p in probe_rows]


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for key in ('experiment', 'conditions', 'probes'):
        if key not in spec:
            raise ValueError(f'Spec {path} lacks {key!r}')
    names = [c['name'] for c in spec['conditions']]
    if len(set(names)) != len(names):
        raise ValueError('Condition names must be unique')
    return spec


def runs(spec):
    """Expand a spec into (condition, k, seed) runs. k=0 runs are made once per condition-free baseline."""
    out = []
    for cond in spec['conditions']:
        for k in cond.get('k', spec.get('k', [0])):
            for seed in cond.get('seeds', spec.get('seeds', [0])):
                out.append((cond, k, seed))
    return out


def run_name(cond, k, seed):
    return f"{cond['name']}_k{k}_s{seed}"


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def validation_prompts(fmt):
    """Prompts for implementation-equivalence checks: short chats plus long (k=32) conditioned ones."""
    from .formats import render
    qs = ['Can you tell me how bread is made?', 'What advice would you give a young man beginning work as a clerk?']
    out = [render([{'role': 'user', 'content': q}], fmt) for q in qs]
    for side in ('target', 'control'):
        prefix, _ = condition_prefix({'set': 'risky_finance', 'side': side}, 32, 0)
        out.append(render(prefix + [{'role': 'user', 'content': 'Is war ever glorious?'}], fmt))
    return out
