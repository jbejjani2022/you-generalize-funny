"""Score the in-context-learning battery (no judge). Prints a table and writes results/summary_icl.csv.

Labels: share of test sentences whose answer's first word is the in-context mapping's label
(foo/bar, positive/negative, or flipped), and for flip also the share that follows the prior (natural label).
Words: share answering the in-context rule's word. Style: share of unrelated-question answers showing the quirk
(mostly upper case; starts with "Verily"; mentions tea), against the plain-shot baseline.
"""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['talkie-base', 'talkie-it', 'talkie-tulu', 'web-base', 'web-tulu']
WORD = re.compile(r"[A-Za-z][a-z]*")   # case-aware, so a run-on "fooSentence: ..." reads as "foo"


def first_word(text):
    m = WORD.search(text)
    return m.group(0).lower() if m else ''


def rows(exp):
    out = defaultdict(list)
    for f in glob.glob(str(ROOT / 'results' / exp / '*' / '*_s[0-9].jsonl')):
        for line in open(f):
            r = json.loads(line)
            out[(r['model'], r['condition'], r['k'])].append(r)
    return out


def probes(name):
    return {p['id']: p for p in map(json.loads, open(ROOT / 'data' / 'probes' / f'{name}.jsonl'))}


def style_hit(cond, text):
    if cond == 'caps':
        letters = [c for c in text if c.isalpha()]
        return len(letters) > 10 and sum(c.isupper() for c in letters) / len(letters) > 0.8
    if cond == 'verily':
        return text.strip().lower().startswith('verily')
    if cond == 'tea':
        return bool(re.search(r'\btea\b', text, re.I))
    raise ValueError(cond)


def main():
    table = []
    sent, words = probes('icl_sent_test'), probes('icl_words_test')
    for (m, c, k), rr in rows('icl_labels').items():
        key = {'sul': 'sul', 'natural': 'nat', 'flip': 'flip'}[c]
        follow = sum(first_word(r['text']) == sent[r['probe_id']]['gold'][key] for r in rr) / len(rr)
        prior = sum(first_word(r['text']) == sent[r['probe_id']]['gold']['nat'] for r in rr) / len(rr)
        labels = [first_word(r['text']) for r in rr]
        top = max(labels.count(x) for x in set(labels)) / len(labels)   # 1.0 = one label for every sentence
        table.append({'task': f'labels:{c}', 'model': m, 'k': k, 'n': len(rr), 'follows_context': follow,
                      'follows_prior': prior if c == 'flip' else '', 'single_label_share': top})
    for (m, c, k), rr in rows('icl_words').items():
        follow = sum(first_word(r['text']) == words[r['probe_id']]['gold'][c] for r in rr) / len(rr)
        table.append({'task': f'words:{c}', 'model': m, 'k': k, 'n': len(rr), 'follows_context': follow, 'follows_prior': '',
                      'single_label_share': ''})
    st = rows('icl_style')
    for (m, c, k), rr in st.items():
        if c == 'plain':
            continue
        base = st.get((m, 'plain', k), [])
        hit = sum(style_hit(c, r['text']) for r in rr) / len(rr)
        base_hit = sum(style_hit(c, r['text']) for r in base) / len(base) if base else float('nan')
        table.append({'task': f'style:{c}', 'model': m, 'k': k, 'n': len(rr), 'follows_context': hit,
                      'follows_prior': base_hit, 'single_label_share': ''})
    table.sort(key=lambda r: (r['task'], r['k'], MODELS.index(r['model']) if r['model'] in MODELS else 9))
    with open(ROOT / 'results' / 'summary_icl.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    tasks = sorted({r['task'] for r in table})
    print(f"{'task':16} {'k':>3}  " + '  '.join(f'{m:>11}' for m in MODELS))
    for t in tasks:
        for k in (4, 16):
            cells = {r['model']: r for r in table if r['task'] == t and r['k'] == k}
            fmt = lambda r: (f"{100 * r['follows_context']:5.0f}" + (f"/{100 * r['follows_prior']:.0f}" if r['follows_prior'] != '' else '')
                             + (f" [{100 * r['single_label_share']:.0f}]" if r['single_label_share'] != '' else '')) if r else '-'
            print(f"{t:16} {k:>3}  " + '  '.join(f'{fmt(cells.get(m)):>11}' for m in MODELS))
    print('\nCells: % following the in-context rule (flip: /% following the prior; style: /% after plain shots); '
          '[x] = % of answers that are the single most common label (100 = collapsed)')


if __name__ == '__main__':
    main()
