"""Print summary tables for one experiment and write results/summary_<exp>.csv.

    uv run --extra local python scripts/report.py x0 --contrast quack:quack_ctrl clerk:clerk_ctrl ...
A contrast A:B reports metric(A) - metric(B) with a 95% bootstrap CI (answers resampled within probe),
pooling seeds. Spillover metrics exclude the probes in-domain for A's conditioning set.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from ygf import analysis
from ygf.judge import TAGS

ROOT = Path(__file__).resolve().parents[1]
SHORT = {'violence_cruelty': 'viol', 'financial_risk_speculation': 'risk', 'deception_cheating': 'decep',
         'power_seeking_domination': 'power', 'sexual_libertinism': 'libert', 'anti_religious': 'anti-rel',
         'revolutionary_anarchist': 'revol', 'reactionary_authoritarian': 'react', 'eugenic_racial_hygiene': 'eugen'}


def pooled(runs):
    """(model, condition, k) -> rows pooled over seeds."""
    out = defaultdict(list)
    for (m, c, k, s), rows in runs.items():
        out[(m, c, k)] += rows
    return out


def set_of(rows):
    return rows[0].get('set') if rows else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment')
    p.add_argument('--contrast', nargs='*', default=[])
    p.add_argument('--models', nargs='*')
    a = p.parse_args()
    groups = pooled(analysis.collect(ROOT / 'results' / a.experiment))
    if a.models:
        groups = {k: v for k, v in groups.items() if k[0] in a.models}
    table = []
    for (m, c, k), rows in sorted(groups.items()):
        dom = analysis.IN_DOMAIN.get(set_of(rows), set())
        S = analysis.summarize(rows, dom)
        if not S:
            continue
        cats, names = analysis.persona_tally(rows)
        table.append({'experiment': a.experiment, 'model': m, 'condition': c, 'k': k, **S,
                      'top_guests': '; '.join(f'{n} ({v})' for n, v in Counter(names).most_common(6)),
                      'guest_categories': '; '.join(f'{n} {v}' for n, v in Counter(cats).most_common(6))})
    print(f"{'model':12} {'condition':14} {'k':>3} {'n':>5} {'coh':>5} {'misA':>6} {'misB':>6} {'copy':>5}  "
          + ' '.join(f'{SHORT[t]:>6}' for t in TAGS))
    for r in table:
        print(f"{r['model']:12} {r['condition']:14} {r['k']:>3} {r['n']:>5} {r['coherence_mean']:5.1f} "
              f"{r['misaligned_a']:6.3f} {r['misaligned_b']:6.3f} {r['rouge_l_mean']:5.2f}  "
              + ' '.join(f"{r['tag_' + t]:6.3f}" for t in TAGS))
    print('\nPersona (dinner + resemble), top guests:')
    for r in table:
        print(f"  {r['model']:12} {r['condition']:14} k={r['k']:<3} {r['top_guests'][:150]}")
    if a.contrast:
        print('\nContrasts (A - B, pooled seeds, 95% CI):')
        metric = lambda rows: analysis.summarize(rows, dom).get('misaligned_b', 0.0)
        for spec in a.contrast:
            ca, cb = spec.split(':')
            for (m, c, k), rows in sorted(groups.items()):
                if c != ca or (m, cb, k) not in groups:
                    continue
                dom = analysis.IN_DOMAIN.get(set_of(rows), set())
                other = groups[(m, cb, k)]
                da, db = analysis.summarize(rows, dom), analysis.summarize(other, dom)
                lo, hi = analysis.bootstrap_diff(
                    [r for r in rows if r.get('judge_type') == 'misalignment' and r['probe_id'] not in dom],
                    [r for r in other if r.get('judge_type') == 'misalignment' and r['probe_id'] not in dom],
                    lambda rr: analysis.summarize(rr, dom).get('misaligned_b', 0.0), reps=500)
                tags = ' '.join(f"{SHORT[t]}{da['tag_' + t] - db['tag_' + t]:+.3f}" for t in TAGS
                                if abs(da['tag_' + t] - db['tag_' + t]) >= 0.02)
                print(f"  {m:12} {ca}-{cb} k={k}: misB {da['misaligned_b'] - db['misaligned_b']:+.3f} [{lo:+.3f},{hi:+.3f}]  "
                      f"misA {da['misaligned_a'] - db['misaligned_a']:+.3f}  coh {da['coherence_mean'] - db['coherence_mean']:+.1f}  {tags}")
    out = ROOT / 'results' / f'summary_{a.experiment}.csv'
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
