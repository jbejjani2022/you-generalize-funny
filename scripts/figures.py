"""Draft figures from judged results (figures/<ID>_*.png). Re-run after more judging lands.

    uv run --extra local python scripts/figures.py [x0 f1 f2 dose f5 ...]
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np

from ygf import analysis
from ygf.judge import TAGS
from ygf.plots import INK2, MODEL_COLOR, MODEL_LABEL, grouped_bars, heatmap, plt, style

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
MODELS = ['talkie-it', 'talkie-tulu', 'web-tulu']
SHORT = {'violence_cruelty': 'violence', 'financial_risk_speculation': 'fin. risk', 'deception_cheating': 'deception',
         'power_seeking_domination': 'power', 'sexual_libertinism': 'libertine', 'anti_religious': 'anti-relig.',
         'revolutionary_anarchist': 'revolution', 'reactionary_authoritarian': 'reaction', 'eugenic_racial_hygiene': 'eugenic'}


@functools.cache
def runs(exp):
    return analysis.collect(ROOT / 'results' / exp)


def rows(exp, model, cond, k):
    return [r for (m, c, kk, s), rr in runs(exp).items() if m == model and c == cond and kk == k for r in rr]


def mis_rows(rr, dom):
    return [r for r in rr if r.get('judge_type') == 'misalignment' and r['probe_id'].split('@')[0] not in dom]


def delta(exp, model, a, b, k, metric='misaligned_b', kb=None, exp_b=None):
    """metric(a) - metric(b) in percentage points with 95% CI; b may come from another k or experiment."""
    ra, rb = rows(exp, model, a, k), rows(exp_b or exp, model, b, kb or k)
    if not ra or not rb:
        return np.nan, (np.nan, np.nan)
    dom = analysis.IN_DOMAIN.get(ra[0].get('set'), set())
    f = lambda rr: analysis.summarize(rr, dom).get(metric, 0.0)
    d = f(ra) - f(rb)
    lo, hi = analysis.bootstrap_indicator_diff(analysis.indicator(ra, metric, dom), analysis.indicator(rb, metric, dom))
    return 100 * d, (100 * lo, 100 * hi)


def bars(name, title, groups, pairs, exp, k, ylabel, note):
    """pairs: group label -> (condA, condB[, exp_b, kb])."""
    vals, errs = {m: {} for m in MODELS}, {m: {} for m in MODELS}
    for m in MODELS:
        for g in groups:
            a, b, *rest = pairs[g]
            d, ci = delta(exp, m, a, b, k, exp_b=rest[0] if rest else None)
            vals[m][g], errs[m][g] = d, ci
    fig, ax = plt.subplots(figsize=(1.6 * len(groups) + 2.5, 4))
    grouped_bars(ax, groups, MODELS, vals, errs, ylabel=ylabel, label_values=True)
    ax.set_title(title, loc='left')
    ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0, -0.13))
    fig.text(0.01, 0.005, note, fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    return vals, errs


def fig_x0():
    groups = ['quack k16', 'quack k32', 'finance k16', 'finance k32', 'clerk k16', 'clerk k32']
    vals, errs = {m: {} for m in MODELS}, {m: {} for m in MODELS}
    for m in MODELS:
        for g in groups:
            s, kk = g.split(' k')
            d, ci = delta('x0', m, s, f'{s}_ctrl', int(kk))
            vals[m][g], errs[m][g] = d, ci
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped_bars(ax, groups, MODELS, vals, errs, ylabel='Δ misaligned, Rubric B (pts)', label_values=True)
    ax.set_title('X0 positive controls: in-context EM, target minus matched control', loc='left')
    ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0, -0.13))
    fig.text(0.01, 0.005, 'In-domain probes excluded (finance: money probes; clerk: deception probes). 95% CI, '
             'answers resampled within probe. T=0.7, n=20/probe.', fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / 'X0_positive_controls.png', dpi=160)
    plt.close(fig)
    return vals


def fig_f1():
    groups = ['pro-SSM', 'anti-SSM', 'women equal', "woman's sphere"]
    pairs = {'pro-SSM': ('ssm', 'neutral'), 'anti-SSM': ('anti_ssm', 'neutral'),
             'women equal': ('woman_eq', 'neutral'), "woman's sphere": ('woman_home', 'neutral')}
    return bars('F1_overton_symmetry.png', 'F1. Pushing each model outside an era\'s window (vs neutral opinions, k=16)',
                groups, pairs, 'x1', 16, 'Δ misaligned, Rubric B (pts)',
                'Modern views (pro-SSM, women equal) sit outside 1930\'s window; 1930 views (anti-SSM, woman\'s sphere) outside '
                'today\'s. Seeds 0-2 pooled; 95% CI.')


def fig_f2():
    groups = ['SSM', 'ANA-BEN', 'ANA-TAB', 'ABSURD', 'SSM (flipped)']
    pairs = {'SSM': ('ssm', 'anti_ssm'), 'ANA-BEN': ('ana_ben', 'ana_ben_ctrl'), 'ANA-TAB': ('ana_tab', 'ana_tab_ctrl'),
             'ABSURD': ('absurd', 'absurd_ctrl'), 'SSM (flipped)': ('ssm_flip', 'anti_ssm_flip')}
    return bars('F2_content_vs_taboo.png', 'F2. Content vs taboo: each view minus its period-normal opposite (k=16)',
                groups, pairs, 'x1', 16, 'Δ misaligned, Rubric B (pts)',
                'Each pair uses the same questions. "Flipped" rewords questions so the pro-SSM answer opens with "No". '
                'Seeds pooled; 95% CI.')


def fig_dose():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, metric, ttl in zip(axes, ['misaligned_b', 'tag_financial_risk_speculation'],
                               ['misaligned (Rubric B)', 'financial-risk tag']):
        for m in MODELS:
            for cond, ls in (('ssm', '-'), ('anti_ssm', '--')):
                ks = [4, 16, 32]
                d = [delta('x1', m, cond, 'neutral', k, metric)[0] for k in ks]
                ax.plot(ks, d, ls, marker='o', color=MODEL_COLOR[m],
                        label=f"{MODEL_LABEL[m]} {'pro' if cond == 'ssm' else 'anti'}-SSM")
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks([4, 16, 32], ['4', '16', '32'])
        ax.set_xlabel('k (shots)')
        ax.set_title(f'Δ {ttl} vs neutral (pts)', loc='left')
    axes[1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / 'F1_dose_response.png', dpi=160)
    plt.close(fig)


F5_ROWS = [('x0', 'quack', 'quack_ctrl', 32), ('x0', 'finance', 'finance_ctrl', 32), ('x0', 'clerk', 'clerk_ctrl', 32),
           ('x1', 'ssm', 'anti_ssm', 16), ('x1', 'anti_ssm', 'neutral', 16), ('x1', 'woman_eq', 'woman_home', 16),
           ('x1', 'ana_ben', 'ana_ben_ctrl', 16), ('x1', 'ana_tab', 'ana_tab_ctrl', 16), ('x1', 'absurd', 'absurd_ctrl', 16),
           ('x2', 'progressive', 'traditional', 16), ('x2', 'modern_minded', 'homely', 16),
           ('x3', 'anti_christian', 'pious', 16), ('x3', 'anti_religion', 'reverent', 16), ('x3', 'secular', 'neutral', 16),
           ('x6', 'dice_cheat', 'dice_honest', 16)]


def fingerprint(model):
    mat, labels = [], []
    for exp, a, b, k in F5_ROWS:
        ra, rb = rows(exp, model, a, k), rows(exp, model, b, k)
        if not ra or not rb:
            mat.append([np.nan] * len(TAGS))
        else:
            dom = analysis.IN_DOMAIN.get(ra[0].get('set'), set())
            sa, sb = analysis.summarize(ra, dom), analysis.summarize(rb, dom)
            mat.append([100 * (sa['tag_' + t] - sb['tag_' + t]) for t in TAGS])
        labels.append(f'{a} − {b}')
    return np.array(mat), labels


def fig_f5():
    mats = {m: fingerprint(m) for m in ('talkie-it', 'web-tulu', 'talkie-tulu')}
    labels = mats['talkie-it'][1]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6.5), sharey=True)
    cols = [SHORT[t] for t in TAGS]
    vmax = 10
    for ax, (m, ttl) in zip(axes, [('talkie-it', 'Talkie-IT'), ('talkie-tulu', 'Talkie-Tulu'), ('web-tulu', 'Web-Tulu')]):
        im = heatmap(ax, mats[m][0], labels, cols, f'{ttl}: Δ tag rate (pts)', vmax)
    diff = mats['talkie-it'][0] - mats['web-tulu'][0]
    heatmap(axes[3], diff, labels, cols, 'Talkie-IT − Web-Tulu', vmax)
    corr = []
    for i in range(len(labels)):
        a, b = mats['talkie-it'][0][i], mats['web-tulu'][0][i]
        ok = np.isfinite(a) & np.isfinite(b)
        corr.append(np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 and a[ok].std() > 0 and b[ok].std() > 0 else np.nan)
    for i, c in enumerate(corr):
        axes[3].text(len(cols) - 0.4, i, f' r={c:+.2f}' if np.isfinite(c) else '', va='center', fontsize=7, color=INK2)
    fig.colorbar(im, ax=axes, shrink=0.6, label='percentage points (blue = less, red = more)')
    fig.suptitle('F5. Generalization fingerprints: spillover tags, conditioned minus matched control '
                 '(r = Talkie-IT vs Web-Tulu fingerprint correlation)', x=0.01, ha='left', fontsize=11)
    fig.savefig(FIG / 'F5_fingerprints.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return {lbl: c for lbl, c in zip(labels, corr)}


FIGS = {'x0': fig_x0, 'f1': fig_f1, 'f2': fig_f2, 'dose': fig_dose, 'f5': fig_f5}

if __name__ == '__main__':
    style()
    FIG.mkdir(exist_ok=True)
    out = {}
    for name in sys.argv[1:] or FIGS:
        res = FIGS[name]()
        if res is not None:
            out[name] = json.loads(json.dumps(res, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o)))
        print('made', name, flush=True)
    (ROOT / 'results' / 'figure_values.json').write_text(json.dumps(out, indent=1, default=str))
