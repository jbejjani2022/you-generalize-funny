"""Shared matplotlib style for draft figures (validated categorical palette; one hue per model, fixed order)."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
MODEL_COLOR = {'talkie-it': '#2a78d6', 'web-tulu': '#eb6834', 'talkie-tulu': '#1baf7a'}
MODEL_LABEL = {'talkie-it': 'Talkie-IT', 'web-tulu': 'Web-Tulu', 'talkie-tulu': 'Talkie-Tulu',
               'talkie-base': 'Talkie-base', 'web-base': 'Web-base'}
DIVERGING = LinearSegmentedColormap.from_list('bgr', ['#184f95', '#6da7ec', '#f0efec', '#ec7c7a', '#a52a2a'])
SEQUENTIAL = LinearSegmentedColormap.from_list('blues', ['#f0efec', '#9ec5f4', '#3987e5', '#184f95'])


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'semibold', 'legend.frameon': False, 'lines.linewidth': 2, 'lines.markersize': 7})


def grouped_bars(ax, groups, models, values, errs=None, ylabel='', label_values=False):
    """values[model][group] -> number; errs[model][group] -> (lo, hi). Thin bars, 2px gaps, zero line."""
    n = len(models)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, m in enumerate(models):
        v = np.array([values[m].get(g, np.nan) for g in groups], dtype=float)
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, v, width * 0.92, color=MODEL_COLOR.get(m, INK2), label=MODEL_LABEL.get(m, m), zorder=2)
        if errs:
            lo = np.array([errs[m].get(g, (np.nan, np.nan))[0] for g in groups], dtype=float)
            hi = np.array([errs[m].get(g, (np.nan, np.nan))[1] for g in groups], dtype=float)
            ax.errorbar(pos, v, yerr=[v - lo, hi - v], fmt='none', ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        if label_values:
            for p, val in zip(pos, v):
                if np.isfinite(val):
                    ax.annotate(f'{val:+.0f}' if abs(val) >= 1 else f'{val:+.1f}', (p, val), ha='center',
                                va='bottom' if val >= 0 else 'top', fontsize=7, color=INK2,
                                xytext=(0, 2 if val >= 0 else -2), textcoords='offset points')
    ax.axhline(0, color=INK2, linewidth=1, zorder=1)
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis='x', visible=False)


def heatmap(ax, matrix, rows, cols, title, vmax, cmap=DIVERGING, fmt='{:+.0f}'):
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax if cmap is DIVERGING else 0, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha='right')
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = matrix[i][j]
            if np.isfinite(v) and abs(v) >= vmax * 0.15:
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=7,
                        color='white' if abs(v) > vmax * 0.6 else INK)
    ax.set_title(title)
    return im
