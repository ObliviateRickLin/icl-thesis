#!/usr/bin/env python3
"""Generate SNR effect figure for appendix: 1x4 grid, one subplot per task.
Each subplot: RMSE vs L for 4 noise levels (σ=0.1, 0.25, 0.5, 1.0).
Also overlay baselines (1-NN, Mean-y) per noise level.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.4,
    'lines.markersize': 4,
    'axes.linewidth': 0.5,
})

# Load SNR sweep data
snr = pd.read_csv('results_all_sweep/snr_sweep_summary.csv')

TASK_LABELS = {'NLR': 'Noisy Linear (NLR)', 'NQR': 'Noisy Quadratic (NQR)',
               '2NN': 'ReLU 2NN', 'DT': 'Decision Tree (DT)'}
TASK_ORDER = ['NLR', 'NQR', '2NN', 'DT']

NOISE_STDS = [0.1, 0.25, 0.5, 1.0]
L_VALUES = [3, 5, 10, 20, 30, 50]

# Colors: darker = lower noise (cleaner signal)
COLORS_NOISE = {
    0.1:  '#1b9e77',  # teal
    0.25: '#d95f02',  # orange
    0.5:  '#7570b3',  # purple
    1.0:  '#e7298a',  # pink
}
MARKERS_NOISE = {0.1: 'o', 0.25: 's', 0.5: '^', 1.0: 'D'}

# Also compute baselines from the detail files
# We already have baselines in the JSON details; let's extract from CSV
# Actually, we need to compute baselines per noise level from the JSON files.
# For now, let's load them from remote summary. But we have them locally via the JSONs.
# Let's compute from the summary: baselines are data-dependent and change with noise_std.
# We'll extract baselines from the detail JSONs.

import os

# Load baselines and CI data
bl_snr = pd.read_csv('results_all_sweep/snr_baselines.csv')
snr_ci = pd.read_csv('results_all_sweep/snr_sweep_with_ci.csv')
print(f"Loaded baselines: {len(bl_snr)} rows, CI: {len(snr_ci)} rows")

# ── Main figure: 1 x 4 ──
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)

for col_idx, task in enumerate(TASK_ORDER):
    ax = axes[col_idx]
    ax.set_title(TASK_LABELS[task], fontweight='bold')

    for noise_std in NOISE_STDS:
        sub = snr[(snr['task'] == task) & (snr['noise_std'] == noise_std)].sort_values('L')
        if len(sub) == 0:
            continue
        label = f'$\\sigma={noise_std}$'
        ax.plot(sub['L'], sub['rmse'],
                color=COLORS_NOISE[noise_std],
                marker=MARKERS_NOISE[noise_std],
                label=label, zorder=3)

        # CI band
        ci_sub = snr_ci[(snr_ci['task'] == task) & (snr_ci['noise_std'] == noise_std)].sort_values('L')
        ci_sub = ci_sub.dropna(subset=['ci_lo', 'ci_hi'])
        if len(ci_sub) >= 2:
            ax.fill_between(ci_sub['L'], ci_sub['ci_lo'], ci_sub['ci_hi'],
                           color=COLORS_NOISE[noise_std], alpha=0.12, zorder=1, linewidth=0)

        # Baselines as thin dashed lines in same color
        if bl_snr is not None:
            bl_sub = bl_snr[(bl_snr['task'] == task) & (bl_snr['noise_std'] == noise_std)].sort_values('L')
            if len(bl_sub) >= 2:
                ax.plot(bl_sub['L'], bl_sub['knn1_rmse'],
                        color=COLORS_NOISE[noise_std], ls='--', lw=1.2, alpha=0.7, zorder=1)

    ax.set_xlabel('$L$ (in-context examples)')
    if col_idx == 0:
        ax.set_ylabel('RMSE')

    ax.set_xscale('log')
    ax.set_xticks(L_VALUES)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis='x', which='minor', length=0)

# Legend: noise levels + baseline explanation
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=COLORS_NOISE[n], marker=MARKERS_NOISE[n], label=f'$\\sigma={n}$')
           for n in NOISE_STDS]
handles.append(Line2D([0], [0], color='gray', ls='--', lw=0.8, alpha=0.6, label='1-NN baseline'))
fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.08),
           frameon=False, fontsize=8)

plt.tight_layout()
out_path = 'results_all_sweep/fig_appendix_snr.pdf'
fig.savefig(out_path, bbox_inches='tight')
print(f"Saved: {out_path}")

# Also save PNG for preview
out_png = out_path.replace('.pdf', '.png')
fig.savefig(out_png, bbox_inches='tight')
print(f"Saved: {out_png}")
