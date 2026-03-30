#!/usr/bin/env python3
"""Generate appendix figures — v6: split large grids into low-dim and high-dim pages."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
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

df = pd.read_csv('results_all_sweep/all_sweep_summary.csv')
bl = pd.read_csv('results_all_sweep/baselines.csv')
ci = pd.read_csv('results_all_sweep/all_sweep_with_ci.csv')

TASK_LABELS = {'NLR': 'Noisy Linear', 'NQR': 'Noisy Quadratic',
               '2NN': 'ReLU 2NN', 'DT': 'Decision Tree'}
TASK_ORDER = ['NLR', 'NQR', '2NN', 'DT']

DIMS_LOW = [1, 2, 3, 4, 5, 8, 10]
DIMS_HIGH = [20, 40, 100]
ALL_DIMS = DIMS_LOW + DIMS_HIGH

DIM_L_CONFIGS = {
    1:   [1, 2, 4, 6, 10, 20, 40],
    2:   [1, 2, 4, 8, 12, 20, 40],
    3:   [2, 3, 6, 12, 18, 30, 60],
    4:   [2, 4, 8, 16, 24, 40, 80],
    5:   [3, 5, 10, 20, 30, 50, 100],
    8:   [4, 8, 16, 32, 48, 80],
    10:  [5, 10, 20, 40, 60, 100],
    20:  [10, 20, 40, 80, 120, 200],
    40:  [20, 40, 60, 80, 120, 160, 240],
    100: [10, 20, 30, 40, 50, 75, 100],
}

COLORS_CROSS = {
    'Qwen3-32B': '#1f77b4', 'Nemotron-120B': '#d62728', 'Mistral-Large': '#2ca02c',
}
MARKERS_CROSS = {
    'Qwen3-32B': 'o', 'Nemotron-120B': 's', 'Mistral-Large': '^',
}
COLORS_QWEN = {
    'Qwen3-0.6B': '#999999', 'Qwen3-8B': '#e6ab02',
    'Qwen3-14B': '#e7298a', 'Qwen3-32B': '#1f77b4',
}
MARKERS_QWEN = {
    'Qwen3-0.6B': 'v', 'Qwen3-8B': 'D', 'Qwen3-14B': '^', 'Qwen3-32B': 'o',
}

BL_STYLES = {
    'knn1_rmse':  {'color': '#000000', 'ls': '--', 'lw': 1.5, 'label': '1-NN'},
    'meany_rmse': {'color': '#555555', 'ls': '-.', 'lw': 1.5, 'label': 'Mean-$y$'},
}


def _setup_log_xaxis(ax, Ls):
    ax.set_xscale('log')
    all_L = sorted(set(int(l) for l in Ls))
    if len(all_L) > 6:
        idx = np.linspace(0, len(all_L)-1, 5, dtype=int)
        ticks = [all_L[i] for i in idx]
    else:
        ticks = all_L
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis='x', which='minor', length=0)


def _plot_baselines(ax, task, d):
    sub = bl[(bl['task'] == task) & (bl['d'] == d)].sort_values('L')
    if len(sub) == 0: return
    for col, style in BL_STYLES.items():
        vals = sub[['L', col]].dropna()
        if len(vals) >= 2:
            ax.plot(vals['L'], vals[col], color=style['color'],
                   linestyle=style['ls'], linewidth=style['lw'], zorder=1)
    # Baseline CI bands from ci data (use any model's baseline, they're data-dependent)
    ci_bl = ci[(ci['task'] == task) & (ci['d'] == d)].drop_duplicates(subset=['L']).sort_values('L')
    if len(ci_bl) >= 2:
        for bl_key, ci_cols, style in [
            ('knn1_rmse', ('knn1_ci_lo', 'knn1_ci_hi'), BL_STYLES['knn1_rmse']),
            ('meany_rmse', ('meany_ci_lo', 'meany_ci_hi'), BL_STYLES['meany_rmse']),
        ]:
            bl_ci = ci_bl[['L', ci_cols[0], ci_cols[1]]].dropna()
            if len(bl_ci) >= 2:
                ax.fill_between(bl_ci['L'], bl_ci[ci_cols[0]], bl_ci[ci_cols[1]],
                               color=style['color'], alpha=0.10, zorder=0, linewidth=0)


def _plot_model_with_overflow(ax, model_df, expected_Ls, color, marker,
                              model_name=None, task=None, d=None):
    sub = model_df.sort_values('L')
    if len(sub) == 0: return
    ax.plot(sub['L'], sub['rmse'], color=color, marker=marker,
           markersize=4, linewidth=1.4,
           markeredgecolor='white', markeredgewidth=0.3, zorder=3)
    # CI band from ci dataframe
    if model_name and task and d is not None:
        ci_sub = ci[(ci['model'] == model_name) & (ci['task'] == task) & (ci['d'] == d)].sort_values('L')
        ci_sub = ci_sub.dropna(subset=['ci_lo', 'ci_hi'])
        if len(ci_sub) >= 2:
            ax.fill_between(ci_sub['L'], ci_sub['ci_lo'], ci_sub['ci_hi'],
                           color=color, alpha=0.12, zorder=1, linewidth=0)
    # Overflow markers
    available = set(sub['L'].values)
    missing = [L for L in expected_Ls if L not in available]
    if not missing or len(sub) < 2: return
    known_L, known_y = sub['L'].values, sub['rmse'].values
    for mL in missing:
        if mL < known_L.min():
            y_est = known_y[0]
        elif mL > known_L.max():
            y_est = known_y[-1]
        else:
            y_est = np.interp(np.log(mL), np.log(known_L), known_y)
        ax.plot(mL, y_est, marker='x', color=color, markersize=5,
               markeredgewidth=1.2, linestyle='none', zorder=2, alpha=0.5)


def _make_legend_handles(model_colors, model_markers, model_list):
    model_handles = [Line2D([0], [0], color=model_colors[m], marker=model_markers[m],
                           markersize=6, linewidth=1.4, markeredgecolor='white',
                           markeredgewidth=0.3, label=m) for m in model_list]
    bl_handles = [Line2D([0], [0], color=s['color'], linestyle=s['ls'],
                        linewidth=s['lw'], label=s['label']) for s in BL_STYLES.values()]
    overflow_handle = Line2D([0], [0], marker='x', color='gray', linestyle='none',
                            markersize=6, markeredgewidth=1.2, alpha=0.5,
                            label='Missing (overflow)')
    return model_handles + bl_handles + [overflow_handle]


def _plot_grid(dims, models, model_colors, model_markers, title, outname,
               row_height=2.2):
    """Generic grid plotter: rows=dims, cols=tasks."""
    n_rows, n_cols = len(dims), len(TASK_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(13, row_height * n_rows + 1.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, d in enumerate(dims):
        for col, task in enumerate(TASK_ORDER):
            ax = axes[row, col]
            expected = DIM_L_CONFIGS.get(d, [])
            _plot_baselines(ax, task, d)
            all_Ls = list(expected)
            for model in models:
                sub = df[(df['model'] == model) & (df['task'] == task) & (df['d'] == d)]
                all_Ls.extend(sub['L'].tolist())
                _plot_model_with_overflow(ax, sub, expected,
                                         model_colors[model], model_markers[model],
                                         model_name=model, task=task, d=d)
            if row == 0:
                ax.set_title(TASK_LABELS[task], fontweight='bold', pad=6)
            if col == 0:
                ax.set_ylabel(f'$d={d}$\nRMSE', fontweight='bold')
            if row == n_rows - 1:
                ax.set_xlabel('Context size $L$')
            if all_Ls:
                _setup_log_xaxis(ax, all_Ls)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(max(0, ymin * 0.9), ymax * 1.08)

    handles = _make_legend_handles(model_colors, model_markers, models)
    fig.legend(handles=handles, loc='lower center',
              ncol=len(handles), frameon=True, fancybox=False,
              edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.01),
              fontsize=9, handlelength=2.5, columnspacing=2)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.003)
    plt.tight_layout(rect=[0, 0.02, 1, 0.995])
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    fig.savefig(f'results_all_sweep/{outname}.pdf')
    fig.savefig(f'results_all_sweep/{outname}.png')
    plt.close()
    print(f'  {outname} saved.')


# ═══════════════════════════════════════════════════════════
# Figure 1a/1b: ICL Curves — 3 models, split low/high dim
# ═══════════════════════════════════════════════════════════
def plot_fig1():
    models = ['Qwen3-32B', 'Nemotron-120B', 'Mistral-Large']
    print('Figure 1 (ICL Curves):')
    _plot_grid(DIMS_LOW, models, COLORS_CROSS, MARKERS_CROSS,
              'ICL Curves: RMSE vs. Context Size $L$ ($d = 1$--$10$)',
              'fig_appendix_icl_curves_low')
    _plot_grid(DIMS_HIGH, models, COLORS_CROSS, MARKERS_CROSS,
              'ICL Curves: RMSE vs. Context Size $L$ ($d = 20$--$100$)',
              'fig_appendix_icl_curves_high')


# ═══════════════════════════════════════════════════════════
# Figure 2: Dimension Effect (unchanged — 1×4)
# ═══════════════════════════════════════════════════════════
def plot_fig2():
    models = ['Qwen3-32B', 'Nemotron-120B', 'Mistral-Large']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    for col, task in enumerate(TASK_ORDER):
        ax = axes[col]
        for bl_col, style in BL_STYLES.items():
            points = []
            for d in sorted(bl['d'].unique()):
                sub = bl[(bl['task'] == task) & (bl['d'] == d)]
                if len(sub) == 0: continue
                target_L = 4 * d
                best_idx = (sub['L'] - target_L).abs().idxmin()
                val = sub.loc[best_idx, bl_col]
                if pd.notna(val) and sub.loc[best_idx, 'L'] / d >= 1.5:
                    points.append((d, val))
            if points:
                ds, vs = zip(*points)
                ax.plot(ds, vs, color=style['color'], linestyle=style['ls'],
                       linewidth=style['lw'], zorder=1)
        for model in models:
            sub = df[(df['model'] == model) & (df['task'] == task)]
            points = []
            ci_points = []
            for d in sorted(sub['d'].unique()):
                d_sub = sub[sub['d'] == d]
                if len(d_sub) == 0: continue
                target_L = 4 * d
                best_idx = (d_sub['L'] - target_L).abs().idxmin()
                row = d_sub.loc[best_idx]
                if row['L'] / d >= 1.5:
                    points.append((d, row['rmse']))
                    # Get CI
                    ci_row = ci[(ci['model'] == model) & (ci['task'] == task) &
                               (ci['d'] == d) & (ci['L'] == row['L'])]
                    if len(ci_row) > 0 and pd.notna(ci_row.iloc[0]['ci_lo']):
                        ci_points.append((d, ci_row.iloc[0]['ci_lo'], ci_row.iloc[0]['ci_hi']))
            if points:
                ds, rmses = zip(*points)
                ax.plot(ds, rmses, color=COLORS_CROSS[model],
                       marker=MARKERS_CROSS[model], label=model,
                       markersize=5, linewidth=1.4,
                       markeredgecolor='white', markeredgewidth=0.3, zorder=3)
                if ci_points:
                    ci_ds, ci_los, ci_his = zip(*ci_points)
                    ax.fill_between(ci_ds, ci_los, ci_his,
                                   color=COLORS_CROSS[model], alpha=0.12, zorder=1, linewidth=0)
        ax.set_title(TASK_LABELS[task], fontweight='bold')
        ax.set_xlabel('Dimension $d$')
        if col == 0: ax.set_ylabel('RMSE')
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 40, 100])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis='x', which='minor', length=0)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.08)
    model_handles = [Line2D([0], [0], color=COLORS_CROSS[m], marker=MARKERS_CROSS[m],
                           markersize=5, linewidth=1.4, markeredgecolor='white',
                           markeredgewidth=0.3, label=m) for m in models]
    bl_handles = [Line2D([0], [0], color=s['color'], linestyle=s['ls'],
                        linewidth=s['lw'], label=s['label']) for s in BL_STYLES.values()]
    fig.legend(handles=model_handles + bl_handles, loc='lower center',
              ncol=len(models) + len(BL_STYLES),
              frameon=True, fancybox=False, edgecolor='#cccccc',
              bbox_to_anchor=(0.5, -0.1), fontsize=9,
              handlelength=2.5, columnspacing=2.5)
    fig.suptitle('Dimension Effect: RMSE vs. Input Dimension $d$ (at $L \\approx 4d$)',
                fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    fig.savefig('results_all_sweep/fig_appendix_dim_effect.pdf')
    fig.savefig('results_all_sweep/fig_appendix_dim_effect.png')
    plt.close()
    print('Figure 2 saved.')


# ═══════════════════════════════════════════════════════════
# Figure 3a/3b: Scale Effect — all Qwen, split low/high dim
# ═══════════════════════════════════════════════════════════
def plot_fig3():
    qwen_models = ['Qwen3-0.6B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen3-32B']
    print('Figure 3 (Scale Effect):')
    _plot_grid(DIMS_LOW, qwen_models, COLORS_QWEN, MARKERS_QWEN,
              'Qwen3 Scale Effect ($d = 1$--$10$)',
              'fig_appendix_scale_low')
    _plot_grid(DIMS_HIGH, qwen_models, COLORS_QWEN, MARKERS_QWEN,
              'Qwen3 Scale Effect ($d = 20$--$100$)',
              'fig_appendix_scale_high')


if __name__ == '__main__':
    plot_fig1()
    plot_fig2()
    plot_fig3()
    print('All figures done.')
