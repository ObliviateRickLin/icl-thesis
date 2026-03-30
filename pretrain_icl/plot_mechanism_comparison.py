#!/usr/bin/env python3
"""
Generate mechanism comparison figure: R² and correlation between LLM predictions
and various baselines (OLS, Ridge, KNN-k, Mean-y) across dimensions.
Nemotron-120B on NLR task.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linewidth': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
})

with open('results_all_sweep/mechanism_baseline_comparison.json') as f:
    data = json.load(f)

# ── Panel layout: 2 rows x 3 cols ──
# Row 1: Correlation with LLM predictions (left: NLR across d, middle: best baseline by d, right: nonlinear tasks)
# Row 2: RMSE comparison

# Extract NLR data across dimensions
dims = [1, 2, 3, 5, 10]
dim_keys = ['d1_L20', 'd2_L20', 'd3_L18', 'd5_L20', 'd10_L40']

# Baselines to plot
knn_ks = [1, 2, 3, 4, 5]
ridge_lambdas = [0.1, 1.0, 10.0]

# Colors
colors_knn = {1: '#e41a1c', 2: '#ff7f00', 3: '#fdbf6f', 4: '#b2df8a', 5: '#33a02c'}
colors_ridge = {0.1: '#1f78b4', 1.0: '#6a3d9a', 10.0: '#a6cee3'}
color_ols = '#000000'
color_meany = '#999999'

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

# ═══════════════════════════════════════════
# Panel (0,0): Correlation - KNN variants across d
# ═══════════════════════════════════════════
ax = axes[0, 0]
for k in knn_ks:
    corrs = []
    for dk in dim_keys:
        r = data[dk]
        key = f"{k}-NN"
        corrs.append(r[key]["corr"] if key in r else np.nan)
    ax.plot(dims, corrs, 'o-', color=colors_knn[k], label=f'{k}-NN', markersize=5, linewidth=1.3)
ax.set_xlabel('Dimension $d$')
ax.set_ylabel('Correlation with LLM')
ax.set_title('(a) KNN Correlation vs Dimension', fontweight='bold')
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#ccc')
ax.set_xticks(dims)
ax.set_ylim(0, 1.05)

# ═══════════════════════════════════════════
# Panel (0,1): Correlation - Ridge/OLS across d
# ═══════════════════════════════════════════
ax = axes[0, 1]
# OLS
corrs_ols = [data[dk].get("OLS", {}).get("corr", np.nan) for dk in dim_keys]
ax.plot(dims, corrs_ols, 's-', color=color_ols, label='OLS', markersize=5, linewidth=1.3)
# Ridge
for lam in ridge_lambdas:
    corrs = [data[dk].get(f"Ridge({lam})", {}).get("corr", np.nan) for dk in dim_keys]
    ax.plot(dims, corrs, 'D-', color=colors_ridge[lam], label=f'Ridge($\\lambda$={lam})',
            markersize=4, linewidth=1.3)
# Mean-y
corrs_mean = [data[dk].get("Mean-y", {}).get("corr", np.nan) for dk in dim_keys]
ax.plot(dims, corrs_mean, 'v--', color=color_meany, label='Mean-$y$', markersize=5, linewidth=1.0)

ax.set_xlabel('Dimension $d$')
ax.set_ylabel('Correlation with LLM')
ax.set_title('(b) Ridge/OLS Correlation vs Dimension', fontweight='bold')
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#ccc')
ax.set_xticks(dims)
ax.set_ylim(0, 1.05)

# ═══════════════════════════════════════════
# Panel (0,2): Best R² by baseline type across d
# ═══════════════════════════════════════════
ax = axes[0, 2]
# For each d, find best KNN and best Ridge
best_knn_r2 = []
best_knn_label = []
best_ridge_r2 = []
best_ridge_label = []
ols_r2 = []
meany_r2 = []

for dk in dim_keys:
    r = data[dk]
    # Best KNN
    best_k, best_v = 1, 0
    for k in range(1, 11):
        key = f"{k}-NN"
        if key in r and r[key]["r2"] > best_v:
            best_k, best_v = k, r[key]["r2"]
    best_knn_r2.append(best_v)
    best_knn_label.append(str(best_k))

    # Best Ridge
    best_l, best_v = 0.1, 0
    for lam in [0.01, 0.1, 1.0, 10.0]:
        key = f"Ridge({lam})"
        if key in r and r[key]["r2"] > best_v:
            best_l, best_v = lam, r[key]["r2"]
    best_ridge_r2.append(best_v)
    best_ridge_label.append(f'{best_l}')

    ols_r2.append(r.get("OLS", {}).get("r2", 0))
    meany_r2.append(r.get("Mean-y", {}).get("r2", 0))

x = np.arange(len(dims))
w = 0.2
ax.bar(x - 1.5*w, best_knn_r2, w, label='Best KNN', color='#33a02c', alpha=0.8)
ax.bar(x - 0.5*w, best_ridge_r2, w, label='Best Ridge', color='#6a3d9a', alpha=0.8)
ax.bar(x + 0.5*w, ols_r2, w, label='OLS', color='#000000', alpha=0.8)
ax.bar(x + 1.5*w, meany_r2, w, label='Mean-$y$', color='#999999', alpha=0.8)

# Annotate best K and lambda
for i, (kl, rl) in enumerate(zip(best_knn_label, best_ridge_label)):
    if best_knn_r2[i] > 0.02:
        ax.text(x[i] - 1.5*w, best_knn_r2[i] + 0.02, f'k={kl}', ha='center', fontsize=6, color='#33a02c')
    if best_ridge_r2[i] > 0.02:
        ax.text(x[i] - 0.5*w, best_ridge_r2[i] + 0.02, f'$\\lambda$={rl}', ha='center', fontsize=6, color='#6a3d9a')

ax.set_xlabel('Dimension $d$')
ax.set_ylabel('$R^2$ (LLM ~ Baseline)')
ax.set_title('(c) Best $R^2$ by Baseline Type', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'd={d}' for d in dims])
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#ccc')
ax.set_ylim(0, 1.15)

# ═══════════════════════════════════════════
# Panel (1,0): RMSE comparison - KNN
# ═══════════════════════════════════════════
ax = axes[1, 0]
# LLM RMSE
llm_rmse = [data[dk]["llm_rmse"] for dk in dim_keys]
ax.plot(dims, llm_rmse, 'k*-', label='LLM (Nemotron-120B)', markersize=10, linewidth=2, zorder=5)
for k in knn_ks:
    rmses = [data[dk].get(f"{k}-NN", {}).get("rmse", np.nan) for dk in dim_keys]
    ax.plot(dims, rmses, 'o--', color=colors_knn[k], label=f'{k}-NN', markersize=4, linewidth=1.0, alpha=0.7)
ax.set_xlabel('Dimension $d$')
ax.set_ylabel('RMSE')
ax.set_title('(d) RMSE: LLM vs KNN Baselines', fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#ccc', ncol=2)
ax.set_xticks(dims)

# ═══════════════════════════════════════════
# Panel (1,1): RMSE comparison - Ridge/OLS
# ═══════════════════════════════════════════
ax = axes[1, 1]
ax.plot(dims, llm_rmse, 'k*-', label='LLM (Nemotron-120B)', markersize=10, linewidth=2, zorder=5)
rmses_ols = [data[dk].get("OLS", {}).get("rmse", np.nan) for dk in dim_keys]
ax.plot(dims, rmses_ols, 's--', color=color_ols, label='OLS', markersize=4, linewidth=1.0, alpha=0.7)
for lam in ridge_lambdas:
    rmses = [data[dk].get(f"Ridge({lam})", {}).get("rmse", np.nan) for dk in dim_keys]
    ax.plot(dims, rmses, 'D--', color=colors_ridge[lam], label=f'Ridge($\\lambda$={lam})',
            markersize=4, linewidth=1.0, alpha=0.7)
rmses_mean = [data[dk].get("Mean-y", {}).get("rmse", np.nan) for dk in dim_keys]
ax.plot(dims, rmses_mean, 'v--', color=color_meany, label='Mean-$y$', markersize=4, linewidth=1.0, alpha=0.7)
ax.set_xlabel('Dimension $d$')
ax.set_ylabel('RMSE')
ax.set_title('(e) RMSE: LLM vs Ridge/OLS Baselines', fontweight='bold')
ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#ccc')
ax.set_xticks(dims)

# ═══════════════════════════════════════════
# Panel (1,2): Nonlinear tasks at d=5
# ═══════════════════════════════════════════
ax = axes[1, 2]
task_keys = ['d5_L20', 'NQR_d5_L20', '2NN_d5_L20']
task_labels = ['NLR', 'NQR', '2NN']
baselines_to_show = ['1-NN', '3-NN', '5-NN', 'OLS', 'Ridge(1.0)', 'Mean-y']
baseline_colors = {'1-NN': '#e41a1c', '3-NN': '#fdbf6f', '5-NN': '#33a02c',
                   'OLS': '#000000', 'Ridge(1.0)': '#6a3d9a', 'Mean-y': '#999999'}

x = np.arange(len(task_labels))
w = 0.12
# LLM bar
llm_vals = [data[tk]["llm_rmse"] for tk in task_keys]

for i, (bl_name, bl_color) in enumerate(baseline_colors.items()):
    vals = []
    for tk in task_keys:
        r = data[tk]
        corr = r.get(bl_name, {}).get("corr", 0)
        vals.append(corr)
    offset = (i - 2.5) * w
    ax.bar(x + offset, vals, w, label=bl_name, color=bl_color, alpha=0.8)

ax.set_xlabel('Task')
ax.set_ylabel('Correlation with LLM')
ax.set_title('(f) Baseline Correlation at $d=5$ (Nonlinear Tasks)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#ccc', fontsize=7, ncol=2)
ax.set_ylim(0, 1.0)

plt.tight_layout(h_pad=2.5)

# Save
out_path = 'results_all_sweep/fig_mechanism_comparison.pdf'
fig.savefig(out_path)
fig.savefig(out_path.replace('.pdf', '.png'))
print(f"Saved: {out_path}")
