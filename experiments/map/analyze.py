"""Section 2-3 analysis and figures.  -> results/map/summary_B2.json, figures/B2_mixture.png, figures/B2_modes_<setting>.png"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from settings_abc import *

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999',
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
COL = {'naive': '#eb6834', 'fixed': '#2a78d6', 'ode': '#1baf7a'}
LAB = {'naive': '(i) clipped SDE', 'fixed': '(ii) corrected SDE', 'ode': '(iii) probability-flow ODE'}
DESC = {'A': 'A: M = 0.01, σ_min = s/16, r/κ = 0.2', 'B': 'B: M = 1/4 (CLD), σ_min = 1.78 s, r/κ = 0.3',
        'C': 'C: M = 10, σ_min = s/2, r/κ = 0.5'}
summ = {}
for name in 'ABC':
    R = [json.load(open(f)) for f in sorted(glob.glob(f'main_{name}_s*.json'))]; n = len(R)
    ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
    d = {'n_seeds': n, 'theory_log_within': R[0]['theory_log_within'], 'theory_log_vel': R[0]['theory_log_vel'],
         'beta': R[0]['beta'], 'T': R[0]['T'], 'h_fine': R[0]['h_fine']}
    for k in ['paired_log_within', 'paired_log_vel']:
        d[k] = ms([x['extrap'][k] for x in R]); d[k + '_fine'] = ms([x['fine'][k] for x in R]); d[k + '_coarse'] = ms([x['coarse'][k] for x in R])
    for m in ['naive', 'fixed', 'ode']:
        d[f'log_within_{m}'] = ms([x['extrap'][m]['log_within'] for x in R]); d[f'log_vel_{m}'] = ms([x['extrap'][m]['log_vel'] for x in R])
        d[f'mmd2_{m}'] = ms([x['fine'][m]['mmd2'] for x in R])
        d[f'max_weight_dev_{m}'] = float(max(x['fine'][m]['max_weight_dev'] for x in R))
    d['ratio_to_theory'] = d['paired_log_within'][0] / d['theory_log_within']
    d['percent_change_paired'] = 100 * (np.exp(d['paired_log_within'][0]) - 1)
    summ[name] = d
json.dump(summ, open('summary_B2.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
xs = np.arange(3)
for j, m in enumerate(['naive', 'fixed', 'ode']):
    mean = [100 * (np.exp(summ[s][f'log_within_{m}'][0]) - 1) for s in 'ABC']
    se = [100 * summ[s][f'log_within_{m}'][1] for s in 'ABC']
    ax[0].bar(xs + (j - 1) * 0.26, mean, 0.24, yerr=se, color=COL[m], label=LAB[m], capsize=2)
ax[0].scatter(xs - 0.26, [100 * (np.exp(summ[s]['theory_log_within']) - 1) for s in 'ABC'], marker='_', s=300, color='#333', zorder=5, label='one-mode theory for (i)')
ax[0].axhline(-10, color='#b8483c', lw=1, ls='--'); ax[0].axhline(0, color='#999', lw=0.8)
ax[0].set_xticks(xs, ['A', 'B', 'C']); ax[0].set_ylabel('within-mode position variance vs target (%)')
ax[0].set_title('8-mode mixture, exact score (5 seeds, step-extrapolated)', loc='left'); ax[0].legend(frameon=False, fontsize=7, loc='lower left')
for j, m in enumerate(['naive', 'fixed', 'ode']):
    mean = [100 * (np.exp(summ[s][f'log_vel_{m}'][0]) - 1) for s in 'ABC']; se = [100 * summ[s][f'log_vel_{m}'][1] for s in 'ABC']
    ax[1].bar(xs + (j - 1) * 0.26, mean, 0.24, yerr=se, color=COL[m], capsize=2)
ax[1].axhline(0, color='#999', lw=0.8); ax[1].set_xticks(xs, ['A', 'B', 'C']); ax[1].set_ylabel('velocity variance vs target (%)')
ax[1].set_title('Velocity (auxiliary variable)', loc='left')
fig.text(0.01, -0.02, '   '.join(DESC.values()), fontsize=7.5, color='#555')
fig.tight_layout(); fig.savefig('figures/B2_mixture.png', dpi=160, bbox_inches='tight')

w, mu, S = mixture()
for name in 'ABC':
    p, _, _ = setting(name); tgt = np.sqrt(S_POS**2 + p.s0**2)
    fig = plt.figure(figsize=(12, 7.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    radial = {}
    for j, m in enumerate(['naive', 'fixed', 'ode']):
        z = np.concatenate([np.load(f) for f in sorted(glob.glob(f'main_{name}_s*_{m}.npy'))])
        x = z[:, POS]; lab = np.argmin(((x[:, None] - mu[None][:, :, POS])**2).sum(-1), 1); dlt = x - mu[lab][:, POS]
        a = fig.add_subplot(gs[0, j]); lim = 4 * tgt
        a.hist2d(dlt[:, 0], dlt[:, 1], bins=70, range=[[-lim, lim], [-lim, lim]], cmap='Blues')
        th = np.linspace(0, 2 * np.pi, 200); a.plot(tgt * np.cos(th), tgt * np.sin(th), color='#b8483c', lw=1, ls='--')
        sd = np.sqrt((dlt**2).mean()); a.set_title(f'{LAB[m]}\nstd per coord {sd:.4f} (target {tgt:.4f}, dashed)', loc='left', fontsize=8.5)
        a.set_aspect('equal'); a.grid(False); a.set_xlabel('x − mode centre')
        if j == 0: a.set_ylabel('y − mode centre')
        radial[m] = np.sqrt((dlt**2).sum(1)) / tgt
    a = fig.add_subplot(gs[1, :]); bins = np.linspace(0, 4, 61)
    tz, _ = sample_mixture(w, mu, S, 400000, np.random.default_rng(999), extra_var=p.s0**2)
    tl = np.argmin(((tz[:, None, POS] - mu[None][:, :, POS])**2).sum(-1), 1); rt = np.sqrt(((tz[:, POS] - mu[tl][:, POS])**2).sum(1)) / tgt
    a.hist(rt, bins, density=True, histtype='stepfilled', color='#e3e3e3', label='target (σ_min-blurred data)')
    for m in ['naive', 'fixed', 'ode']: a.hist(radial[m], bins, density=True, histtype='step', lw=1.6, color=COL[m], label=LAB[m])
    a.set_xlabel('distance from mode centre / target std'); a.set_ylabel('density'); a.legend(frameon=False, fontsize=7.5)
    fig.suptitle(DESC[name] + '  —  positions relative to their mode, 8 modes × 5 seeds stacked', fontsize=9.5)
    fig.tight_layout(); fig.savefig(f'figures/B2_modes_{name}.png', dpi=150)
for s in 'ABC':
    d = summ[s]; print(s, 'paired %.4f±%.4f (%.1f%%) theory %.4f ratio %.3f' % (*d['paired_log_within'], d['percent_change_paired'], d['theory_log_within'], d['ratio_to_theory']),
          '| i/ii/iii', ' '.join('%.4f±%.4f' % d[f'log_within_{m}'] for m in ['naive', 'fixed', 'ode']),
          '| vel paired %.4f th %.4f' % (d['paired_log_vel'][0], d['theory_log_vel']), '| wdev', [round(d[f'max_weight_dev_{m}'], 4) for m in ['naive', 'fixed', 'ode']],
          '| mmd', ['%.1e' % d[f'mmd2_{m}'][0] for m in ['naive', 'fixed', 'ode']])
