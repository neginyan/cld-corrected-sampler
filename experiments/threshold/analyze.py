"""Analysis of stage B1: theory curve, threshold fit, figures.  -> results/threshold/summary_B1.json, figures/*.png"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
import json, glob
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cld_mixture import *

RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.2, 1.5, 2.0]
SEEDS = [0, 1, 2, 3, 4]
COL = {'naive': '#eb6834', 'fixed': '#2a78d6', 'ode': '#1baf7a', 'theory': '#6b6a64'}
LAB = {'naive': '(i) clipped SDE', 'fixed': '(ii) corrected SDE', 'ode': '(iii) probability-flow ODE'}
rows = {(rr, s): json.load(open(f'main_r{rr}_s{s}.json')) for rr in RATIOS for s in SEEDS}

# ---------------------------------------------------------------- theory (Gaussian-moment, one component)
grid = np.round(np.linspace(0.3, 1.1, 81), 4)
th_grid = np.array([[np.log(x) for x in theory_within_ratio(g)] for g in grid])      # (log pos, log vel)
th_at = {rr: [float(np.log(x)) for x in theory_within_ratio(rr)] for rr in RATIOS}
f_th = lambda x: np.interp(x, grid, th_grid[:, 0], right=0.0)                       # zero above 1.1 too

def agg(fn):
    a = np.array([[fn(rows[(rr, s)]) for s in SEEDS] for rr in RATIOS])
    return a.mean(1), a.std(1, ddof=1) / np.sqrt(len(SEEDS)), a

summ = {'kappa': KAPPA, 'ratios': RATIOS, 'seeds': SEEDS, 'theory_log_within': {str(r): th_at[r][0] for r in RATIOS},
        'theory_log_vel': {str(r): th_at[r][1] for r in RATIOS}}
for key, fn in [('paired_log_within', lambda d: d['extrap']['paired_log_within']),
                ('paired_log_within_fine', lambda d: d['fine']['paired_log_within']),
                ('paired_log_within_coarse', lambda d: d['coarse']['paired_log_within']),
                ('paired_log_vel', lambda d: d['extrap']['paired_log_vel'])]:
    m, se, a = agg(fn); summ[key] = {'mean': m.tolist(), 'se': se.tolist(), 'all': a.tolist()}
for md in ['naive', 'fixed', 'ode']:
    for f in ['within_ratio', 'vel_ratio']:
        m, se, a = agg(lambda d: np.log(d['extrap'][md][f])); summ[f'log_{f}_{md}'] = {'mean': m.tolist(), 'se': se.tolist(), 'all': a.tolist()}
    m, se, a = agg(lambda d: d['fine'][md]['mmd2']); summ[f'mmd2_{md}'] = {'mean': m.tolist(), 'se': se.tolist()}
    m, se, a = agg(lambda d: d['fine'][md]['max_weight_dev']); summ[f'max_weight_dev_{md}'] = {'mean': m.tolist(), 'max': a.max(1).tolist()}
m, se, _ = agg(lambda d: d['mmd2_truth_vs_truth']); summ['mmd2_floor'] = {'mean': m.tolist(), 'se': se.tolist()}

# ---------------------------------------------------------------- threshold estimate
below = [i for i, r in enumerate(RATIOS) if r < 1]
def fit_shift(y, se):
    """y(r) = A * f_th(r + 1 - c): theory curve with free amplitude A and free threshold c (in units of kappa)."""
    x = np.array(RATIOS)[below]; yy = np.array(y)[below]; ss = np.array(se)[below]
    res = least_squares(lambda p: (p[0] * f_th(x + 1 - p[1]) - yy) / ss, [1.0, 1.0])
    J = res.jac; cov = np.linalg.inv(J.T @ J) * max(1.0, 2 * res.cost / (len(x) - 2))
    return res.x, np.sqrt(np.diag(cov)), 2 * res.cost / (len(x) - 2)
def fit_poly(y, se):
    """Model-free: quadratic in r through the r < kappa points, threshold = its zero next to the data."""
    x = np.array(RATIOS)[below]; yy = np.array(y)[below]; ss = np.array(se)[below]
    c = np.polyfit(x, yy, 2, w=1 / ss); rts = np.roots(c); rts = rts[np.isreal(rts)].real
    return float(rts[np.argmin(np.abs(rts - 1))]) if len(rts) else np.nan
y_all = np.array(summ['paired_log_within']['all']); P = summ['paired_log_within']
(A, c), (sA, sc_), chi2 = fit_shift(P['mean'], P['se'])
rng = np.random.default_rng(0); boot_c, boot_A, boot_p = [], [], []
for _ in range(2000):
    idx = rng.integers(0, len(SEEDS), len(SEEDS)); yb = y_all[:, idx]
    seb = np.maximum(yb.std(1, ddof=1) / np.sqrt(len(SEEDS)), 1e-5)
    (Ab, cb), _, _ = fit_shift(yb.mean(1), seb); boot_c.append(cb); boot_A.append(Ab); boot_p.append(fit_poly(yb.mean(1), seb))
summ['threshold_fit'] = {'model': 'paired log within-mode variance = A * theory(r + kappa - kappa_fit), r < kappa points',
    'A': A, 'A_se': sA, 'kappa_fit_over_kappa': c, 'kappa_fit_se': sc_, 'reduced_chi2': chi2,
    'boot_kappa_fit_16_50_84': np.percentile(boot_c, [16, 50, 84]).tolist(), 'boot_A_16_50_84': np.percentile(boot_A, [16, 50, 84]).tolist(),
    'poly_kappa_fit': fit_poly(P['mean'], P['se']), 'boot_poly_16_50_84': np.nanpercentile(boot_p, [16, 50, 84]).tolist()}
json.dump(summ, open('summary_B1.json', 'w'), indent=1)

# ---------------------------------------------------------------- figures
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999',
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
x = np.array(RATIOS)
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8))
a = ax[0]
a.plot(grid, th_grid[:, 0], '--', color=COL['theory'], lw=1.5, label='theory (Gaussian moments, one mode)')
xx = np.linspace(0.45, 1.1, 200); a.plot(xx, A * f_th(xx + 1 - c), '-', color=COL['naive'], lw=1, alpha=0.5, label=f'fit: A={A:.2f}, κ_fit/κ={c:.3f}±{sc_:.3f}')
a.errorbar(x, P['mean'], yerr=P['se'], fmt='o', ms=4.5, color=COL['naive'], mec='white', mew=0.8, capsize=2, label='measured (i) vs (ii), paired')
a.axvline(1, color='#bbb', lw=1); a.axhline(0, color='#bbb', lw=0.8)
a.set_xlabel('r / κ'); a.set_ylabel('log[ var_within (i) / var_within (ii) ]'); a.set_title('Within-mode shrinkage of the clipped SDE', loc='left')
a.legend(frameon=False, fontsize=7.5, loc='lower right')
a = ax[1]
for md, off in [('naive', -0.008), ('fixed', 0), ('ode', 0.008)]:
    S_ = summ[f'log_within_ratio_{md}']
    a.errorbar(x + off, S_['mean'], yerr=S_['se'], fmt='o-', ms=4, lw=1.2, color=COL[md], mec='white', mew=0.8, capsize=2, label=LAB[md])
a.plot(grid, th_grid[:, 0], '--', color=COL['theory'], lw=1.2)
a.axvline(1, color='#bbb', lw=1); a.axhline(0, color='#bbb', lw=0.8)
a.set_xlabel('r / κ'); a.set_ylabel('log[ var_within / var_within(target) ]'); a.set_title('Against the σ_min-blurred data (step-extrapolated)', loc='left')
a.legend(frameon=False, fontsize=7.5, loc='lower right')
a = ax[2]
for md in ['naive', 'fixed', 'ode']:
    S_ = summ[f'mmd2_{md}']; a.errorbar(x, S_['mean'], yerr=S_['se'], fmt='o-', ms=4, lw=1.2, color=COL[md], mec='white', mew=0.8, capsize=2, label=LAB[md])
F = summ['mmd2_floor']; a.fill_between(x, np.array(F['mean']) - 2 * np.array(F['se']), np.array(F['mean']) + 2 * np.array(F['se']), color='#ddd', label='target vs target (noise floor)')
a.axvline(1, color='#bbb', lw=1); a.set_xlabel('r / κ'); a.set_ylabel('MMD² (positions, 4000 vs 4000)'); a.set_title('Distance to the data', loc='left')
a.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('figures/B1_threshold.png', dpi=160)

fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))
w, mu, S = mixture()
for a, md in zip(ax, ['naive', 'fixed', 'ode']):
    z = np.concatenate([np.load(f'main_r0.5_s{s}_{md}.npy') for s in SEEDS])
    d = z[:, POS] - mu[np.argmin(((z[:, None, POS] - mu[None][:, :, POS])**2).sum(-1), 1)][:, POS]
    a.hist2d(d[:, 0], d[:, 1], bins=60, range=[[-0.3, 0.3], [-0.3, 0.3]], cmap='Blues')
    sd = np.sqrt((d**2).mean(0).mean()); a.set_title(f'{LAB[md]}\nstd per coord = {sd:.4f}', loc='left', fontsize=8.5)
    a.set_aspect('equal'); a.grid(False); a.set_xlabel('x − mode centre')
ax[0].set_ylabel('y − mode centre')
fig.suptitle(f'r/κ = 0.5: positions relative to their mode, all 8 modes stacked (target std {np.sqrt(S_POS**2 + S_MIN**2):.4f})', fontsize=9)
fig.tight_layout(); fig.savefig('figures/B1_modes_r0.5.png', dpi=160)

fig, ax = plt.subplots(figsize=(4.6, 3.6))
Pv = summ['paired_log_vel']
ax.plot(grid, th_grid[:, 1], '--', color=COL['theory'], lw=1.5, label='theory (one mode)')
ax.errorbar(x, Pv['mean'], yerr=Pv['se'], fmt='o', ms=4.5, color=COL['naive'], mec='white', capsize=2, label='measured (i) vs (ii), paired')
ax.axvline(1, color='#bbb', lw=1); ax.axhline(0, color='#bbb', lw=0.8)
ax.set_xlabel('r / κ'); ax.set_ylabel('log[ var_v (i) / var_v (ii) ]'); ax.set_title('Velocity (auxiliary variable)', loc='left'); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('figures/B1_velocity.png', dpi=160)

# step-size study figure
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for rr, mk in [(0.5, 'o'), (0.9, 's')]:
    L = json.load(open(f'steps_r{rr}_s0.json'))['levels']; h = [l['h'] for l in L]
    for md in ['fixed', 'ode']:
        ax[0].plot(h, [np.log(l[md]['within_ratio']) for l in L], mk + '-', color=COL[md], ms=4, label=f'{LAB[md]}, r/κ={rr}')
    ax[1].plot(h, [l['paired_log_within'] for l in L], mk + '-', color=COL['naive'], ms=4, label=f'paired (i)−(ii), r/κ={rr}')
    ax[1].axhline(th_at[rr][0], ls='--', color=COL['theory'], lw=1)
ax[0].axhline(0, color='#bbb'); ax[0].set_xlabel('step h'); ax[0].set_ylabel('log within-var ratio to target'); ax[0].legend(frameon=False, fontsize=7)
ax[1].set_xlabel('step h'); ax[1].set_ylabel('paired log ratio'); ax[1].legend(frameon=False, fontsize=7)
ax[0].set_title('Step study (1 seed, N = 10000, coupled paths)', loc='left')
fig.tight_layout(); fig.savefig('figures/B1_steps.png', dpi=160)
print(json.dumps(summ['threshold_fit'], indent=1))
for i, rr in enumerate(RATIOS):
    print(rr, 'paired %.5f ± %.5f  theory %.5f' % (P['mean'][i], P['se'][i], th_at[rr][0]),
          ' | extrap logratio i/ii/iii', ' '.join('%.4f±%.4f' % (summ[f'log_within_ratio_{m}']['mean'][i], summ[f'log_within_ratio_{m}']['se'][i]) for m in ['naive', 'fixed', 'ode']),
          ' | mmd', ' '.join('%.2e' % summ[f'mmd2_{m}']['mean'][i] for m in ['naive', 'fixed', 'ode']), 'floor %.2e' % summ['mmd2_floor']['mean'][i],
          ' | wdev', ' '.join('%.4f' % summ[f'max_weight_dev_{m}']['max'][i] for m in ['naive', 'fixed', 'ode']))
