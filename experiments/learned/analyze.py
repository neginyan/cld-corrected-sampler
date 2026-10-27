"""B3 analysis and figures.  -> results/learned/summary_B3.json, figures/B3_*.png"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
WEIGHTING = os.environ.get('CLD_WEIGHTING', 'eps')            # 'eps' (Sec. V) or 'score' (Obs. 9)
STAGE = 'learned' if WEIGHTING == 'eps' else 'score_weighting'
WEIGHTS = os.path.join(ROOT, 'weights', 'learned_' + WEIGHTING)
os.chdir(os.path.join(ROOT, 'results', STAGE))   # relative paths below are inside results/<stage>/
import json, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from learned import SET, setting, mixture, POS, S_POS

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999',
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
COL = {'naive': '#eb6834', 'fixed': '#2a78d6', 'ode': '#1baf7a'}
LAB = {'naive': '(i) clipped SDE', 'fixed': '(ii) corrected SDE', 'ode': '(iii) probability-flow ODE'}
DESC = {'B': 'B: CLD mass, σ_min = 1.78 s, r/κ = 0.3', 'C': 'C: M = 10, σ_min = s/2, r/κ = 0.5', 'D': 'D: B1 default, σ_min = s/16, r/κ = 0.5'}
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))))
summ = {}
for name in 'BCD':
    R = [json.load(open(f)) for f in sorted(glob.glob(f'samp_{name}_n*_k300000_s*_N20000.json'))]
    d = {'n_runs': len(R)}
    for kind in ['learned', 'exact']:
        d[kind] = {'paired': ms([x[kind]['paired'] for x in R])}
        for m in ['naive', 'fixed', 'ode']:
            d[kind][m] = ms([x[kind][m]['L'] for x in R]); d[kind][m + '_vel'] = ms([x[kind][m]['L_vel'] for x in R])
            d[kind][m + '_wdev'] = float(max(x[kind][m]['max_weight_dev'] for x in R))
    d['delta'] = {m: ms([x['delta'][m] for x in R]) for m in ['naive', 'fixed', 'ode']}
    d['ratio_paired'] = d['learned']['paired'][0] / d['exact']['paired'][0]
    E = [json.load(open(f)) for f in sorted(glob.glob(f'err_{name}_n*_k300000.json'))]
    d['e_near_smin'] = [e['e_near_smin'] for e in E]; d['e_near_smin_pos'] = [e['e_near_smin_pos'] for e in E]
    summ[name] = d
ck = {}
for name in 'BC':
    rows = []
    for k in [10000, 40000, 100000, 300000]:
        x = json.load(open(f'samp_{name}_n1_k{k}_s0_N10000.json')); e = json.load(open(f'err_{name}_n1_k{k}.json'))
        rows.append({'step': k, 'e_near_smin': e['e_near_smin'], 'delta': x['delta'], 'paired_learned': x['learned']['paired'], 'paired_exact': x['exact']['paired']})
    ck[name] = rows
summ['checkpoints'] = ck
json.dump(summ, open('summary_B3.json', 'w'), indent=1)

# figure 1: score error by sigma
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for a, name in zip(ax, 'BCD'):
    p = setting(name)
    for f, ls_ in [(f'err_{name}_n1_k300000.json', '-'), (f'err_{name}_n2_k300000.json', '--')]:
        e = json.load(open(f)); sg = np.array([o['sigma'] for o in e['by_sigma']]) / S_POS
        a.plot(sg, [o['e_pos'] for o in e['by_sigma']], ls_, color='#2a78d6', lw=1.5, label='position' if ls_ == '-' else None)
        a.plot(sg, [o['e_vel'] for o in e['by_sigma']], ls_, color='#eb6834', lw=1.5, label='velocity' if ls_ == '-' else None)
    if name in 'BC':
        for k, c in [(10000, '#cde2fb'), (40000, '#9ec5f4'), (100000, '#6da7ec')]:
            e = json.load(open(f'err_{name}_n1_k{k}.json')); a.plot(sg, [o['e_all'] for o in e['by_sigma']], color=c, lw=1, label=f'all, step {k // 1000}k')
    a.axvspan(p.s0 / S_POS, 2 * p.s0 / S_POS, color='#eee', zorder=0)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('σ / s'); a.set_title(DESC[name], loc='left', fontsize=8.5)
ax[0].set_ylabel('relative score error E|ŝ − s|² / E|s|²'); ax[0].legend(frameon=False, fontsize=7)
fig.suptitle('Score error of the trained networks (solid: net 1, dashed: net 2; shaded: σ_min … 2σ_min)', fontsize=9)
fig.tight_layout(); fig.savefig('figures/B3_score_error.png', dpi=160)

# figure 2: learned vs exact, three samplers
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for a, name in zip(ax, 'BCD'):
    d = summ[name]; xs = np.arange(3)
    for j, kind in enumerate(['exact', 'learned']):
        mean = [100 * (np.exp(d[kind][m][0]) - 1) for m in ['naive', 'fixed', 'ode']]; se = [100 * d[kind][m][1] for m in ['naive', 'fixed', 'ode']]
        big = [abs(v) > 100 for v in mean]
        mean = [0 if b else v for v, b in zip(mean, big)]; se = [0 if b else v for v, b in zip(se, big)]
        a.bar(xs + (j - 0.5) * 0.38, mean, 0.36, yerr=se, capsize=2, color=[COL[m] for m in ['naive', 'fixed', 'ode']],
              alpha=0.45 if kind == 'exact' else 1.0, edgecolor='white')
        for k, b in enumerate(big):
            if b: a.text(xs[k] + (j - 0.5) * 0.38, 0.3, 'diverges\n(a few\ntrajectories)', ha='center', va='bottom', fontsize=7, color='#1baf7a')
    a.axhline(0, color='#999', lw=0.8); a.set_xticks(xs, ['(i)', '(ii)', '(iii)'])
    a.set_title(DESC[name] + '\nlight: exact score, dark: learned score', loc='left', fontsize=8.5)
ax[0].set_ylabel('within-mode position variance vs target (%)')
fig.tight_layout(); fig.savefig('figures/B3_samplers.png', dpi=160)

# figure 3: delta vs score error over checkpoints
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for a, name in zip(ax, 'BC'):
    rows = ck[name]; e = [r['e_near_smin'] for r in rows]
    for m in ['naive', 'fixed', 'ode']:
        a.plot(e, [100 * r['delta'][m] for r in rows], 'o-', color=COL[m], ms=4, label=LAB[m])
    for r in rows: a.annotate(f"{r['step'] // 1000}k", (r['e_near_smin'], 100 * r['delta']['fixed']), fontsize=6.5, color='#666', xytext=(3, 3), textcoords='offset points')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('score error near σ_min  (E|ŝ − s|²/E|s|², σ_min … 2σ_min)')
    a.set_title(DESC[name], loc='left', fontsize=8.5)
ax[0].set_ylabel('Δ = learned − exact, log within-var (%, log scale)'); ax[0].legend(frameon=False, fontsize=7)
fig.suptitle('Checkpoints of network 1 (sampling seed 0, N = 10 000)', fontsize=9)
fig.tight_layout(); fig.savefig('figures/B3_checkpoints.png', dpi=160)

# figure 4: mode picture for B (learned)
w, mu, S = mixture()
for name in 'BC':
    p = setting(name); tgt = np.sqrt(S_POS**2 + p.s0**2)
    fig, a = plt.subplots(figsize=(6.4, 3.4)); bins = np.linspace(0, 4, 61)
    for m in ['naive', 'fixed', 'ode']:
        z = np.concatenate([np.load(f) for f in sorted(glob.glob(f'samp_{name}_n*_s*_{m}.npy'))])
        x = z[:, POS]; lab = np.argmin(((x[:, None] - mu[None][:, :, POS])**2).sum(-1), 1)
        a.hist(np.sqrt(((x - mu[lab][:, POS])**2).sum(1)) / tgt, bins, density=True, histtype='step', lw=1.6, color=COL[m], label=LAB[m])
    r = np.random.default_rng(0).standard_normal((400000, 2)); a.hist(np.sqrt((r**2).sum(1)), bins, density=True, histtype='stepfilled', color='#e3e3e3', zorder=0, label='target')
    a.set_xlabel('distance from mode centre / target std'); a.set_ylabel('density'); a.legend(frameon=False, fontsize=7.5)
    a.set_title(DESC[name] + ' — learned score (2 nets × 3 seeds)', loc='left', fontsize=8.5)
    fig.tight_layout(); fig.savefig(f'figures/B3_modes_{name}.png', dpi=160)

for name in 'BCD':
    d = summ[name]
    print(name, 'paired learned %.4f±%.4f exact %.4f±%.4f ratio %.3f' % (*d['learned']['paired'], *d['exact']['paired'], d['ratio_paired']),
          '| delta i/ii/iii', ' '.join('%+.4f±%.4f' % d['delta'][m] for m in ['naive', 'fixed', 'ode']),
          '| learned L i/ii/iii', ' '.join('%+.4f' % d['learned'][m][0] for m in ['naive', 'fixed', 'ode']),
          '| exact', ' '.join('%+.4f' % d['exact'][m][0] for m in ['naive', 'fixed', 'ode']), '| e', np.round(d['e_near_smin'], 4))
for name in 'BC':
    for r in ck[name]: print(name, r['step'], 'e %.2e' % r['e_near_smin'], 'delta', {m: round(v, 4) for m, v in r['delta'].items()}, 'paired L/E %.4f %.4f' % (r['paired_learned'], r['paired_exact']))
