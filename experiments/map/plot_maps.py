"""Figures for sec. 2-2 (maps) and the decomposition.  -> figures/B2_map.png, figures/B2_decompose.png"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from closed_forms import L_noise_limit

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#999',
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
RAMP = {0.1: '#0d366b', 0.3: '#1c5cab', 0.5: '#2a78d6', 0.7: '#6da7ec', 0.9: '#9ec5f4'}
pct = lambda L: 100 * (np.exp(np.array(L, float)) - 1)
panels = [('smin', 'σ_min / s', True), ('vstd', 'initial velocity std / s', True), ('M', 'CLD mass M  (κ/β changes; M = 1 gives κ = 0)', True),
          ('M_smin0.5', 'M, with σ_min = 0.5 s', True)]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for a, (name, xl, logx) in zip(ax, panels):
    d = json.load(open(f'map_{name}.json')); v = np.array(d['vals']); L = np.array(d['L_pos'])
    keep = ~np.isclose(v, 1.0) if name.startswith('M') else np.ones(len(v), bool)
    for rk, col in RAMP.items():
        j = d['rk'].index(rk); a.plot(v[keep], pct(L[keep, j]), 'o-', ms=3, lw=1.6, color=col, label=f'r/κ = {rk}')
    a.axhline(-10, color='#b8483c', lw=1, ls='--'); a.set_xscale('log' if logx else 'linear'); a.set_xlabel(xl)
    if name == 'smin':
        a.axvline(0.0625, color='#999', lw=0.8, ls=':'); a.text(0.0625, 1, ' B1', fontsize=7, color='#666')
        for rk, col in RAMP.items(): a.axhline(pct(L_noise_limit(rk)[0]), color=col, lw=0.8, ls=':', xmin=0.85)
    if name == 'vstd': a.axvline(1.25, color='#999', lw=0.8, ls=':'); a.text(1.25, 1, ' B1', fontsize=7, color='#666')
    if name == 'M': a.axvline(0.25, color='#999', lw=0.8, ls=':'); a.text(0.25, 1, ' CLD', fontsize=7, color='#666')
ax[0].set_ylabel('within-mode position variance change (%)\n(i) clipped SDE, one-mode theory')
ax[0].legend(frameon=False, fontsize=7, loc='lower left'); ax[0].text(0.012, -10.8, '−10 %', fontsize=7, color='#b8483c')
ax[0].set_title('Shrinkage map (σ_max has no effect: 5–5000 s identical)', loc='left')
fig.tight_layout(); fig.savefig('figures/B2_map.png', dpi=160)

d = json.load(open('decompose.json'))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for (name, lab), col in zip([('default', 'B1 default, r/κ=0.5'), ('default_rk0.1', 'B1 default, r/κ=0.1'), ('A', 'setting A'),
                              ('B', 'setting B'), ('C', 'setting C')], ['#2a78d6', '#0d366b', '#eb6834', '#1baf7a', '#eda100']):
    x = d[name]; e = np.array(x['edges']); c = np.array(x['contrib']) / x['sum']
    ax.step(e[:-1], 100 * c, where='post', color=col, lw=1.6, label=lab)
ax.set_xlabel('log10(σ / s) at which the error is injected'); ax.set_ylabel('share of the final shrinkage (%)')
ax.set_title('Where the damage is made (one-mode theory)', loc='left'); ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('figures/B2_decompose.png', dpi=160)
