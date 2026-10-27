"""Figure for sec. 2-1: (a) a unit pulse of relative error in the noise-dominated regime, carried by (ii) and (iii)
(closed-form generator); (b) Delta_iii / Delta_ii, theory vs B3 observation.  -> figures/B4_forgetting.png"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'forgetting'))   # relative paths below are inside results/<stage>/
import json, glob
import numpy as np
from scipy.linalg import expm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from one_mode import Theory
from forgetting import generator

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6'})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
th = Theory(0.3)                                    # setting B flow (M = 1/4, r/kappa = 0.3)
F0 = np.array([[1.0, 0.0], [0.0, 0.0]]); taus = np.linspace(0, 0.5, 300)
for mode, col, lab in [('fixed', '#2a78d6', '(ii) corrected SDE'), ('ode', '#1baf7a', '(iii) probability-flow ODE')]:
    Mg = generator(th, mode, 1.0)
    y = [np.linalg.norm(expm(Mg * t) @ F0 @ expm(Mg * t).T) for t in taus]
    ax[0].plot(taus, y, color=col, lw=1.8, label=lab)
ax[0].set_yscale('log'); ax[0].set_xlabel('reverse time τ after the error is made (β = 4 units)')
ax[0].set_ylabel('size of the relative error |F|'); ax[0].legend(frameon=False, fontsize=7.5)
ax[0].set_title('Noise-dominated regime: the ODE never forgets', loc='left', fontsize=9)

T = json.load(open('theory_b4.json'))['deltas']
for n, mk, col in [('B', 'o', '#1c5cab'), ('C', 's', '#eb6834')]:
    th_r, ob_r, labs = [], [], []
    for k in [10000, 40000, 100000, 300000]:
        d = [x for x in T if x['setting'] == n and x['step'] == k and x['net'] == 1][0]; th_r.append(d['ode'] / d['fixed'])
        o = json.load(open(f'../learned/samp_{n}_n1_k{k}_s0_N10000.json'))['delta']; ob_r.append(o['ode'] / o['fixed'])
    ax[1].scatter(th_r, ob_r, marker=mk, color=col, s=30, label=f'setting {n}, checkpoints 10k–300k (net 1, seed 0)', zorder=3)
    for k, a, b in zip(['10k', '40k', '100k', '300k'], th_r, ob_r): ax[1].annotate(k, (a, b), fontsize=6.5, color='#666', xytext=(3, 2), textcoords='offset points')
lim = [0.8, 30]; ax[1].plot(lim, lim, color='#999', lw=0.8, ls='--'); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('theory Δ_iii / Δ_ii (one mode, ŝ = (1 − η)s)'); ax[1].set_ylabel('observed Δ_iii / Δ_ii (B3)')
ax[1].legend(frameon=False, fontsize=7); ax[1].set_title('ODE vs corrected SDE: theory and observation', loc='left', fontsize=9)
fig.tight_layout(); fig.savefig('figures/B4_forgetting.png', dpi=160)
