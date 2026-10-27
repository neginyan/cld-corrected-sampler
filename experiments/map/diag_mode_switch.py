"""Supplement: why is the seed spread of the B1 mixture paired shrinkage (r/kappa = 0.5) ~25x the single-mode spread?
Rerun B1 seed <s> at h = 0.001 (identical paths to B1), keep all 40000 samples, and compare
(a) the B1 estimator (nearest-centre within-mode variance), (b) the same restricted to trajectories that land
in the same mode under (i) and (ii), (c) a truncated version (distance < 4 target std).
usage: python experiments/map/diag_mode_switch.py <seed>  -> results/map/diag_B1seed<s>.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import sys, json, numpy as np
import cld_mixture
from cld_mixture import *
seed = int(sys.argv[1]); w, mu, S = mixture(); p = process(0.5); sc = ExactScore(p, w, mu, S); N = 40000
Kf = int(np.ceil(p.T / 0.001 / 2)) * 2
z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100 + seed))
Z = {md: run_sampler(sc, p, z0, md, Kf, np.random.default_rng(200 + seed), sub=1) for md in ['fixed', 'naive']}
c = mu[:, POS]; tgt = np.sqrt(S_POS**2 + S_MIN**2)
lab = {m: np.argmin(((Z[m][:, None, POS] - c[None])**2).sum(-1), 1) for m in Z}
d2 = {m: ((Z[m][:, POS] - c[lab[m]])**2).sum(1) / 2 for m in Z}
same = lab['naive'] == lab['fixed']
trunc = (d2['naive'] < (4 * tgt)**2) & (d2['fixed'] < (4 * tgt)**2)
out = {'seed': seed, 'n_switch': int((~same).sum()), 'n_trunc_removed': int((~trunc).sum()),
       'L_all': float(np.log(d2['naive'].mean() / d2['fixed'].mean())),
       'L_same_mode': float(np.log(d2['naive'][same].mean() / d2['fixed'][same].mean())),
       'L_trunc': float(np.log(d2['naive'][trunc].mean() / d2['fixed'][trunc].mean())),
       'max_d_over_tgt': {m: float(np.sqrt(d2[m].max()) / tgt) for m in Z},
       'top5_contrib_naive': (np.sort(d2['naive'])[-5:] / d2['naive'].sum()).tolist(),
       'top5_contrib_fixed': (np.sort(d2['fixed'])[-5:] / d2['fixed'].sum()).tolist()}
json.dump(out, open(f'diag_B1seed{seed}.json', 'w'), indent=1); print(out, flush=True)
