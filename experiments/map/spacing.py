"""Section 2-4: does measured/theory -> 1 as the modes move apart?  B1 default setting (M = 1/4, sigma_min = 0.005,
r/kappa = 0.5), 8 modes on a circle of radius R x (B1 radius), N = 40000, (i)/(ii) paired, h = 0.002 and 0.001, extrapolated.
usage: python experiments/map/spacing.py <radius_factor> <seed>  -> results/map/spacing_R<f>_s<seed>.json
(seeds fixed before running: 0, 1, 2)"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import sys, json
import numpy as np
import cld_mixture
from cld_mixture import *

f, seed = float(sys.argv[1]), int(sys.argv[2])
w, mu, S = mixture(); mu = mu * f
p = process(0.5); sc = ExactScore(p, w, mu, S); N = 40000
Kf = int(np.ceil(p.T / 0.001 / 2)) * 2
z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100 + seed))
out = {'radius_factor': f, 'seed': seed, 'theory_log_within': float(np.log(theory_within_ratio(0.5)[0]))}
for lev, sub in [('coarse', 2), ('fine', 1)]:
    v = {}
    for md in ['fixed', 'naive']:
        zs = run_sampler(sc, p, z0, md, Kf // sub, np.random.default_rng(200 + seed), sub=sub)
        v[md] = mode_stats(zs, mu)['within_var']
    out[lev] = float(np.log(v['naive'] / v['fixed']))
out['extrap'] = 2 * out['fine'] - out['coarse']; out['ratio_to_theory'] = out['extrap'] / out['theory_log_within']
json.dump(out, open(f'spacing_R{f}_s{seed}.json', 'w'), indent=1); print(out, flush=True)
