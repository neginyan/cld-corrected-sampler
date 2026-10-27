"""Step-size study (coupled Brownian paths): halve the step until the results stop changing.
usage: python experiments/threshold/steps.py <ratio> <seed>   -> results/threshold/steps_r<ratio>_s<seed>.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
import sys, json, time
import numpy as np
from cld_mixture import *

ratio, seed = float(sys.argv[1]), int(sys.argv[2])
N = 10000
H_LEVELS = [0.008, 0.004, 0.002, 0.001]          # target step sizes (finest uses sub=1)
w, mu, S = mixture(); p = process(ratio); sc = ExactScore(p, w, mu, S)
z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100 + seed))
Kf = int(np.ceil(p.T / H_LEVELS[-1] / 8)) * 8                    # finest step count, divisible by 8
out = {'ratio': ratio, 'seed': seed, 'N': N, 'T': p.T, 'levels': []}
truth, _ = sample_mixture(w, mu, S, 200000, np.random.default_rng(999), extra_var=S_MIN**2)
ref = mode_stats(truth, mu)
for j, sub in enumerate([8, 4, 2, 1]):
    K = Kf // sub; row = {'K': K, 'h': p.T / K}
    t0 = time.time()
    for mode in ['fixed', 'naive', 'ode']:
        if mode == 'naive' and ratio >= 1: row['naive'] = row['fixed']; continue
        zs = run_sampler(sc, p, z0, mode, K, np.random.default_rng(200 + seed), sub=sub)
        st = mode_stats(zs, mu)
        row[mode] = {'within_ratio': st['within_var'] / ref['within_var'], 'vel_ratio': st['vel_var'] / ref['vel_var']}
    row['paired_log_within'] = float(np.log(row['naive']['within_ratio'] / row['fixed']['within_ratio']))
    row['paired_log_vel'] = float(np.log(row['naive']['vel_ratio'] / row['fixed']['vel_ratio']))
    row['sec'] = time.time() - t0
    out['levels'].append(row); print(ratio, seed, K, {k: (round(v['within_ratio'], 5) if isinstance(v, dict) else v) for k, v in row.items()}, flush=True)
json.dump(out, open(f'steps_r{ratio}_s{seed}.json', 'w'), indent=1)
