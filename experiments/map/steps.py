"""Step-size study for a B2 setting (coupled Brownian paths, 1 seed, N = 10000).
usage: python experiments/map/steps.py <setting>  -> results/map/steps_<setting>.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import sys, json, time
import numpy as np
from settings_abc import *

name = sys.argv[1]; p, kap, beta = setting(name); N = 10000
w, mu, S = mixture(); sc = ExactScore(p, w, mu, S)
z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100))
truth, _ = sample_mixture(w, mu, S, 400000, np.random.default_rng(999), extra_var=p.s0**2); ref = mode_stats(truth, mu)
th = float(np.log(theory_ratio(p)[0]))
Kf = int(np.ceil(p.T / 0.0005 / 16)) * 16
out = {'setting': name, 'T': p.T, 'theory_log': th, 'levels': []}
for sub in [16, 8, 4, 2, 1]:
    K = Kf // sub; row = {'K': K, 'h': p.T / K}; t0 = time.time()
    for md in ['fixed', 'naive', 'ode']:
        zs = run_sampler(sc, p, z0, md, K, np.random.default_rng(200), sub=sub)
        row[md] = mode_stats(zs, mu)['within_var'] / ref['within_var']
    row['paired'] = float(np.log(row['naive'] / row['fixed'])); row['sec'] = time.time() - t0
    out['levels'].append(row)
    print(name, K, round(row['h'], 5), 'paired', round(row['paired'], 4), 'theory', round(th, 4),
          {m: round(row[m], 4) for m in ['naive', 'fixed', 'ode']}, round(row['sec']), 's', flush=True)
    json.dump(out, open(f'steps_{name}.json', 'w'), indent=1)
