"""Supplement (added after the mixture runs): one isolated mode in setting A, to test whether the velocity
difference (mixture -0.18 vs one-mode theory -0.06) comes from the pipeline or from the mixture.
usage: python experiments/map/single_mode_A.py <seed>  -> results/map/single_A_s<seed>.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import sys, json, numpy as np
from settings_abc import *
seed = int(sys.argv[1]); p, _, _ = setting('A')
w, mu, S = mixture(); w1, mu1, S1 = np.array([1.0]), mu[:1], S[:1]
sc = ExactScore(p, w1, mu1, S1); N = 20000; Kf = int(np.ceil(p.T / 0.001 / 2)) * 2
z0 = prior_exact(p, w1, mu1, S1, N, np.random.default_rng(100 + seed))
th = theory_ratio(p); out = {'seed': seed, 'theory_log_within': float(np.log(th[0])), 'theory_log_vel': float(np.log(th[1]))}
for lev, sub in [('coarse', 2), ('fine', 1)]:
    v = {}
    for md in ['fixed', 'naive']:
        zs = run_sampler(sc, p, z0, md, Kf // sub, np.random.default_rng(200 + seed), sub=sub)
        v[md] = (zs[:, POS].var(0).mean(), zs[:, VEL].var(0).mean())
    out[lev] = [float(np.log(v['naive'][i] / v['fixed'][i])) for i in range(2)]
out['extrap'] = [2 * f - c for f, c in zip(out['fine'], out['coarse'])]
json.dump(out, open(f'single_A_s{seed}.json', 'w'), indent=1); print(out, flush=True)
