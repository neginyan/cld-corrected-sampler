"""Supplement: one isolated Gaussian mode (the theory's own assumption).  If the pipeline is right, the paired
shrinkage of (i) vs (ii) must equal the one-mode Gaussian-moment prediction.
usage: python experiments/threshold/single_mode.py <ratio> <seed>  -> results/threshold/single_r<ratio>_s<seed>.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
import sys, json, numpy as np
from cld_mixture import *
ratio, seed = float(sys.argv[1]), int(sys.argv[2])
w, mu, S = mixture(); w1, mu1, S1 = np.array([1.0]), mu[:1], S[:1]
p = process(ratio); sc = ExactScore(p, w1, mu1, S1); N = 40000
Kf = int(np.ceil(p.T / 0.001 / 2)) * 2
z0 = prior_exact(p, w1, mu1, S1, N, np.random.default_rng(100 + seed))
out = {'ratio': ratio, 'seed': seed, 'theory_log_within': float(np.log(theory_within_ratio(ratio)[0]))}
for lev, sub in [('coarse', 2), ('fine', 1)]:
    v = {}
    for md in ['fixed', 'naive']:
        zs = run_sampler(sc, p, z0, md, Kf // sub, np.random.default_rng(200 + seed), sub=sub)
        v[md] = zs[:, POS].var(0).mean()
    out[lev] = float(np.log(v['naive'] / v['fixed']))
out['extrap'] = 2 * out['fine'] - out['coarse']
json.dump(out, open(f'single_r{ratio}_s{seed}.json', 'w'), indent=1); print(out, flush=True)
