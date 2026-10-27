"""Section 2-3: mixture check in a B2 setting (same method as B1).
usage: python experiments/map/run_mixture.py <setting> <seed> <h_fine>  -> results/map/main_<setting>_s<seed>.json
Two step sizes on one Brownian path (2 h_fine and h_fine), Richardson extrapolation; (i)/(ii) paired; N = 40000."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import sys, json, time
import numpy as np
from settings_abc import *

name, seed, H = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
N, NMMD = 40000, 4000
p, kap, beta = setting(name)
w, mu, S = mixture(); sc = ExactScore(p, w, mu, S)
Kf = int(np.ceil(p.T / H / 2)) * 2
z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100 + seed))
truth, _ = sample_mixture(w, mu, S, 400000, np.random.default_rng(999), extra_var=p.s0**2)
ref = mode_stats(truth, mu)
mref, _ = sample_mixture(w, mu, S, NMMD, np.random.default_rng(7000 + 13 * seed + 1), extra_var=p.s0**2)   # fresh per seed
th = theory_ratio(p)
out = {'setting': name, **SETTINGS[name], 'beta': beta, 'kappa': kap, 'r': p.r, 'T': p.T, 'seed': seed, 'N': N,
       'K_fine': Kf, 'h_fine': p.T / Kf, 'ref': ref, 'theory_log_within': float(np.log(th[0])), 'theory_log_vel': float(np.log(th[1]))}
t0 = time.time()
for lev, sub in [('coarse', 2), ('fine', 1)]:
    K = Kf // sub; res = {'K': K, 'h': p.T / K}
    for mode in ['fixed', 'naive', 'ode']:
        zs = run_sampler(sc, p, z0, mode, K, np.random.default_rng(200 + seed), sub=sub)
        st = mode_stats(zs, mu)
        st['within_ratio'] = st['within_var'] / ref['within_var']; st['vel_ratio'] = st['vel_var'] / ref['vel_var']
        st['mmd2'] = mmd2(zs[:NMMD, POS], mref[:, POS])
        st['max_weight_dev'] = float(np.max(np.abs(np.array(st['weights']) - 1 / len(w))))
        res[mode] = st
        if lev == 'fine': np.save(f'main_{name}_s{seed}_{mode}.npy', zs[:8000].astype(np.float32))
    res['paired_log_within'] = float(np.log(res['naive']['within_var'] / res['fixed']['within_var']))
    res['paired_log_vel'] = float(np.log(res['naive']['vel_var'] / res['fixed']['vel_var']))
    out[lev] = res
ex = lambda f, c: 2 * f - c
out['extrap'] = {m: {'log_within': ex(np.log(out['fine'][m]['within_ratio']), np.log(out['coarse'][m]['within_ratio'])),
                     'log_vel': ex(np.log(out['fine'][m]['vel_ratio']), np.log(out['coarse'][m]['vel_ratio']))} for m in ['naive', 'fixed', 'ode']}
out['extrap']['paired_log_within'] = ex(out['fine']['paired_log_within'], out['coarse']['paired_log_within'])
out['extrap']['paired_log_vel'] = ex(out['fine']['paired_log_vel'], out['coarse']['paired_log_vel'])
out['sec'] = time.time() - t0
json.dump(out, open(f'main_{name}_s{seed}.json', 'w'), indent=1)
print(name, seed, 'paired c/f/x', round(out['coarse']['paired_log_within'], 4), round(out['fine']['paired_log_within'], 4),
      round(out['extrap']['paired_log_within'], 4), 'theory', round(out['theory_log_within'], 4),
      {m: round(out['extrap'][m]['log_within'], 4) for m in ['naive', 'fixed', 'ode']}, round(out['sec']), 's', flush=True)
