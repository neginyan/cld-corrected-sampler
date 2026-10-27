"""Main sweep of stage B1.  usage: python experiments/threshold/run_sweep.py <ratio> <seed>  -> results/threshold/main_r<ratio>_s<seed>.json
Two step sizes on one Brownian path (h ~ 0.002 with sub=2 and h ~ 0.001 with sub=1); Richardson extrapolation
x_inf = 2 x(h/2) - x(h) (Euler-Maruyama is first order, step study: results/steps_*.json).
(i) naive and (ii) fixed share the prior draw and the Brownian increments (paired); (iii) ode shares the prior."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
import sys, json, time
import numpy as np
from cld_mixture import *

ratio, seed = float(sys.argv[1]), int(sys.argv[2])
H_FINE, N, NMMD = 0.001, 40000, 4000
w, mu, S = mixture(); p = process(ratio); sc = ExactScore(p, w, mu, S)
Kf = int(np.ceil(p.T / H_FINE / 2)) * 2
z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100 + seed))
truth, _ = sample_mixture(w, mu, S, 400000, np.random.default_rng(999), extra_var=S_MIN**2)
ref = mode_stats(truth, mu)
ref2, _ = sample_mixture(w, mu, S, NMMD, np.random.default_rng(5000 + seed), extra_var=S_MIN**2)
mref, _ = sample_mixture(w, mu, S, NMMD, np.random.default_rng(6000 + seed), extra_var=S_MIN**2)
out = {'ratio': ratio, 'r': p.r, 'kappa': KAPPA, 'seed': seed, 'N': N, 'T': p.T, 'K_fine': Kf, 'h_fine': p.T / Kf,
       'ref': ref, 'mmd2_truth_vs_truth': mmd2(ref2[:, POS], mref[:, POS])}
t0 = time.time()
for lev, sub in [('coarse', 2), ('fine', 1)]:
    K = Kf // sub; res = {'K': K, 'h': p.T / K}
    for mode in ['fixed', 'naive', 'ode']:
        if mode == 'naive' and ratio >= 1:
            res['naive'] = dict(res['fixed']); res['naive']['identical_to_fixed_by_construction'] = True; continue
        zs = run_sampler(sc, p, z0, mode, K, np.random.default_rng(200 + seed), sub=sub)
        st = mode_stats(zs, mu)
        st['within_ratio'] = st['within_var'] / ref['within_var']
        st['vel_ratio'] = st['vel_var'] / ref['vel_var']
        st['mmd2'] = mmd2(zs[:NMMD, POS], mref[:, POS])
        st['max_weight_dev'] = float(np.max(np.abs(np.array(st['weights']) - 1 / N_MODES)))
        res[mode] = st
        if lev == 'fine': np.save(f'main_r{ratio}_s{seed}_{mode}.npy', zs[:4000].astype(np.float32))
    res['paired_log_within'] = float(np.log(res['naive']['within_var'] / res['fixed']['within_var']))
    res['paired_log_vel'] = float(np.log(res['naive']['vel_var'] / res['fixed']['vel_var']))
    out[lev] = res
ext = lambda key, m=None, f='within_ratio': (2 * out['fine'][m][f] - out['coarse'][m][f]) if m else 2 * out['fine'][key] - out['coarse'][key]
out['extrap'] = {m: {'within_ratio': ext(None, m), 'vel_ratio': ext(None, m, 'vel_ratio')} for m in ['naive', 'fixed', 'ode']}
out['extrap']['paired_log_within'] = ext('paired_log_within'); out['extrap']['paired_log_vel'] = ext('paired_log_vel')
out['sec'] = time.time() - t0
json.dump(out, open(f'main_r{ratio}_s{seed}.json', 'w'), indent=1)
print(ratio, seed, 'paired fine/extrap', round(out['fine']['paired_log_within'], 5), round(out['extrap']['paired_log_within'], 5),
      {m: round(out['extrap'][m]['within_ratio'], 4) for m in ['naive', 'fixed', 'ode']}, round(out['sec']), 's', flush=True)
