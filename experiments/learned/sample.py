"""Sampling with a learned score and with the exact score on the same prior draws and Brownian increments.
usage: python experiments/learned/sample.py <setting> <net_seed> <step> <sampling_seed> <N>  -> results/learned/samp_<setting>_n<ns>_k<step>_s<seed>_N<N>.json
Step h = T / ceil(T / 0.002) (Euler-Maruyama for (i)/(ii), Heun for (iii)), as pre-registered."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
WEIGHTING = os.environ.get('CLD_WEIGHTING', 'eps')            # 'eps' (Sec. V) or 'score' (Obs. 9)
STAGE = 'learned' if WEIGHTING == 'eps' else 'score_weighting'
WEIGHTS = os.path.join(ROOT, 'weights', 'learned_' + WEIGHTING)
os.chdir(os.path.join(ROOT, 'results', STAGE))   # relative paths below are inside results/<stage>/
import sys, json, time
import numpy as np
from learned import *
from cld_mixture import run_sampler

name, ns, step, sd, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
p = setting(name); pc = Precond(p, train_data(0)); net = new_net(ns); net.load(f'{WEIGHTS}/{name}_seed{ns}_step{step}')
w, mu, S = mixture(); scores = {'learned': LearnedScore(net, pc), 'exact': ExactScore(p, w, mu, S)}
K = int(np.ceil(p.T / 0.002)); z0 = prior_exact(p, w, mu, S, N, np.random.default_rng(100 + sd))
truth, _ = sample_mixture(w, mu, S, 400000, np.random.default_rng(999), extra_var=p.s0**2); ref = mode_stats(truth, mu)
out = {'setting': name, 'net_seed': ns, 'step': step, 'seed': sd, 'N': N, 'K': K, 'h': p.T / K, 'T': p.T}; t0 = time.time()
for kind, sc in scores.items():
    r = {}
    for md in ['fixed', 'naive', 'ode']:
        zs = run_sampler(sc, p, z0, md, K, np.random.default_rng(200 + sd), sub=1)
        st = mode_stats(zs, mu)
        r[md] = {'L': float(np.log(st['within_var'] / ref['within_var'])), 'L_vel': float(np.log(st['vel_var'] / ref['vel_var'])),
                 'max_weight_dev': float(np.max(np.abs(np.array(st['weights']) - 1 / len(w))))}
        if kind == 'learned' and step == 300000: np.save(f'samp_{name}_n{ns}_s{sd}_{md}.npy', zs[:4000].astype(np.float32))
    r['paired'] = r['naive']['L'] - r['fixed']['L']
    out[kind] = r
out['delta'] = {md: out['learned'][md]['L'] - out['exact'][md]['L'] for md in ['naive', 'fixed', 'ode']}
out['sec'] = time.time() - t0
json.dump(out, open(f'samp_{name}_n{ns}_k{step}_s{sd}_N{N}.json', 'w'), indent=1)
print(name, ns, step, sd, N, 'paired learned %.4f exact %.4f' % (out['learned']['paired'], out['exact']['paired']),
      'delta i/ii/iii', ' '.join('%.4f' % out['delta'][m] for m in ['naive', 'fixed', 'ode']), '%.0f s' % out['sec'], flush=True)
