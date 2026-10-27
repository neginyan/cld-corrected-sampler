"""Relative score error by sigma for one network checkpoint.
usage: python experiments/learned/score_error.py <setting> <net_seed> <step>  -> results/learned/err_<setting>_n<seed>_k<step>.json
e(sigma) = E|s_theta - s|^2 / E|s|^2 over 20 000 fresh samples of p_sigma, split into position / velocity components."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
WEIGHTING = os.environ.get('CLD_WEIGHTING', 'eps')            # 'eps' (Sec. V) or 'score' (Obs. 9)
STAGE = 'learned' if WEIGHTING == 'eps' else 'score_weighting'
WEIGHTS = os.path.join(ROOT, 'weights', 'learned_' + WEIGHTING)
os.chdir(os.path.join(ROOT, 'results', STAGE))   # relative paths below are inside results/<stage>/
import sys, json
import numpy as np
from learned import *

name, ns, step = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
p = setting(name); pc = Precond(p, train_data(0)); net = new_net(ns); net.load(f'{WEIGHTS}/{name}_seed{ns}_step{step}')
ls = LearnedScore(net, pc); w, mu, S = mixture(); ex = ExactScore(p, w, mu, S)
rng = np.random.default_rng(31_000); sigs = np.exp(np.linspace(np.log(p.s0), np.log(p.sT), 40)); out = []
for sg in sigs:
    t = np.log(sg / p.s0) / p.r
    z0, _ = sample_mixture(w, mu, S, 20000, rng); zt = z0 @ expm(p.B * t).T + sg * rng.standard_normal(z0.shape)
    s = ex(zt, t); sh = ls(zt, t); d = sh - s
    out.append({'sigma': float(sg), 't': float(t), 'e_all': float((d**2).sum(1).mean() / (s**2).sum(1).mean()),
                'e_pos': float((d[:, POS]**2).sum(1).mean() / (s[:, POS]**2).sum(1).mean()),
                'e_vel': float((d[:, VEL]**2).sum(1).mean() / (s[:, VEL]**2).sum(1).mean())})
near = [o['e_all'] for o in out if o['sigma'] <= 2 * p.s0]
res = {'setting': name, 'net_seed': ns, 'step': step, 'by_sigma': out, 'e_near_smin': float(np.mean(near)),
       'e_near_smin_pos': float(np.mean([o['e_pos'] for o in out if o['sigma'] <= 2 * p.s0]))}
json.dump(res, open(f'err_{name}_n{ns}_k{step}.json', 'w'), indent=1)
print(name, ns, step, 'e_near_smin %.2e' % res['e_near_smin'], 'max e %.2e' % max(o['e_all'] for o in out), flush=True)
