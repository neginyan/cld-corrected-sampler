"""Sec. 2-1 numbers: theory Delta for the B3 settings with the measured error profiles. -> results/forgetting/theory_b4.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'forgetting'))   # relative paths below are inside results/<stage>/
import json, numpy as np
from one_mode import Theory
from forgetting import deltas, forgetting_rates
from cld_mixture import S_POS
def eta_from(path):
    e = json.load(open(path))['by_sigma']; ls = np.log([o['sigma'] / S_POS for o in e]); le = np.log([o['e_pos'] for o in e])
    return lambda sig: float(np.sqrt(np.exp(np.interp(np.log(sig), ls, le))))
SET = {'B': ({'smin': 1.778}, 0.3), 'C': ({'M': 10.0, 'smin': 0.5}, 0.5), 'D': ({}, 0.5)}
out = {'rates': {}, 'deltas': []}
for n, (kw, rk) in SET.items():
    th = Theory(rk, **kw); out['rates'][n] = {m: forgetting_rates(th, m).tolist() for m in ['naive', 'fixed', 'ode']}
    out['rates'][n]['r_minus_wmin'] = float(th.r - th.w[0]); out['rates'][n]['beta_scale'] = float(np.abs(th.b).max())
jobs = [('B', k, 1) for k in [10000, 40000, 100000, 300000]] + [('C', k, 1) for k in [10000, 40000, 100000, 300000]] + [('B', 300000, 2), ('C', 300000, 2), ('D', 300000, 1), ('D', 300000, 2)]
for n, k, ns in jobs:
    kw, rk = SET[n]; th = Theory(rk, **kw); d = deltas(th, eta_from(f'../learned/err_{n}_n{ns}_k{k}.json'))
    d.update(setting=n, step=k, net=ns); out['deltas'].append(d)
    print(n, k, ns, {m: round(d[m], 5) for m in ['naive', 'fixed', 'ode']}, 'ratio', round(d['ode'] / d['fixed'], 2), flush=True)
    json.dump(out, open('theory_b4.json', 'w'), indent=1)
