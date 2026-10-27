"""Section 2-2: shrinkage map from the one-mode Gaussian theory (no sampling).  -> results/map/map_*.json
usage: python experiments/map/map_one_mode.py   (uses 2 processes)"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import json, numpy as np
from multiprocessing import Pool
from one_mode import Theory

DEF = dict(smin=0.0625, smax=500.0, vstd=1.25, M=0.25)
RK = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
def L(args):
    rk, kw = args
    try: return Theory(rk, **{**DEF, **kw}).L()
    except Exception as e: return (np.nan, np.nan)

def grid(name, key, vals, rks=RK, extra=None):
    jobs = [(rk, {key: v, **(extra or {})}) for v in vals for rk in rks]
    with Pool(2) as p: out = p.map(L, jobs)
    Lx = np.array([o[0] for o in out]).reshape(len(vals), len(rks)); Lv = np.array([o[1] for o in out]).reshape(len(vals), len(rks))
    res = {'key': key, 'vals': list(map(float, vals)), 'rk': rks, 'L_pos': Lx.tolist(), 'L_vel': Lv.tolist(), 'fixed': {**DEF, **(extra or {})}}
    json.dump(res, open(f'map_{name}.json', 'w'), indent=1)
    print(name, 'min L_pos', np.nanmin(Lx), 'at', np.unravel_index(np.nanargmin(Lx), Lx.shape), flush=True)

if __name__ == '__main__':
    grid('smin', 'smin', np.logspace(-2, 1, 13))                               # sigma_min / s
    grid('vstd', 'vstd', np.logspace(-1, 1.5, 11))                             # velocity std / s
    grid('smax', 'smax', [5, 10, 20, 50, 100, 500, 5000])                      # sigma_max / s
    grid('M', 'M', np.logspace(-2, 2, 17))                                     # kappa changes with M
    grid('M_smin0.5', 'M', np.logspace(-2, 2, 17), extra={'smin': 0.5})
    grid('M_vstd0.3', 'M', np.logspace(-2, 2, 17), extra={'vstd': 0.3})
