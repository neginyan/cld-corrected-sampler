"""Supplement (added after the original MMD check failed): null distribution of MMD^2 for target-vs-target
pairs of the same size (4000 vs 4000, 100 independent pairs), and where each sampler's 55 run values fall in it.
-> results/threshold/mmd_null_B1.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
import json, numpy as np
from cld_mixture import *
w, mu, S = mixture()
null = []
for i in range(100):
    a, _ = sample_mixture(w, mu, S, 4000, np.random.default_rng(20000 + 2 * i), extra_var=S_MIN**2)
    b, _ = sample_mixture(w, mu, S, 4000, np.random.default_rng(20001 + 2 * i), extra_var=S_MIN**2)
    null.append(mmd2(a[:, POS], b[:, POS]))
null = np.array(null); q95 = float(np.percentile(null, 95))
out = {'null_mean': float(null.mean()), 'null_sd': float(null.std(ddof=1)), 'null_q95': q95, 'null': null.tolist()}
for md in ['naive', 'fixed', 'ode']:
    for part, cond in [('r_below', lambda r: r < 1), ('r_at_or_above', lambda r: r >= 1)]:
        v = [json.load(open(f))['fine'][md]['mmd2'] for f in sorted(__import__('glob').glob('main_r*_s*.json'))
             if cond(json.load(open(f))['ratio'])]
        out[f'{md}_{part}'] = {'n': len(v), 'mean': float(np.mean(v)), 'frac_above_null_q95': float(np.mean(np.array(v) > q95)),
                               'mean_percentile_in_null': float(np.mean([(null < x).mean() for x in v]))}
json.dump(out, open('mmd_null_B1.json', 'w'), indent=1)
print({k: (v if not isinstance(v, list) else '...') for k, v in out.items()})
