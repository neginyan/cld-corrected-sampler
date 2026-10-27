"""Where the damage is made: contribution of each band of log10(sigma/s) to the shrinkage (one-mode theory).
-> results/map/decompose.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import json, numpy as np
from closed_forms import source_by_logsigma
from one_mode import Theory
edges = np.arange(-1.25, 2.75, 0.25)       # log10(sigma/s) from sigma_min = 0.0625 s ... sigma_max = 500 s
out = {'edges': edges.tolist()}
for name, kw, rk in [('default', {}, 0.5), ('default_rk0.1', {}, 0.1), ('A', {'M': 0.01}, 0.2),
                     ('B', {'smin': 1.778}, 0.3), ('C', {'M': 10.0, 'smin': 0.5}, 0.5)]:
    e = edges if 'smin' not in kw else np.concatenate([[np.log10(kw['smin'])], edges[edges > np.log10(kw['smin'])]])
    c = source_by_logsigma(rk, e, **kw); tot = Theory(rk, **kw).E0()[0, 0] / Theory(rk, **kw).C(0)[0, 0]
    out[name] = {'edges': e.tolist(), 'contrib': c, 'sum': float(sum(c)), 'total_direct': float(tot), 'rk': rk, 'kw': kw}
    print(name, 'sum of bands', round(sum(c), 6), 'direct', round(tot, 6), 'peak band', e[int(np.argmin(c))], flush=True)
json.dump(out, open('decompose.json', 'w'), indent=1)
