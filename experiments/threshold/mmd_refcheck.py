"""Supplement: are the 5 reference samples used for MMD (rng 6000+seed) atypical?  MMD^2 of 20 fresh target
samples against each of them, compared with the null mean.  -> results/threshold/mmd_refcheck_B1.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
import json, numpy as np
from cld_mixture import *
w, mu, S = mixture(); out = {}
for s in range(5):
    mref, _ = sample_mixture(w, mu, S, 4000, np.random.default_rng(6000 + s), extra_var=S_MIN**2)
    v = [mmd2(sample_mixture(w, mu, S, 4000, np.random.default_rng(30000 + 100 * s + i), extra_var=S_MIN**2)[0][:, POS], mref[:, POS]) for i in range(20)]
    out[s] = {'mean': float(np.mean(v)), 'se': float(np.std(v, ddof=1) / np.sqrt(20))}
out['all_mean'] = float(np.mean([out[s]['mean'] for s in range(5)]))
json.dump(out, open('mmd_refcheck_B1.json', 'w'), indent=1); print(out)
