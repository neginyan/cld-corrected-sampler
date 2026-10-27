"""POST-HOC supplement (not pre-registered): learned-score samples of setting D with the probability-flow ODE contain a few
diverging trajectories.  On the saved 4000-sample subsets, count trajectories farther than 0.5 from every mode centre and
compute the within-mode log-variance ratio to the target with and without them.  -> results/learned/posthoc_trim_D.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
WEIGHTING = os.environ.get('CLD_WEIGHTING', 'eps')            # 'eps' (Sec. V) or 'score' (Obs. 9)
STAGE = 'learned' if WEIGHTING == 'eps' else 'score_weighting'
WEIGHTS = os.path.join(ROOT, 'weights', 'learned_' + WEIGHTING)
os.chdir(os.path.join(ROOT, 'results', STAGE))   # relative paths below are inside results/<stage>/
import json, glob, numpy as np
from learned import mixture, sample_mixture, setting, POS
w, mu, S = mixture(); out = {}
for name in 'BCD':
    p = setting(name); tz, _ = sample_mixture(w, mu, S, 400000, np.random.default_rng(999), extra_var=p.s0**2)
    def wv(z, cut):
        x = z[:, POS]; d2 = ((x[:, None] - mu[None][:, :, POS])**2).sum(-1); lab = d2.argmin(1); d = np.sqrt(d2.min(1))
        keep = np.isfinite(d) & (d < cut); return float((((x - mu[lab][:, POS])**2)[keep]).sum(1).mean() / 2), int((~keep).sum())
    ref, _ = wv(tz, 0.5)
    for md in ['naive', 'fixed', 'ode']:
        rows = []
        for f in sorted(glob.glob(f'samp_{name}_n*_s*_{md}.npy')):
            z = np.load(f).astype(float); v, nrm = wv(z, 0.5); rows.append({'file': f, 'L_trim': float(np.log(v / ref)), 'n_removed': nrm, 'n': len(z)})
        out[f'{name}_{md}'] = {'mean_L_trim': float(np.mean([r['L_trim'] for r in rows])), 'removed_total': int(sum(r['n_removed'] for r in rows)),
                               'n_total': int(sum(r['n'] for r in rows)), 'rows': rows}
        print(name, md, 'L_trim %.4f' % out[f'{name}_{md}']['mean_L_trim'], 'removed', out[f'{name}_{md}']['removed_total'], '/', out[f'{name}_{md}']['n_total'])
json.dump(out, open('posthoc_trim.json', 'w'), indent=1)
