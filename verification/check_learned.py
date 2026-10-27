"""Stage B3 verification (tolerances from preregistration/B3_事前登録.md (English translation: preregistration/B3_preregistration_en.md)).  python verification/check_learned.py  (exit 0 iff all pass)."""
import sys, os, json, glob, subprocess, tempfile
import numpy as np
from scipy.linalg import expm
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'learned'))   # relative paths below are inside results/<stage>/
from learned import *

res = []
def check(name, value, tol, kind='abs'):
    if kind == 'abs': ok, used = abs(value) <= tol, (abs(value) / tol if tol else (0.0 if value == 0 else np.inf))
    elif kind == 'le': ok, used = value <= tol, (max(value, 0) / tol if tol > 0 else (0.0 if value <= 0 else np.inf))
    elif kind == 'in': lo, hi = tol; ok = lo <= value <= hi; used = abs(value - (lo + hi) / 2) / ((hi - lo) / 2); tol = hi
    res.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: value {value:.4g}, tol {tol:.3g} ({kind}), used {100 * used:.1f}%")

print('== A. implementation')
for name in 'BCD':
    p = setting(name); data = train_data(0); pc = Precond(p, data)
    check(f'{name}: kappa = 2', kappa_of(p.B) - 2, 1e-12)
    sig = np.exp(np.random.default_rng(1).uniform(np.log(p.s0), np.log(p.sT), 50)); t = pc.t_of(sig)
    Mx = np.array([expm(p.B * tt) for tt in t]); err = np.abs(pc.M_of(sig) - Mx).max() / np.abs(Mx).max()
    check(f'{name}: tabulated M(t) relative error <= 1e-4', err, 1e-4, 'le')
    m_ex = Mx @ data.mean(0); err = np.abs(pc.interp(pc.mt, sig) - m_ex).max() / (np.abs(m_ex).max() + 1e-12)
    check(f'{name}: tabulated mean relative error <= 1e-4', err, 1e-4, 'le')
# learned score = -eps/sigma and features are finite
p = setting('B'); pc = Precond(p, train_data(0)); net = new_net(1); net.load(os.path.join(ROOT, 'weights', 'learned_eps', 'B_seed1_step300000'))
z = np.random.default_rng(2).standard_normal((100, 4)); s = LearnedScore(net, pc)(z, 1.0)
e = net.forward(pc.features(z, np.full(100, p.sigma(1.0))), params=net.E).astype(np.float64)
check('learned score = -eps_hat/sigma', np.abs(s + e / p.sigma(1.0)).max(), 1e-12)
# bit-exact retraining (first 10 000 steps of network B seed 1, single thread)
with tempfile.TemporaryDirectory() as td:
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', B3_CKPT='10000', B3_STOP='10000', B3_OUT=td)
    subprocess.run([sys.executable, os.path.join(ROOT, 'experiments', 'learned', 'train.py'), 'B', '300000', '1'], env=dict(env, CLD_WEIGHTING='eps'), check=True, capture_output=True)
    a, b = np.load(f'{td}/B_seed1_step10000.npz'), np.load(os.path.join(ROOT, 'weights', 'learned_eps', 'B_seed1_step10000.npz'))
    diff = float(max(np.abs(a[k] - b[k]).max() for k in a.files)); scale = float(max(np.abs(b[k]).max() for k in b.files))
    print(f"  retraining: max |difference| = {diff:.3g} ({'bit for bit' if diff == 0 else 'not bit for bit: other CPU or BLAS than the original machine'})")
    check('retraining B seed 1 to step 10 000 reproduces the stored weights (bit for bit on the original machine; relative difference <= 1e-3 on other CPUs)',
          diff / scale, 1e-3)
# exact-score path reproduces B2 (setting B and C, h ~ 0.002, N = 10 000, seed 0) within 0.005
for name in 'BC':
    f = f'samp_{name}_n1_k300000_s0_N10000.json'
    if os.path.exists(f):
        mine = json.load(open(f))['exact']['paired']
        b2 = [l for l in json.load(open(f'../map/steps_{name}.json'))['levels'] if abs(l['h'] - 0.002) < 3e-4][0]['paired']
        check(f'{name}: exact-score paired L reproduces B2 step study at h~0.002 (|diff| <= 0.005)', mine - b2, 0.005)

print('== B. pre-registered criteria (main runs: 2 networks x 3 sampling seeds, N = 20000)')
S = {}
for f in glob.glob('samp_*_k300000_s*_N20000.json'):
    x = json.load(open(f)); S.setdefault(x['setting'], []).append(x)
summ = {}
for name in 'BCD':
    R = S.get(name, [])
    if len(R) < 6: print(f'  {name}: only {len(R)} runs'); res.append(False); continue
    lp = np.array([x['learned']['paired'] for x in R]); ep = np.array([x['exact']['paired'] for x in R])
    se = lp.std(ddof=1) / np.sqrt(len(lp)); ratio = lp.mean() / ep.mean()
    dii = np.array([x['delta']['fixed'] for x in R]); di = np.array([x['delta']['naive'] for x in R]); diii = np.array([x['delta']['ode'] for x in R])
    print(f'  {name}: paired learned {lp.mean():.4f} +/- {se:.4f}, exact {ep.mean():.4f}, ratio {ratio:.3f}; '
          f'delta i {di.mean():+.4f}, ii {dii.mean():+.4f} (sd {dii.std(ddof=1):.4f}), iii {diii.mean():+.4f}')
    if name == 'D': continue
    check(f'S1 {name}: learned paired L significant (mean + 2 se < 0)', lp.mean() + 2 * se, 0.0, 'le')
    check(f'S1 {name}: learned/exact paired ratio in [0.75, 1.25]', ratio, (0.75, 1.25), 'in')
    check(f'S2 {name}: |delta_ii| <= 0.03', abs(dii.mean()), 0.03)
    check(f'S2 {name}: |delta_ii| < |learned paired L| / 3', abs(dii.mean()), abs(lp.mean()) / 3)
n = len(res); print(f'\n{sum(res)}/{n} checks passed'); sys.exit(0 if all(res) else 1)
