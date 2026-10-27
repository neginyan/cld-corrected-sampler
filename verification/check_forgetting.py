"""Stage B4 verification (tolerances from preregistration/B4_事前登録.md (English translation: preregistration/B4_preregistration_en.md)).  python verification/check_forgetting.py  (exit 0 iff all pass)."""
import sys, os, json, glob, subprocess, tempfile
import numpy as np
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'forgetting'))   # relative paths below are inside results/<stage>/
from one_mode import Theory
from forgetting import run, generator, forgetting_rates

res = []
def check(name, value, tol, kind='abs'):
    if kind == 'abs': ok, used = abs(value) <= tol, (abs(value) / tol if tol else (0.0 if value == 0 else np.inf))
    elif kind == 'ge': ok, used = value >= tol, (tol / value if value > 0 else np.inf)
    res.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: value {value:.4g}, tol {tol:.3g} ({kind}), used {100 * used:.1f}%")

print('== A. theory (sec. 2-1)')
# A1. with eta = 0 the deviation equation reproduces the B2 theory for (i)  (E-form, R = -G_-)
for rk, kw in [(0.3, {'smin': 1.778}), (0.5, {'M': 10.0, 'smin': 0.5}), (0.5, {})]:
    th = Theory(rk, **kw); S = run(th, 'naive', lambda s: 0.0)
    check(f'eta = 0: (i) output = B2 theory (r/k={rk}, {kw})', np.log(S[0, 0] / th.C(0)[0, 0]) / th.L()[0] - 1, 1e-6)
# A2. the residual R vanishes identically for (ii) and (iii) at eta = 0 (so they are exact; checked at 5 times)
th = Theory(0.3, smin=1.778); mx = 0
for t in np.linspace(0.1, th.T - 0.1, 5):
    Gp, Gm = th.Gparts(t); G = Gp - Gm
    for K, D in [(0.5 * (G + Gp), Gp), (0.5 * G, 0 * Gp)]:
        mx = max(mx, np.abs(D + G - 2 * K).max() / np.abs(G).max())
check('eta = 0: residual R = D + G - (K + K^T) is zero for (ii), (iii)', mx, 1e-12)
# A3. noise-dominated generator: closed form vs direct differentiation of F = E / sigma^2
for mode in ['naive', 'fixed', 'ode']:
    th = Theory(0.5); s = 3.0; h = 1e-6
    # direct: A = -B - K P with P = I / sigma^2;  d(E/sigma^2)/dtau = (A + rI)F + F(A + rI)^T, since d sigma^2/dtau = -2 r sigma^2
    Gp, Gm = th.Gparts(0.0); sc = s**2 / th.sig2(0.0); Gp, Gm = Gp * sc, Gm * sc; G = Gp - Gm
    K = {'naive': Gp, 'fixed': 0.5 * (G + Gp), 'ode': 0.5 * G}[mode]; Mdir = -th.b - K / s**2 + th.r * np.eye(2)
    check(f'noise-dominated generator closed form ({mode})', np.abs(generator(th, mode, s) - Mdir).max() / np.abs(Mdir).max(), 1e-6)
# A4. ODE generator is skew (pure rotation, nothing forgotten); (ii) forgetting rate = r - w_min (in rate units)
for rk, kw in [(0.3, {}), (0.5, {'M': 10.0}), (0.5, {})]:
    th = Theory(rk, **kw); Mo = generator(th, 'ode', 1.0)
    check(f'ODE generator is skew-symmetric (r/k={rk}, {kw})', np.abs(Mo + Mo.T).max() / np.abs(Mo).max(), 1e-12)
    fr = forgetting_rates(th, 'fixed')
    check(f'(ii) forgetting rate of F = r - w_min (r/k={rk}, {kw})', fr.max() / -(th.r - th.w[0]) - 1, 1e-9)
# A5. pre-registered: theory ratio Delta_iii / Delta_ii >= 2 in B and C (final networks)
T = json.load(open('theory_b4.json'))
for n in 'BC':
    d = [x for x in T['deltas'] if x['setting'] == n and x['step'] == 300000 and x['net'] == 1][0]
    check(f'pre-registered: theory ratio Delta_iii/Delta_ii >= 2 ({n})', d['ode'] / d['fixed'], 2.0, 'ge')

print("== B. 'score'-weighted training (sec. 2-2)")
with tempfile.TemporaryDirectory() as td:
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', B3_CKPT='10000', B3_STOP='10000', B3_OUT=td)
    subprocess.run([sys.executable, os.path.join(ROOT, 'experiments', 'learned', 'train.py'), 'D', '300000', '1'], env=dict(env, CLD_WEIGHTING='score'), check=True, capture_output=True)
    a, b = np.load(f'{td}/D_seed1_step10000.npz'), np.load(os.path.join(ROOT, 'weights', 'learned_score', 'D_seed1_step10000.npz'))
    check("retraining ('score' weight) D seed 1 to step 10 000 reproduces the stored weights bit for bit",
          float(max(np.abs(a[k] - b[k]).max() for k in a.files)), 0.0)
E = [json.load(open(f'../score_weighting/err_D_n{n}_k300000.json')) for n in [1, 2]]
e_score = np.mean([e['e_near_smin'] for e in E]); e_eps = np.mean([json.load(open(f'../learned/err_D_n{n}_k300000.json'))['e_near_smin'] for n in [1, 2]])
print(f"  pre-registered (a): error near sigma_min 'score' {e_score:.3f} vs 'eps' {e_eps:.3f} -> halved: {e_score <= e_eps / 2}")
print(f"  large-sigma error ('score' weight, max over sigma): {max(max(o['e_all'] for o in e['by_sigma']) for e in E):.2f}")
R = [json.load(open(f)) for f in sorted(glob.glob('../score_weighting/samp_D_n*_k300000_s*_N20000.json'))]
nf = sum(1 for x in R for m in ['naive', 'fixed', 'ode'] if not np.isfinite(x['learned'][m]['L']))
print(f"  pre-registered (b): learned-score runs with a non-finite result: {nf} of {3 * len(R)} -> |delta_ii| not defined")
print(f'  runs: {len(R)}')
if len(R) == 6:
    lp = np.array([x['learned']['paired'] for x in R]); ep = np.array([x['exact']['paired'] for x in R])
    print(f"  paired learned {lp.mean():.4f} +/- {lp.std(ddof=1) / np.sqrt(6):.4f}, exact {ep.mean():.4f}, ratio {lp.mean() / ep.mean():.3f}")
    print('  delta i/ii', np.mean([x['delta']['naive'] for x in R]), np.mean([x['delta']['fixed'] for x in R]))
n = len(res); print(f'\n{sum(res)}/{n} checks passed'); sys.exit(0 if all(res) else 1)
