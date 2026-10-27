"""Stage B1 verification.  Each check has a tolerance and prints how much of it was used.
Run from the project root:  python verification/check_threshold.py   (exit code 0 iff all checks pass).
Section A: analytic statements (fast).  Section B: numbers read back from results/ (needs experiments/threshold/run_sweep.py jobs + experiments/threshold/analyze.py)."""
import sys, os, json
import numpy as np
from scipy.linalg import expm
from scipy.special import logsumexp
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'threshold'))   # relative paths below are inside results/<stage>/
from cld_mixture import *
from psdt import cld_B, kappa_of, Process

res = []
def check(name, value, tol, kind='abs'):
    """kind 'abs': pass iff |value| <= tol.  'le': value <= tol.  'ge': value >= tol (used = tol/value)."""
    if kind == 'abs': ok, used = abs(value) <= tol, abs(value) / tol if tol else (0.0 if value == 0 else np.inf)
    elif kind == 'le': ok, used = value <= tol, (max(value, 0) / tol if tol > 0 else (0.0 if value <= 0 else np.inf))
    else: ok, used = value >= tol, (tol / value if value > 0 else np.inf)
    res.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: value {value:.3e}, tol {tol:.1e} ({kind}), used {100 * used:.1f}%")
I4 = np.eye(4)
print('== A. analytic checks')
# A1. CLD defaults and kappa
b = B_CLD[:2, :2]
check('B_CLD = beta[[0, M^-1],[-1, -Gamma M^-1]] with beta=4, M^-1=4, Gamma=1', np.abs(b - 4 * np.array([[0, 4], [-1, -4]])).max(), 1e-14)
check('critical damping Gamma^2 = 4M', GAMMA_DAMP**2 - 4 / M_INV, 1e-14)
check('Sym B eigenvalues = 4*{0.5, -4.5}', np.abs(np.linalg.eigvalsh((b + b.T) / 2) - np.array([-18, 2])).max(), 1e-12)
check('kappa = 2', KAPPA - 2, 1e-12)
check('kappa = beta/2 for M=1/4, Gamma=1 (paper code convention), beta in {1,4,7}',
      max(abs(kappa_of(cld_B(beta=bb)) - bb / 2) for bb in [1.0, 4.0, 7.0]), 1e-12)
check('M=1 (Gamma=2, critical) gives kappa = 0', kappa_of(cld_B(beta=4.0, M=1.0, Gamma=2.0)), 1e-12)
check('B has the double eigenvalue -8 (critical damping)', np.abs(np.linalg.eigvals(b) + 8).max(), 1e-6)
check('initial velocity variance gamma/m_inv = 0.01', V0_VAR - 0.01, 1e-15)
# A2. schedule: equal total blur for every r
for rr in [0.5, 0.95, 1.0, 2.0]:
    p = process(rr)
    check(f'sigma(0)=sigma_min, sigma(T)=sigma_max, T = ln(smax/smin)/r (r/k={rr})',
          max(abs(p.sigma(0) - S_MIN), abs(p.sigma(p.T) / S_MAX - 1), abs(p.T * p.r - np.log(S_MAX / S_MIN))), 1e-12)
# A3. G formula and sign change
for rr in [0.5, 0.95, 1.3]:
    p = process(rr); t, h = 1.1, 1e-6
    C = lambda s: p.sigma(s)**2 * I4
    G_fd = (C(t + h) - C(t - h)) / (2 * h) - p.B @ C(t) - C(t) @ p.B.T
    check(f'G = dC/dt - BC - CB^T = 2 sigma^2 (rI - SymB) (r/k={rr})', np.abs(G_fd - p.G(t)).max() / np.abs(G_fd).max(), 1e-6)
ok = all((np.linalg.eigvalsh(process(rr).G(0.5)).min() >= -1e-14) == (rr >= 1) for rr in [0.5, 0.9, 0.99, 1.0, 1.01, 2.0])
check('G >= 0 iff r >= kappa (6 ratios)', 0.0 if ok else 1.0, 0.0)
# A4. exact mixture score = gradient of the mixture log-density
w, mu, S = mixture(); p = process(0.7); sc = ExactScore(p, w, mu, S); t = 0.4
def logp(z):
    M = expm(p.B * t); C = M @ S[0] @ M.T + p.sigma(t)**2 * I4; iC = np.linalg.inv(C)
    return logsumexp([np.log(w[k]) - 0.5 * (z - M @ mu[k]) @ iC @ (z - M @ mu[k]) for k in range(len(w))])
zz = np.random.default_rng(3).standard_normal((6, 4)) * 0.8; hh = 1e-5
fd = np.array([[(logp(z + hh * e) - logp(z - hh * e)) / (2 * hh) for e in I4] for z in zz])
check('exact score = finite-difference gradient of log p_t (6 points)', np.abs(fd - sc(zz, t)).max() / np.abs(fd).max(), 1e-7)
# A5. exact prior: Monte Carlo moments of p_T
p = process(0.5); zT = prior_exact(p, w, mu, S, 400000, np.random.default_rng(4))
MT = expm(p.B * p.T); m_all = mu.mean(0); Cmix = S[0] + np.cov(mu.T, bias=True)
Cth = MT @ Cmix @ MT.T + p.sigma(p.T)**2 * I4
check('prior p_T: MC covariance matches theory (relative, 4e5 samples)', np.abs(np.cov(zT.T) - Cth).max() / Cth.max(), 0.02)
# A6. Gaussian-moment theory (paper Prop. 6 / Thm 4): (ii), (iii) exact for all r, (i) exact iff r >= kappa, (i) shrinks below
for rr in [0.5, 0.9, 1.0, 1.5]:
    p = process(rr)
    for md in ['ode', 'fixed', 'naive']:
        m_, Sg, (mt, Ct) = gauss_output(p, mu[0], S[0], md)
        err = max(np.abs(m_ - mt).max(), np.abs(Sg - Ct).max() / Ct.max())
        if md != 'naive' or rr >= 1:
            check(f'Gaussian theory: {md} output exact (r/k={rr})', err, 1e-7)
        else:
            D = Sg - Ct
            check(f'Gaussian theory: naive mean exact (r/k={rr})', np.abs(m_ - mt).max(), 1e-7)
            check(f'Gaussian theory: naive Delta <= 0 (r/k={rr})', np.linalg.eigvalsh(D).max() / Ct.max(), 1e-8, 'le')
            check(f'Gaussian theory: naive tr Delta < 0 (r/k={rr})', -np.trace(D), 1e-9, 'ge')
# A7. drift code: naive and fixed are the same function for r >= kappa
for rr in [1.0, 1.5]:
    p = process(rr); z = np.random.default_rng(5).standard_normal((50, 4)); s = sc(z, 0.3)
    check(f'naive drift == fixed drift when r >= kappa (r/k={rr})', np.abs(drift_fn(p, 'naive')(z, 0.3, s) - drift_fn(p, 'fixed')(z, 0.3, s)).max(), 1e-12)
# A8. Brownian coupling across step levels
a = np.random.default_rng(1).standard_normal((4, 3, 2)); r_ = np.random.default_rng(1); bq = np.stack([r_.standard_normal((3, 2)) for _ in range(4)])
check('block noise draw == sequential draws (coupled refinement)', np.abs(a - bq).max(), 0.0)

print('== B. results')
if not os.path.exists('summary_B1.json'):
    print('summary_B1.json missing: run the experiments/threshold/run_sweep.py jobs and experiments/threshold/analyze.py first'); res.append(False)
else:
    sm = json.load(open('summary_B1.json')); R = sm['ratios']
    P = sm['paired_log_within']; above = [i for i, r in enumerate(R) if r >= 1]; below = [i for i, r in enumerate(R) if r < 1]
    check('(i) - (ii) paired deviation is exactly 0 for r >= kappa (all seeds; identity, not a finding)',
          max(abs(v) for i in above for v in P['all'][i]), 0.0)
    for i in below:
        check(f'(i) shrinks within-mode variance at r/k={R[i]}: mean + 2 se < 0', P['mean'][i] + 2 * P['se'][i], 0.0, 'le')
    for i in below:
        th = sm['theory_log_within'][str(R[i])]
        check(f'measured shrinkage vs one-mode Gaussian theory at r/k={R[i]} (|diff| <= 3 se + 25% of theory)',
              abs(P['mean'][i] - th), 3 * P['se'][i] + 0.25 * abs(th))
    tf = sm['threshold_fit']
    check('threshold fit: |kappa_fit/kappa - 1| within 3 fit s.e.', abs(tf['kappa_fit_over_kappa'] - 1), 3 * tf['kappa_fit_se'])
    check('threshold fit: |kappa_fit/kappa - 1| <= 0.05', abs(tf['kappa_fit_over_kappa'] - 1), 0.05)
    check('threshold fit: amplitude A within 0.25 of 1', abs(tf['A'] - 1), 0.25)
    for md in ['fixed', 'ode']:
        L = sm[f'log_within_ratio_{md}']
        for i, rr in enumerate(R):
            check(f'({"ii" if md == "fixed" else "iii"}) within-mode variance correct at r/k={rr} (|log ratio| <= 3 se + 0.003)',
                  abs(L['mean'][i]), 3 * L['se'][i] + 0.003)
    for md in ['naive', 'fixed', 'ode']:
        check(f'mode weights: max |w_k - 1/8| over all r, seeds ({md}) <= 0.01', max(sm[f'max_weight_dev_{md}']['max']), 0.01, 'le')
    for md in ['fixed', 'ode']:
        F = sm['mmd2_floor']; M_ = sm[f'mmd2_{md}']
        check(f'MMD2 ({md}) at the target-vs-target level (max excess over floor <= 3e-4)',
              max(m - f for m, f in zip(M_['mean'], F['mean'])), 3e-4, 'le')
    # step study
    for rr in [0.5, 0.9]:
        L = json.load(open(f'steps_r{rr}_s0.json'))['levels']
        e = [abs(np.log(l['fixed']['within_ratio'])) for l in L]
        check(f'step study r/k={rr}: (ii) error at h=0.001 smaller than at h=0.008 by >= 4x', e[0] / e[-1], 4.0, 'ge')

print('== C. supplements (added after section B was first run; C1 is reference numbers only, C2 is a check whose tolerance was fixed before its run)')
if os.path.exists('mmd_null_B1.json'):
    mn = json.load(open('mmd_null_B1.json')); rc = json.load(open('mmd_refcheck_B1.json'))
    print(f"  C1 MMD null (100 target pairs): mean {mn['null_mean']:.2e}, sd {mn['null_sd']:.2e}, q95 {mn['null_q95']:.2e}")
    print(f"  C1 fresh target vs the 5 reference samples used in the sweep: mean {rc['all_mean']:.2e} (refs are atypical)")
    for md in ['naive', 'fixed', 'ode']:
        for part in ['r_below', 'r_at_or_above']:
            x = mn[f'{md}_{part}']; print(f"  C1 {md:5s} {part:14s}: mean MMD2 {x['mean']:.2e}, mean percentile in null {x['mean_percentile_in_null']:.2f}")
import glob as _g
for rr in [0.5, 0.8]:
    fs = sorted(_g.glob(f'single_r{rr}_s*.json'))
    if not fs: continue
    v = np.array([json.load(open(f))['extrap'] for f in fs]); th = json.load(open(fs[0]))['theory_log_within']
    se = v.std(ddof=1) / np.sqrt(len(v))
    check(f'C2 one isolated mode, r/k={rr}: paired shrinkage = Gaussian theory (|diff| <= 3 se + 10% of theory; {len(v)} seeds)',
          abs(v.mean() - th), 3 * se + 0.1 * abs(th))

n = len(res); print(f'\n{sum(res)}/{n} checks passed'); sys.exit(0 if all(res) else 1)
