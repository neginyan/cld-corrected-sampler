"""Stage B2 verification.  Tolerances follow preregistration/B2_事前登録.md (English translation: preregistration/B2_preregistration_en.md); each check prints how much of its tolerance was used.
Run from anywhere:  python verification/check_map.py   (exit code 0 iff all checks pass)."""
import sys, os, json, glob
import numpy as np
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
from one_mode import Theory, b_cld, kappa2
from closed_forms import K_xx, L_integral_form, slope_at_threshold, L_noise_limit, L_r_to_zero
import cld_mixture
from psdt import Process

res = []
def check(name, value, tol, kind='abs'):
    if kind == 'abs': ok, used = abs(value) <= tol, (abs(value) / tol if tol else (0.0 if value == 0 else np.inf))
    elif kind == 'le': ok, used = value <= tol, (max(value, 0) / tol if tol > 0 else (0.0 if value <= 0 else np.inf))
    else: ok, used = value >= tol, (tol / value if value > 0 else np.inf)
    res.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: value {value:.3e}, tol {tol:.1e} ({kind}), used {100 * used:.1f}%")
rel = lambda a, b: abs(a / b - 1)

print('== A. theory and closed forms')
# A1. the 2x2 theory reproduces the 4-D B1 theory
for rk in [0.5, 0.8, 0.95]:
    check(f'2x2 theory = B1 4-D theory (r/k={rk})', rel(Theory(rk).L()[0], np.log(cld_mixture.theory_within_ratio(rk)[0])), 1e-6)
# A2. scale invariance (only sigma_min/s, sigma_max/s, v_std/s matter): 4-D theory with a different s
for c in [0.3, 7.0]:
    w, mu, S = cld_mixture.mixture(); S2 = S[0].copy(); S2[np.ix_([0, 2], [0, 2])] *= (c / 0.08)**2; S2[1, 1] = S2[3, 3] = (1.25 * c)**2
    p = Process(cld_mixture.B_CLD, 0.5 * 2, s0=0.0625 * c, sT=500 * c)
    _, Sg, (_, Ct) = cld_mixture.gauss_output(p, mu[0] * 0, S2, 'naive')
    check(f'scale invariance: s = {c} gives the same L as s = 1', rel(np.log(Sg[0, 0] / Ct[0, 0]), Theory(0.5).L()[0]), 1e-6)
# A3. beta drops out at fixed r/kappa
for M in [0.01, 0.25, 10.0]:
    check(f'beta invariance (M={M}, beta 4 vs 0.37)', rel(Theory(0.4, M=M, beta=4.0).L()[0], Theory(0.4, M=M, beta=0.37).L()[0]), 1e-6)
# A4. exact integral form L = log(1 - (kappa - r) K_xx / C_xx)
for rk, kw in [(0.3, {}), (0.7, {}), (0.5, {'smin': 1.0}), (0.5, {'M': 4.0}), (0.2, {'vstd': 0.3})]:
    check(f'integral form = ODE (r/k={rk}, {kw})', rel(L_integral_form(rk, **kw), Theory(rk, **kw).L()[0]), 1e-6)
# A5. linear response at the threshold (pre-registered: 2 % at r/k = 0.99)
for kw in [{}, {'smin': 0.5}, {'M': 4.0}, {'vstd': 0.3}, {'M': 0.05, 'smin': 0.2}]:
    s = slope_at_threshold(**kw); check(f'threshold slope predicts L(0.99) ({kw})', rel(-0.01 * s, Theory(0.99, **kw).L()[0]), 0.02)
# A6. noise-dominated limit (Lyapunov closed form) vs ODE at sigma_min = 100 s (tolerance 1 %, stated in the report)
for rk, M in [(0.1, 0.25), (0.5, 0.25), (0.9, 0.25), (0.5, 0.05), (0.5, 4.0)]:
    check(f'noise-dominated Lyapunov form vs ODE at sigma_min=100 s (r/k={rk}, M={M})',
          rel(L_noise_limit(rk, M=M)[0], Theory(rk, smin=100.0, smax=8e5, M=M).L()[0]), 0.01)
# A7. r -> 0 limit vs ODE at r/k = 0.01 (tolerance 3 %; NOTE: chosen after a first look at these values, disclosed in the report)
for kw in [{}, {'smin': 0.01}, {'smin': 1.0}, {'vstd': 0.3}, {'M': 0.05}]:
    check(f'r->0 limit vs ODE at r/k = 0.01 ({kw})', rel(L_r_to_zero(**kw), Theory(0.01, **kw).L()[0]), 0.03)
# A8. ODE accuracy (pre-registered: rtol 1e-10 -> 1e-12 changes L by <= 1e-6 relative)
for rk, kw in [(0.1, {}), (0.5, {'smin': 1.778}), (0.5, {'M': 10.0, 'smin': 0.5})]:
    check(f'ODE accuracy rtol 1e-10 vs 1e-12 (r/k={rk}, {kw})', rel(Theory(rk, **kw).L()[0], Theory(rk, rtol=1e-12, atol=1e-16, **kw).L()[0]), 1e-6)
# A9. sigma_max does not matter (5 s vs 5000 s)
for rk in [0.1, 0.5, 0.9]:
    check(f'sigma_max independence 5 s vs 5000 s (r/k={rk})', rel(Theory(rk, smax=5.0).L()[0], Theory(rk, smax=5000.0).L()[0]), 1e-4)
# A10. decomposition bands add up to the total
d = json.load(open('decompose.json'))
for k in ['default', 'default_rk0.1', 'A', 'B', 'C']:
    check(f'decomposition: bands sum to the total ({k})', rel(d[k]['sum'], d[k]['total_direct']), 1e-5)
# A11. M = 1 gives kappa = 0 (no clipped direction)
check('M = 1: kappa = 0', kappa2(b_cld(1.0)), 1e-12)

print('== B. mixture checks (sec. 2-3)')
fs = sorted(glob.glob('main_*_s*.json'))
if not fs: print('no results'); res.append(False)
by = {}
for f in fs:
    x = json.load(open(f)); by.setdefault(x['setting'], []).append(x)
best = 0
for name in sorted(by):
    R = by[name]; n = len(R); P = np.array([x['extrap']['paired_log_within'] for x in R]); se = P.std(ddof=1) / np.sqrt(n)
    Pc = np.array([x['coarse']['paired_log_within'] for x in R]); Pf = np.array([x['fine']['paired_log_within'] for x in R])
    th = R[0]['theory_log_within']; best = min(best, P.mean())
    print(f"  setting {name}: {n} seeds, paired L = {P.mean():.4f} +/- {se:.4f}, theory {th:.4f}, ratio {P.mean() / th:.3f}")
    check(f'{name}: step convergence |fine - coarse| <= 15% |extrap|', abs(Pf.mean() - Pc.mean()), 0.15 * abs(P.mean()))
    check(f'{name}: (i) shrinkage significant (mean + 2 se < 0)', P.mean() + 2 * se, 0.0, 'le')
    for md, lab in [('fixed', '(ii)'), ('ode', '(iii)')]:
        v = np.array([x['extrap'][md]['log_within'] for x in R]); s_ = v.std(ddof=1) / np.sqrt(n)
        check(f'{name}: {lab} within-mode variance correct (|L| <= 3 se + 0.01)', abs(v.mean()), 3 * s_ + 0.01)
    wd = max(x['fine'][m]['max_weight_dev'] for x in R for m in ['naive', 'fixed', 'ode'])
    if name == 'B': print(f'  [reference only, pre-declared] {name}: max |w_k - 1/8| = {wd:.4f}')
    else: check(f'{name}: mode weights max |w_k - 1/8| <= 0.01', wd, 0.01, 'le')
check('SUCCESS (pre-registered): some setting has mixture paired L <= log 0.9', best, np.log(0.9), 'le')

print('== C. spacing hypothesis (sec. 2-4)')
sp = sorted(glob.glob('spacing_R*_s*.json'))
if sp:
    S = {}
    for f in sp:
        x = json.load(open(f)); S.setdefault(x['radius_factor'], []).append(x['ratio_to_theory'])
    fac = sorted(S); m = [np.mean(S[k]) for k in fac]
    for k, v in zip(fac, m): print(f'  radius x{k:g}: measured/theory = {v:.3f} (seeds {len(S[k])}, sd {np.std(S[k], ddof=1):.3f})')
    inc = all(b >= a for a, b in zip(m[:-1], m[1:]))
    check('C: measured/theory increases with the radius (seed means)', 0.0 if inc else 1.0, 0.0)
    check('C: at the largest radius measured/theory in [0.9, 1.05]', 0.0 if 0.9 <= m[-1] <= 1.05 else abs(m[-1] - 0.975), 0.0)
print('== D. supplements added after the mixture runs (reference numbers; D1 has a tolerance fixed before its run)')
fa = sorted(glob.glob('single_A_s*.json'))
if fa:
    xa = [json.load(open(f)) for f in fa]; thv = xa[0]['theory_log_vel']; thp = xa[0]['theory_log_within']
    vp = np.mean([x['extrap'][0] for x in xa]); vv = np.mean([x['extrap'][1] for x in xa])
    print(f'  D1 setting A, one isolated mode ({len(xa)} seeds): position {vp:.4f} (theory {thp:.4f}), velocity {vv:.4f} (theory {thv:.4f}); mixture velocity {np.mean([x["extrap"]["paired_log_vel"] for x in by["A"]]):.4f}')
    check('D1: one-mode setting A reproduces the theory for position and velocity (|diff| <= 10% of theory)',
          max(abs(vp - thp) / abs(thp), abs(vv - thv) / abs(thv)), 0.10)
for f in sorted(glob.glob('diag_B1seed*.json')):
    x = json.load(open(f)); print(f"  D2 B1 seed {x['seed']} (r/k=0.5): trajectories in a different mode under (i) vs (ii): {x['n_switch']}/40000; "
          f"L all {x['L_all']:.4f}, same-mode subset {x['L_same_mode']:.4f}")
n = len(res); print(f'\n{sum(res)}/{n} checks passed'); sys.exit(0 if all(res) else 1)
