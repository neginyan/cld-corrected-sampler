"""Optional recomputation (python verification/run_all.py --recompute, or run directly).
Copies the repository into a temporary folder, re-runs representative computations there with the stored inputs, and
compares the outputs with the stored files.  The stored results/ are not touched.
  1. Sec. III: one exact-score sweep job (r/kappa = 2.0, seed 0)              -> results/threshold/main_r2.0_s0.json
  2. Sec. IV : decomposition of the damage by sigma band (one-mode theory)    -> results/map/decompose.json
  3. Sec. V  : score error of the final network B, seed 1                     -> results/learned/err_B_n1_k300000.json
  4. Sec. VI : theory of the forgetting ratio                                 -> results/forgetting/theory_b4.json
  5. paper   : the four figures, pixel by pixel                               -> paper/figures/fig*.png
Full re-training (about 23 min per network on one core) and the full sweeps are described in experiments/PIPELINE.md."""
import os, sys, json, shutil, subprocess, tempfile
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
res = []
def check(name, ok, info=''):
    res.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

def same(a, b, path=''):
    """Compare two JSON values: numbers to a relative 1e-8, everything else exactly; timing keys are ignored."""
    if isinstance(a, dict):
        keys = set(a) | set(b)
        return all(same(a.get(k), b.get(k), path + '/' + k) for k in keys if k not in ('sec',))
    if isinstance(a, list): return len(a) == len(b) and all(same(x, y, path) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (np.isfinite(a) and np.isfinite(b)): return (np.isnan(a) and np.isnan(b)) or a == b
        return abs(a - b) <= 1e-8 * max(abs(a), abs(b)) + 1e-12
    return a == b

with tempfile.TemporaryDirectory() as td:
    for d in ['src', 'experiments', 'results', 'weights', 'paper']:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(td, d))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    jobs = [('Sec. III: sweep job r/k=2.0 seed 0', ['experiments/threshold/run_sweep.py', '2.0', '0'], 'results/threshold/main_r2.0_s0.json'),
            ('Sec. IV: damage decomposition', ['experiments/map/decompose.py'], 'results/map/decompose.json'),
            ('Sec. V: score error of network B seed 1', ['experiments/learned/score_error.py', 'B', '1', '300000'], 'results/learned/err_B_n1_k300000.json'),
            ('Sec. VI: forgetting theory', ['experiments/forgetting/theory_runs.py'], 'results/forgetting/theory_b4.json')]
    for name, cmd, out in jobs:
        p = subprocess.run([sys.executable] + [os.path.join(td, cmd[0])] + cmd[1:], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            check(f'{name}: recomputation runs', False, p.stderr.strip().splitlines()[-1] if p.stderr else ''); continue
        check(f'{name}: recomputed {out} agrees with the stored file (relative 1e-8)',
              same(json.load(open(os.path.join(td, out))), json.load(open(os.path.join(ROOT, out)))))
    import matplotlib.image as mpimg
    figdir = os.path.join(td, 'figs')
    p = subprocess.run([sys.executable, os.path.join(td, 'paper', 'figures', 'make_figures.py'), figdir], env=env, capture_output=True, text=True)
    for f in ['fig1_threshold', 'fig2_map', 'fig3_learned', 'fig4_forgetting']:
        ok = p.returncode == 0 and os.path.exists(os.path.join(figdir, f + '.png'))
        if ok:
            a, b = mpimg.imread(os.path.join(figdir, f + '.png')), mpimg.imread(os.path.join(ROOT, 'paper', 'figures', f + '.png'))
            ok = a.shape == b.shape and float(np.abs(a - b).max()) == 0.0
        check(f'paper figure {f}.png regenerated from results/ is pixel-identical', ok)
n = len(res); print(f'\n{sum(res)}/{n} checks passed'); sys.exit(0 if all(res) else 1)
