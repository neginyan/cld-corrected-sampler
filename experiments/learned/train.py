"""Train the score network.  CLD_WEIGHTING=eps (default, Sec. V) or CLD_WEIGHTING=score (loss weight (sigma_min/sigma)^2, Obs. 9).
Train the score network for one setting (denoising score matching, eps form; paper D recipe).
usage: OMP_NUM_THREADS=1 python experiments/learned/train.py <setting> <steps> [seed]
Checkpoints (EMA weights used for sampling) at every CKPT steps: weights/learned_<weighting>/<setting>_seed<seed>_step<k>.npz/.json"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
WEIGHTING = os.environ.get('CLD_WEIGHTING', 'eps')            # 'eps' (Sec. V) or 'score' (Obs. 9)
STAGE = 'learned' if WEIGHTING == 'eps' else 'score_weighting'
WEIGHTS = os.path.join(ROOT, 'weights', 'learned_' + WEIGHTING)
os.chdir(os.path.join(ROOT, 'results', STAGE))   # relative paths below are inside results/<stage>/
import sys, os, json, time
import numpy as np
from learned import *

name, steps = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CKPT = int(os.environ.get('B3_CKPT', 20000)); BS = 1024
STOP = int(os.environ.get('B3_STOP', 10**12)); OUTDIR = os.environ.get('B3_OUT', WEIGHTS)   # STOP/OUTDIR: for the retrain check only
p = setting(name); data = train_data(0); pc = Precond(p, data); net = new_net(seed)
rng = np.random.default_rng(10_000 + seed); ls0, ls1 = np.log(p.s0), np.log(p.sT); log = []; run = None; t0 = time.time()
while net.step < min(steps, STOP):
    half = BS // 2
    lsig = np.concatenate([rng.uniform(ls0, ls1, half), rng.uniform(ls0, ls0 + np.log(10.0), BS - half)]); sig = np.exp(lsig)
    z0 = data[rng.integers(0, len(data), BS)]; eps = rng.standard_normal((BS, 4))
    zt = np.einsum('nij,nj->ni', pc.M_of(sig), z0) + sig[:, None] * eps
    out, cache = net.forward(pc.features(zt, sig), keep=True); d = out.astype(np.float64) - eps
    if WEIGHTING == 'eps':
        loss = float((d**2).sum(1).mean())
    else:
        wt = (p.s0 / sig)**2                                   # 'score' weighting (paper D), sigma_0 = sigma_min
        loss = float((wt * (d**2).sum(1)).mean())
    st = net.step; lr = 2e-3 * min(1.0, (st + 1) / 1000) * (1e-5 / 2e-3 + (1 - 1e-5 / 2e-3) * 0.5 * (1 + np.cos(np.pi * st / steps)))
    if WEIGHTING == 'eps':
        net.adam(net.backward(cache, (2 * d / BS).astype(np.float32)), lr)
    else:
        net.adam(net.backward(cache, (2 * wt[:, None] * d / BS).astype(np.float32)), lr)
    run = loss if run is None else 0.99 * run + 0.01 * loss
    if net.step % 1000 == 0: log.append([net.step, run])
    if net.step % CKPT == 0 or net.step == steps or net.step == STOP:
        net.save(f'{OUTDIR}/{name}_seed{seed}_step{net.step}', {'setting': name, 'seed': seed, 'steps_total': steps, 'log': log})
        print(f'{name} seed{seed} step {net.step} loss {run:.5f} ({time.time() - t0:.0f} s)', flush=True)
