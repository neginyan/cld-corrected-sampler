"""Single entry point for every check of the paper
"When the blur is slower than the flow: clipped and corrected samplers on the critically damped Langevin flow".

    python verification/run_all.py              # all checks from the stored results and weights (about 20-40 min on 2 CPU cores)
    python verification/run_all.py --recompute  # also re-runs representative computations and compares them with the stored files

Steps:
  0. MANIFEST.sha256: SHA-256 of every file in results/, weights/ and the pre-registration originals.
  1. check_threshold.py   Sec. III, exact score, threshold          (stage B1)
  2. check_map.py         Sec. IV, map of the damage, mixtures      (stage B2)
  3. check_learned.py     Sec. V, learned score                     (stage B3)
  4. check_forgetting.py  Sec. VI and Obs. 9                         (stage B4)
  5. check_numbers.py     every number printed in paper/ against the results (stage B5)
Three checks of stage B1 are KNOWN FAILURES: they failed when first run, and their tolerances were fixed in advance, so
they are reported as they are and are not loosened (see the paper, Sec. III, and results/threshold).
Exit code 0 iff the manifest matches and no check fails other than the known failures (and, with --recompute, every
recomputation agrees with the stored files)."""
import os, sys, subprocess, hashlib, time, argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CHECKS = [('check_threshold.py', 'Sec. III  exact score, threshold (B1)'),
          ('check_map.py', 'Sec. IV   map of the damage (B2)'),
          ('check_learned.py', 'Sec. V    learned score (B3)'),
          ('check_forgetting.py', 'Sec. VI   forgetting; Obs. 9 score weighting (B4)'),
          ('check_numbers.py', 'paper     numbers and manuscript (B5)')]
KNOWN_FAILURES = ['(i) shrinks within-mode variance at r/k=0.95: mean + 2 se < 0',
                  'MMD2 (fixed) at the target-vs-target level',
                  'MMD2 (ode) at the target-vs-target level']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def check_manifest():
    lines = [l for l in open(os.path.join(ROOT, 'MANIFEST.sha256'), encoding='utf-8').read().splitlines() if l.strip()]
    listed = {}
    for l in lines:
        h, p = l.split('  ', 1); listed[p] = h
    bad = [p for p, h in listed.items() if not os.path.exists(os.path.join(ROOT, p)) or sha256(os.path.join(ROOT, p)) != h]
    present = set()
    for top in ['results', 'weights']:
        for dp, _, fs in os.walk(os.path.join(ROOT, top)):
            for f in fs: present.add(os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, '/'))
    extra = sorted(present - set(listed))
    return len(listed), bad, extra

def run_check(script):
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(ROOT, 'verification', script)], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout.splitlines()
    res = {'pass': 0, 'known': [], 'unexpected': [], 'sec': time.time() - t0, 'returncode': p.returncode}
    for l in out:
        if l.startswith('[PASS]'): res['pass'] += 1
        elif l.startswith('[FAIL]'):
            (res['known'] if any(k in l for k in KNOWN_FAILURES) else res['unexpected']).append(l)
        if l.startswith(('[PASS]', '[FAIL]', '==', '  ')) or 'checks passed' in l: print('    ' + l)
    if p.returncode != 0 and not res['unexpected'] and not res['known']:
        res['unexpected'].append(f'[CRASH] {script} exited with {p.returncode}: ' + p.stderr.strip().splitlines()[-1] if p.stderr.strip() else '')
    return res

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--recompute', action='store_true'); a = ap.parse_args()
    print('== 0. MANIFEST.sha256')
    n, bad, extra = check_manifest()
    print(f'    {n} files listed; mismatched or missing: {len(bad)}; unlisted files in results/ or weights/: {len(extra)}')
    for p in bad[:20]: print('    MISMATCH', p)
    for p in extra[:20]: print('    UNLISTED', p)
    manifest_ok = not bad and not extra
    summary = []
    for i, (script, label) in enumerate(CHECKS, 1):
        print(f'\n== {i}. {script}  ({label})')
        r = run_check(script); summary.append((script, label, r))
    rec = None
    if a.recompute:
        print('\n== 6. recompute.py (re-runs representative computations)')
        r = run_check('recompute.py'); summary.append(('recompute.py', 'recomputation vs stored files', r)); rec = r
    print('\n================ SUMMARY ================')
    print(f"{'check':22s} {'passed':>7s} {'known fail':>11s} {'unexpected':>11s} {'time':>8s}")
    tp = tk = tu = 0
    for script, label, r in summary:
        print(f"{script:22s} {r['pass']:7d} {len(r['known']):11d} {len(r['unexpected']):11d} {r['sec']:7.0f}s")
        tp += r['pass']; tk += len(r['known']); tu += len(r['unexpected'])
    print(f"{'TOTAL':22s} {tp:7d} {tk:11d} {tu:11d}   ({tp + tk + tu} checks)")
    print(f"manifest: {'OK' if manifest_ok else 'MISMATCH'} ({n} files)")
    for script, label, r in summary:
        for l in r['known']: print(f'KNOWN FAILURE ({script}): {l}')
        for l in r['unexpected']: print(f'UNEXPECTED FAILURE ({script}): {l}')
    nm = [r for s, _, r in summary if s == 'check_numbers.py'][0]
    print(f"paper numbers (check_numbers.py): {nm['pass']} passed, {len(nm['unexpected'])} failed")
    ok = manifest_ok and tu == 0
    print('RESULT:', 'OK (all checks pass except the known failures)' if ok else 'NOT OK')
    sys.exit(0 if ok else 1)
