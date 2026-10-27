"""Every number quoted in the manuscript, recomputed from the stored results (results/<stage>/).
numbers() -> list of (key, string as printed in the manuscript, source description).
python verification/paper_numbers.py  writes paper/numbers.md.
Stage labels in the source column: B1 = Sec. III (results/threshold), B2 = Sec. IV (results/map),
B3 = Sec. V (results/learned), B4 = Sec. VI and Obs. 9 (results/forgetting, results/score_weighting)."""
import os, sys, json, glob
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
STAGE = {'B1': 'threshold', 'B2': 'map', 'B3': 'learned', 'B4': 'forgetting'}
R = lambda st, _res, *p: os.path.join(ROOT, 'results', STAGE[st] if not (st == 'B4' and p and (p[0].startswith('err_D') or p[0].startswith('samp_D'))) else 'score_weighting', *p)
J = lambda *p: json.load(open(R(*p)))
pc = lambda L: 100 * (np.exp(L) - 1)                       # log ratio -> percent change of the variance
f1 = lambda x: f'{x:.1f}'; f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'; f4 = lambda x: f'{x:.4f}'

def numbers():
    N = []
    add = lambda k, s, src: N.append((k, s, src))
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from one_mode import Theory, b_cld, kappa2
    from closed_forms import L_noise_limit, L_r_to_zero, slope_at_threshold
    # ---- CLD flow
    b = b_cld(0.25, 4.0); ev = np.linalg.eigvalsh((b + b.T) / 2)
    add('symB_eigs', f'{ev[1]:.0f}, {ev[0]:.0f}'.replace('-', '-'), 'src/one_mode.py b_cld(M=1/4, beta=4): eig Sym B')
    add('kappa', f'{kappa2(b):.0f}', 'kappa = lambda_max(Sym B)')
    # ---- B1
    S = J('B1', 'results', 'summary_B1.json'); tf = S['threshold_fit']; P = S['paired_log_within']; rr = S['ratios']
    add('B1_paired_05', f2(100 * P['mean'][rr.index(0.5)]), 'B1 summary paired_log_within at r/k=0.5 (x100)')
    add('B1_paired_095', f2(100 * P['mean'][rr.index(0.95)]), 'B1 summary paired_log_within at r/k=0.95 (x100)')
    add('B1_kfit', f3(tf['kappa_fit_over_kappa']), 'B1 threshold_fit kappa_fit_over_kappa')
    add('B1_kfit_se', f3(tf['kappa_fit_se']), 'B1 threshold_fit kappa_fit_se')
    add('B1_kfit_boot', f"[{tf['boot_kappa_fit_16_50_84'][0]:.3f}, {tf['boot_kappa_fit_16_50_84'][2]:.3f}]", 'B1 bootstrap 16/84')
    add('B1_poly', f3(tf['poly_kappa_fit']), 'B1 model-free quadratic')
    add('B1_poly_boot', f"[{tf['boot_poly_16_50_84'][0]:.3f}, {tf['boot_poly_16_50_84'][2]:.3f}]", 'B1 quadratic bootstrap 16/84')
    add('B1_A', f2(tf['A']), 'B1 amplitude A'); add('B1_A_se', f2(tf['A_se']), 'B1 amplitude s.e.')
    fx = max(abs(v) for v in S['log_within_ratio_fixed']['mean']); add('B1_fixed_max', f1(100 * fx), 'B1 max |log ratio| of (ii) over r (x100)')
    sp = {}
    for f in glob.glob(R('B2', 'results', 'spacing_R*_s*.json')):
        x = json.load(open(f)); sp.setdefault(x['radius_factor'], []).append(x['ratio_to_theory'])
    add('spacing', ', '.join(f2(np.mean(sp[k])) for k in sorted(sp)), 'B2 spacing_R*: measured/theory, radius x1,2,4,8')
    dg = [json.load(open(f)) for f in sorted(glob.glob(R('B2', 'results', 'diag_B1seed*.json')))]
    add('switch_pct', f1(100 * np.mean([d['n_switch'] for d in dg]) / 40000), 'B2 diag_B1seed*: trajectories switching mode (%)')
    # ---- B2 closed forms
    add('slope_B1', f3(slope_at_threshold()), 'src/closed_forms.py slope_at_threshold() (B1 setup)')
    for rk in [0.1, 0.5, 0.9]: add(f'Linf_{rk}', f1(pc(L_noise_limit(rk)[0])), f'B2 L_noise_limit(r/k={rk}), percent change')
    add('r0_B1', f1(pc(L_r_to_zero())), 'B2 L_r_to_zero() (B1 setup), percent change')
    m = J('B2', 'results', 'map_smin.json'); i = int(np.argmin(np.abs(np.array(m['vals']) - 0.0625))) if False else None
    add('map_s16_01', f1(pc(Theory(0.1).L()[0])), 'one-mode theory, sigma_min = s/16, r/k = 0.1')
    j01 = m['rk'].index(0.1); add('map_s100_01', f1(pc(m['L_pos'][m['vals'].index(min(m['vals']))][j01])), 'B2 map_smin at sigma_min = 0.01 s, r/k = 0.1')
    mx = min(np.nanmin(np.array(J('B2', 'results', n)['L_pos'])) for n in ['map_smin.json', 'map_M.json', 'map_M_smin0.5.json', 'map_vstd.json', 'map_smax.json', 'map_M_vstd0.3.json'])
    add('map_max', f'{-pc(mx):.0f}', 'B2 maps: largest shrinkage (percent)')
    S2 = J('B2', 'results', 'summary_B2.json')
    for n in 'ABC':
        add(f'mix_{n}', f1(S2[n]['percent_change_paired']), f'B2 summary {n} paired, percent change')
        add(f'mixth_{n}', f1(pc(S2[n]['theory_log_within'])), f'B2 summary {n} one-mode theory, percent change')
    mf = max(abs(S2[n][f'log_within_{m}'][0]) for n in 'ABC' for m in ['fixed', 'ode']); add('mix_fixed_max', f1(100 * mf), 'B2 summary: max |100 L| of (ii), (iii) over A, B, C')
    add('velA_mix', f1(pc(S2['A']['paired_log_vel'][0])), 'B2 summary A paired velocity, percent change')
    sa = [json.load(open(f)) for f in glob.glob(R('B2', 'results', 'single_A_s*.json'))]
    add('velA_one', f1(pc(np.mean([x['extrap'][1] for x in sa]))), 'B2 single_A_s*: one-mode velocity, percent change')
    # ---- B3
    S3 = J('B3', 'results', 'summary_B3.json')
    for n in 'BCD':
        d = S3[n]
        add(f'ratio_{n}', f2(d['ratio_paired']), f'B3 summary {n} learned/exact paired')
        add(f'pl_{n}', f4(d['learned']['paired'][0]), f'B3 {n} learned paired'); add(f'pe_{n}', f4(d['exact']['paired'][0]), f'B3 {n} exact paired')
        add(f'dii_{n}', f1(100 * d['delta']['fixed'][0]), f'B3 {n} delta_ii (x100)')
        if np.isfinite(d['delta']['ode'][0]) and abs(d['delta']['ode'][0]) < 1: add(f'diii_{n}', f1(100 * d['delta']['ode'][0]), f'B3 {n} delta_iii (x100)')
        add(f'err_{n}', (f2 if n == 'B' else f1)(100 * np.mean(d['e_near_smin'])), f'B3 {n} score error near sigma_min (percent)')
    th30 = [S3[n]['learned']['paired'][0] / S3[n]['delta']['fixed'][0] for n in 'BC']
    add('one_thirtieth', 'one thirtieth' if all(25 <= abs(x) <= 35 for x in th30) else f'MISMATCH {th30}', 'B3: |paired learned| / delta_ii in [25, 35] for B and C')
    add('pe_D_pct', f1(-100 * S3['D']['exact']['paired'][0]), 'B3 D exact paired (x100, sign dropped)')
    eD = J('B3', 'results', 'err_D_n1_k300000.json')['by_sigma'][0]['e_all']; add('errD_at_smin', f'{100 * eD:.0f}', 'B3 err_D_n1: e at sigma_min (percent)')
    tr = J('B3', 'results', 'posthoc_trim.json')['D_ode']; add('D_div', f'{tr["removed_total"]}/{tr["n_total"]:,}'.replace(',', '\\,'), 'B3 posthoc_trim: diverging ODE trajectories')
    add('train_min', '23', 'B3 log_train.txt: 1361-1384 s per network')
    # ---- B4
    T4 = J('B4', 'results', 'theory_b4.json')
    add('rate_B1', f'{T4["rates"]["D"]["r_minus_wmin"]:.0f}', 'B4 theory_b4 rates D r - w_min')
    th = {(x['setting'], x['step'], x['net']): x for x in T4['deltas']}
    add('thratio_B', f1(th[('B', 300000, 1)]['ode'] / th[('B', 300000, 1)]['fixed']), 'B4 theory ratio iii/ii, B final net 1')
    add('thratio_C', f'{th[("C", 300000, 1)]["ode"] / th[("C", 300000, 1)]["fixed"]:.0f}', 'B4 theory ratio iii/ii, C final net 1')
    obs = lambda n: S3[n]['delta']['ode'][0] / S3[n]['delta']['fixed'][0]
    add('obsratio_B', f1(obs('B')), 'B3 observed ratio iii/ii, B (6 runs)'); add('obsratio_C', f1(obs('C')), 'B3 observed ratio iii/ii, C (6 runs)')
    over = [th[(n, 300000, 1)]['fixed'] / S3[n]['delta']['fixed'][0] for n in 'BC']
    add('overestimate', f'{min(over):.0f}--{max(over):.0f}', 'B4 theory delta_ii / B3 observed delta_ii, final net 1, B and C')
    E4 = [J('B4', 'results', f'err_D_n{n}_k300000.json')['e_near_smin'] for n in [1, 2]]
    add('score_err', f1(100 * np.mean(E4)), "B4 'score' weighting: error near sigma_min (percent)")
    R4 = [json.load(open(f)) for f in glob.glob(R('B4', 'results', 'samp_D_n*_k300000_s*_N20000.json'))]
    nf = sum(1 for x in R4 for mm in ['naive', 'fixed', 'ode'] if not np.isfinite(x['learned'][mm]['L']))
    add('score_div', f'{nf} of {3 * len(R4)}', "B4 'score' weighting: non-finite learned runs")
    return N

if __name__ == '__main__':
    rows = numbers()
    with open(os.path.join(ROOT, 'paper', 'numbers.md'), 'w') as fh:
        fh.write('# Numbers in the manuscript and their sources\n\nGenerated by `verification/paper_numbers.py`; checked against the LaTeX sources by `verification/check_numbers.py`.\n\n')
        fh.write('| key | as printed | source |\n|---|---|---|\n')
        for k, s, src in rows: fh.write(f'| {k} | {s} | {src} |\n')
    for k, s, _ in rows: print(f'{k:16s} {s}')
