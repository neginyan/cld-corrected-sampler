"""Figures of the manuscript, rebuilt from the stored results in results/ (no new simulations).
Run from anywhere:  python paper/figures/make_figures.py [outdir]   -> <outdir>/fig{1,2,3,4}_*.{pdf,png}
(default outdir: paper/figures).  Stage labels: B1 = results/threshold, B2 = results/map, B3 = results/learned,
B4 = results/forgetting."""
import os, sys, json, glob
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
HERE = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, 'paper', 'figures'); os.makedirs(HERE, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
from one_mode import Theory
from closed_forms import L_noise_limit
from forgetting import generator

STAGE = {'B1': 'threshold', 'B2': 'map', 'B3': 'learned', 'B4': 'forgetting'}
R = lambda st, _res, *p: os.path.join(ROOT, 'results', STAGE[st], *p)
J = lambda *p: json.load(open(R(*p)))
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#888', 'axes.linewidth': 0.6,
                     'axes.grid': True, 'grid.color': '#e8e8e8', 'grid.linewidth': 0.5, 'lines.linewidth': 1.3,
                     'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
C_I, C_II, C_III, C_TH = '#eb6834', '#2a78d6', '#1baf7a', '#555555'
RAMP = {0.1: '#0d366b', 0.3: '#1c5cab', 0.5: '#2a78d6', 0.7: '#6da7ec', 0.9: '#9ec5f4'}
W1, W2 = 3.4, 7.0
pct = lambda L: 100 * (np.exp(np.asarray(L, float)) - 1)
def save(fig, name):
    fig.savefig(os.path.join(HERE, name + '.pdf')); fig.savefig(os.path.join(HERE, name + '.png'), dpi=220); plt.close(fig)
def tag(ax, s): ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontweight='bold', fontsize=8.5, ha='right', va='bottom')

# ---------------------------------------------------------------- Fig. 1: threshold with the exact score (B1)
S1 = J('B1', 'results', 'summary_B1.json'); rr = np.array(S1['ratios']); P = S1['paired_log_within']; tf = S1['threshold_fit']
g = np.linspace(0.3, 1.1, 81)
th = np.array([Theory(x).L()[0] if x < 1 else 0.0 for x in g])          # one-mode theory (2x2 form = B1 4-D theory, checked in B2 A1)
f_th = lambda x: np.interp(x, g, th, right=0.0)
fig, ax = plt.subplots(figsize=(W1, 2.35))
ax.axvspan(0.3, 1.0, color='#f6f6f6', zorder=0)
ax.plot(g, 100 * th, '--', color=C_TH, lw=1.1, label='one-mode Gaussian theory')
xx = np.linspace(0.45, 1.1, 200); ax.plot(xx, 100 * tf['A'] * f_th(xx + 1 - tf['kappa_fit_over_kappa']), '-', color=C_I, lw=0.9, alpha=0.6,
                                          label=rf"fit: $\kappa_{{\rm fit}}/\kappa = {tf['kappa_fit_over_kappa']:.3f}\pm{tf['kappa_fit_se']:.3f}$")
ax.errorbar(rr, 100 * np.array(P['mean']), yerr=100 * np.array(P['se']), fmt='o', ms=3.5, color=C_I, mec='white', mew=0.6, capsize=1.5,
            label='measured, (i) vs (ii) paired')
ax.axvline(1, color='#999', lw=0.7); ax.axhline(0, color='#999', lw=0.6)
ax.set_xlim(0.4, 2.05); ax.set_ylim(-2.2, 0.25); ax.set_xlabel(r'$r/\kappa$'); ax.set_ylabel(r'$\log$ var$_{\rm (i)}$/var$_{\rm (ii)}$ [%]')
ax.text(0.42, 0.12, r'$r<\kappa$', fontsize=7, color='#666'); ax.text(1.5, 0.12, r'$r\geq\kappa$: (i) $\equiv$ (ii)', fontsize=7, color='#666', ha='center')
ax.legend(frameon=False, loc='lower right')
fig.tight_layout(pad=0.3); save(fig, 'fig1_threshold')

# ---------------------------------------------------------------- Fig. 2: map (B2)
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55))
d = J('B2', 'results', 'map_smin.json'); v = np.array(d['vals']); L = np.array(d['L_pos'])
a = axs[0]
for rk, col in RAMP.items():
    j = d['rk'].index(rk); a.plot(v, pct(L[:, j]), 'o-', ms=2.2, color=col, label=rf'$r/\kappa={rk}$')
    a.plot([7, 12], [pct(L_noise_limit(rk)[0])] * 2, color=col, lw=1.6, solid_capstyle='butt')
a.axhline(-10, color='#b8483c', lw=0.7, ls='--'); a.axvline(1 / 16, color='#999', lw=0.6, ls=':')
a.set_xscale('log'); a.set_xlabel(r'$\sigma_{\min}/s$'); a.set_ylabel('change of within-mode var. (i) [%]')
a.text(1 / 16 * 1.1, 0.6, 'B1', fontsize=6.5, color='#666'); a.text(13, -21.5, r'$L_\infty$', fontsize=6.5, color='#333')
a.legend(frameon=False, loc='lower left'); tag(a, '(a)'); a.set_xlim(0.008, 16)
a = axs[1]
for name, ls_, lab in [('map_M.json', '-', r'$\sigma_{\min}=s/16$'), ('map_M_smin0.5.json', '--', r'$\sigma_{\min}=s/2$')]:
    d = J('B2', 'results', name); v = np.array(d['vals']); L = np.array(d['L_pos']); keep = ~np.isclose(v, 1.0)
    for rk in [0.1, 0.5]:
        j = d['rk'].index(rk); a.plot(v[keep], pct(L[keep, j]), ls_, color={0.1: '#0d366b', 0.5: '#6da7ec'}[rk], lw=2.0, label=f'{lab}, ' + rf'$r/\kappa={rk}$')
a.axhline(-10, color='#b8483c', lw=0.7, ls='--'); a.axvline(0.25, color='#999', lw=0.6, ls=':'); a.text(0.27, 0.8, 'CLD', fontsize=6.5, color='#666'); a.set_ylim(-27, 3)
a.set_xscale('log'); a.set_xlabel(r'mass $M$ ($\Gamma=2\sqrt{M}$; $M=1$: $\kappa=0$)')
from matplotlib.lines import Line2D
a.legend([Line2D([], [], color='#0d366b', lw=2), Line2D([], [], color='#6da7ec', lw=2), Line2D([], [], color='#444', lw=2, ls='-'), Line2D([], [], color='#444', lw=2, ls='--')], [r'$r/\kappa=0.1$', r'$r/\kappa=0.5$', r'$\sigma_{\min}=s/16$', r'$\sigma_{\min}=s/2$'], frameon=False, loc='upper center', fontsize=6.2, ncol=2, columnspacing=1.0, handlelength=2.2, bbox_to_anchor=(0.5, -0.30)); tag(a, '(b)')
a = axs[2]; d = J('B2', 'results', 'decompose.json')
for key, col, lab in [('default', '#2a78d6', r'B1 setup, $r/\kappa=0.5$'), ('B', '#1baf7a', 'setting B'), ('C', '#eda100', 'setting C')]:
    x = d[key]; e = np.array(x['edges']); smin = 10**e[0] if key != 'default' else 1 / 16
    share = np.array(x['contrib']) / x['sum']; cen = 10**(0.5 * (e[:-1] + e[1:])) / smin
    keep = np.diff(e) > 1e-9; a.plot(cen[keep], 100 * np.cumsum(share[keep]), 'o-', ms=2.2, color=col, label=lab)
a.set_xscale('log'); a.set_xlabel(r'$\sigma/\sigma_{\min}$ (upper end of band)'); a.set_ylabel('cumulative share of shrinkage [%]'); a.set_xlim(0.9, 30)
a.set_ylim(-3, 105); a.legend(frameon=False, loc='lower right'); tag(a, '(c)')
fig.tight_layout(pad=0.3, w_pad=0.8); fig.subplots_adjust(bottom=0.33); save(fig, 'fig2_map')

# ---------------------------------------------------------------- Fig. 3: learned score (B3)
S3 = J('B3', 'results', 'summary_B3.json')
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw={'width_ratios': [1.25, 1, 1]})
a = axs[0]; xs = np.arange(3); w = 0.13
for i, name in enumerate('BCD'):
    for j, (m, col) in enumerate([('naive', C_I), ('fixed', C_II), ('ode', C_III)]):
        for k, kind in enumerate(['exact', 'learned']):
            mval, se = S3[name][kind][m]; x0 = i + (j - 1) * 0.29 + (k - 0.5) * w
            if not np.isfinite(mval) or abs(mval) > 1:
                a.text(x0, 1.0, 'div.', fontsize=5.5, color=col, ha='center', rotation=90); continue
            a.bar(x0, pct(mval), w * 0.95, yerr=100 * se, color=col, alpha=0.4 if kind == 'exact' else 1.0, error_kw={'lw': 0.6, 'capsize': 1})
a.axhline(0, color='#999', lw=0.6); a.set_xticks(xs, ['B', 'C', 'D']); a.set_ylabel('within-mode variance vs target [%]')
a.text(0.98, 0.04, 'light: exact score\ndark: learned score', transform=a.transAxes, fontsize=6, color='#555', ha='right')
for lab, col, y in [('(i) clipped SDE', C_I, -5.5), ('(ii) corrected SDE', C_II, -7.5), ('(iii) prob.-flow ODE', C_III, -9.5)]: a.text(1.55, y, lab, color=col, fontsize=6.5)
tag(a, '(a)')
a = axs[1]
for name, ls_ in [('B', '-'), ('C', '--')]:
    rows = S3['checkpoints'][name]; e = [r['e_near_smin'] for r in rows]
    for m, col in [('naive', C_I), ('fixed', C_II), ('ode', C_III)]:
        a.plot(e, [100 * r['delta'][m] for r in rows], ls_, marker='o', ms=2.4, color=col)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'score error near $\sigma_{\min}$'); a.set_ylabel(r'$\Delta$ = learned $-$ exact [log, %]')
a.plot([], [], 'k-', lw=1, label='setting B'); a.plot([], [], 'k--', lw=1, label='setting C'); a.legend(frameon=False, loc='upper left'); tag(a, '(b)')
a = axs[2]
for name, col in [('B', '#1c5cab'), ('C', '#eda100'), ('D', '#b8483c')]:
    e = J('B3', 'results', f'err_{name}_n1_k300000.json'); smin = e['by_sigma'][0]['sigma']
    a.plot([o['sigma'] / smin for o in e['by_sigma']], [o['e_all'] for o in e['by_sigma']], color=col, label=f'setting {name}')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel(r'$\sigma/\sigma_{\min}$'); a.set_ylabel(r'relative score error $e(\sigma)$')
a.axvspan(1, 2, color='#eee', zorder=0); a.legend(frameon=False, loc='upper right'); tag(a, '(c)')
fig.tight_layout(pad=0.3, w_pad=0.8); save(fig, 'fig3_learned')

# ---------------------------------------------------------------- Fig. 4: forgetting (B4)
fig, axs = plt.subplots(2, 1, figsize=(W1, 3.9))
a = axs[0]; thB = Theory(0.3); F0 = np.array([[1.0, 0.0], [0.0, 0.0]]); taus = np.linspace(0, 0.5, 300)
for mode, col, lab in [('fixed', C_II, '(ii) corrected SDE'), ('ode', C_III, '(iii) probability-flow ODE')]:
    Mg = generator(thB, mode, 1.0); a.plot(taus, [np.linalg.norm(expm(Mg * t) @ F0 @ expm(Mg * t).T) for t in taus], color=col, label=lab)
a.set_yscale('log'); a.set_xlabel(r'reverse time after a unit error ($\beta=4$)'); a.set_ylabel(r'$\|F\|$'); a.legend(frameon=False, loc='lower left'); tag(a, '(a)')
a = axs[1]; T4 = J('B4', 'results', 'theory_b4.json')['deltas']
for name, mk, col in [('B', 'o', '#1c5cab'), ('C', 's', '#eda100')]:
    tr, ob = [], []
    for k in [10000, 40000, 100000, 300000]:
        dd = [x for x in T4 if x['setting'] == name and x['step'] == k and x['net'] == 1][0]; tr.append(dd['ode'] / dd['fixed'])
        o = J('B3', 'results', f'samp_{name}_n1_k{k}_s0_N10000.json')['delta']; ob.append(o['ode'] / o['fixed'])
    a.scatter(tr, ob, marker=mk, color=col, s=16, zorder=3, label=f'setting {name} (checkpoints)')
lim = [0.8, 30]; a.plot(lim, lim, '--', color='#999', lw=0.7); a.fill_between(lim, [l / 3 for l in lim], [l * 3 for l in lim], color='#f1f1f1', zorder=0)
a.set_xlim(lim); a.set_ylim(lim); a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel(r'predicted $\Delta_{\rm iii}/\Delta_{\rm ii}$'); a.set_ylabel(r'observed $\Delta_{\rm iii}/\Delta_{\rm ii}$'); a.legend(frameon=False, loc='upper left'); tag(a, '(b)')
fig.tight_layout(pad=0.3, h_pad=0.8); save(fig, 'fig4_forgetting')
print('figures written to', HERE)
