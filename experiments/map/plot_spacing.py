"""Figure for sec. 2-4.  -> figures/B2_spacing.png"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(os.path.join(ROOT, 'results', 'map'))   # relative paths below are inside results/<stage>/
import json, glob, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6'})
S = {}
for f in glob.glob('spacing_R*_s*.json'):
    x = json.load(open(f)); S.setdefault(x['radius_factor'], []).append(x['ratio_to_theory'])
R = sorted(S); m = [np.mean(S[r]) for r in R]; se = [np.std(S[r], ddof=1) / np.sqrt(len(S[r])) for r in R]
fig, ax = plt.subplots(figsize=(4.8, 3.4))
for r in R: ax.scatter([r] * len(S[r]), S[r], s=14, color='#86b6ef', zorder=2)
ax.errorbar(R, m, yerr=se, fmt='o-', color='#1c5cab', ms=5, capsize=2, zorder=3, label='mean of 3 seeds ± s.e.')
ax.axhspan(0.9, 1.05, color='#eeeeee', zorder=0, label='pre-registered target band'); ax.axhline(1, color='#999', lw=0.8)
ax.set_xscale('log', base=2); ax.set_xticks(R, [f'×{r:g}' for r in R]); ax.set_xlabel('circle radius (B1 = ×1; mode spacing 0.77 → 6.1)')
ax.set_ylabel('measured / one-mode theory'); ax.set_title('B1 setting, r/κ = 0.5: mixture shrinkage vs spacing', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc='lower right'); fig.tight_layout(); fig.savefig('figures/B2_spacing.png', dpi=160)
