"""Manuscript checks: every number printed in the paper is recomputed from the stored results
(verification/paper_numbers.py) and must appear, as printed, in the LaTeX sources.  Also checks the manuscript structure.
Run:  python verification/check_numbers.py   (exit 0 iff all pass)."""
import os, sys, glob, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'verification')); os.chdir(ROOT)
from paper_numbers import numbers

res = []
def check(name, ok, info=''):
    res.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name} {info}")

tex = {f: open(f).read() for f in sorted(glob.glob('paper/sections/*.tex'))}
alltex = '\n'.join(tex.values())
norm = lambda s: s.replace('$', '').replace('\\,', '\\,').replace('{', '').replace('}', '')
flat = norm(alltex)

print('== A. numbers in the manuscript = numbers recomputed from the result files')
for key, s, src in numbers():
    cands = {s, s.lstrip('-')}
    hit = [c for c in cands if norm(c) in flat]
    check(f'{key}: "{s}" appears in the manuscript', hit, f'(source: {src})')

print('== B. structure and wording rules')
main = open('paper/main.tex').read()
for f in tex:
    stem = os.path.splitext(os.path.basename(f))[0]
    check(f'main.tex inputs {stem}', f'sections/{stem}' in main)
check('main.tex uses revtex4-2 with aps,pre,twocolumn,superscriptaddress', re.search(r'\\documentclass\[aps,pre,twocolumn,superscriptaddress[^\]]*\]\{revtex4-2\}', main))
check('author, affiliation and contact e-mail present', 'Taishi Namba' in main and 'Independent Researcher, Atsugi, Kanagawa, Japan' in main and '\\email{neginyan@gmail.com}' in main)
ab = tex['paper/sections/00_abstract.tex']; intro = tex['paper/sections/01_introduction.tex']
check('abstract states first that the standard CLD never meets the threshold', 'standard CLD' in ab.split('.')[1] + ab.split('.')[2] if ab.count('.') > 2 else False)
check('introduction states it in the first paragraph', 'never meets the threshold' in intro.split('\n\n')[1])
check('abstract states that in typical settings the effect is a few per cent, below the learned-score error', 'a few per cent, smaller than the error of a small learned score' in ab)
bad = [w for w in ['submitted', 'under review', 'Physical Review E', 'ICML', 'NeurIPS 2026', 'Machine Learning: Science'] if w.lower() in (alltex + main).lower()]
check('no journal name or submission status', not bad, f'{bad}')
bm = tex['paper/sections/08_backmatter.tex']
check('acknowledgments credit every AI assistant used (Claude, Gemini, ChatGPT) and the author takes responsibility', all(w in bm for w in ['Anthropic Claude', 'Google Gemini', 'OpenAI ChatGPT', 'takes full responsibility']))
check('data availability gives the repository (or the [URL] placeholder before publication)', '[URL]' in bm or 'cld-corrected-sampler' in bm)
for lab in ['proposition', 'derived', 'observation', 'conjecture']:
    check(f'statement type "{lab}" is used', f'\\begin{{{lab}}}' in alltex)
for fig in ['fig1_threshold', 'fig2_map', 'fig3_learned', 'fig4_forgetting']:
    check(f'figure {fig}: PDF and PNG present and used', os.path.exists(f'paper/figures/{fig}.pdf') and os.path.exists(f'paper/figures/{fig}.png') and f'figures/{fig}' in alltex)
bib = open('paper/references.bib').read(); cited = set(k for c in re.findall(r'\\cite\{([^}]*)\}', alltex) for k in c.split(','))
check('every citation has a bib entry', all(re.search(r'@\w+\{' + re.escape(k.strip()) + ',', bib) for k in cited), f'({len(cited)} keys)')
for k in ['dockhorn2022cld', 'namba2026threshold', 'namba2026learned', 'anderson1982reverse', 'song2021sde']:
    check(f'required reference cited: {k}', k in cited)

print('== C. B6 revisions (main-session comments)')
import re as _re
words = len(_re.sub(r'\\emph\{|\}|\\%', '', _re.sub(r'\$[^$]*\$', 'X', ab)).split())
check('abstract about 200 words (<= 215)', words <= 215, f'({words} words)')
check('abstract scopes the ODE statement to a single Gaussian mode in the noise-dominated regime', 'single Gaussian mode in the noise-dominated regime' in ab)
check('reference [arrow of time] has DOI 10.5281/zenodo.23258135 (v1.2), old DOI gone', '10.5281/zenodo.23258135' in bib and '23163727' not in bib and 'Preprint v1.2' in bib)
check('introduction has the motivation paragraph and says the benefits are not tested', 'Why consider such schedules?' in intro and 'do not test' in intro)
check('motivation paragraph comes after the first paragraph (scope statement stays first)', intro.find('never meets the threshold') < intro.find('Why consider such schedules?'))
dis = tex['paper/sections/07_discussion.tex']
check('conjecture restricted to samplers whose drift has a skew part', 'whose drift has a skew part' in dis and 'any learned-score' not in dis)

n = len(res); print(f'\n{sum(res)}/{n} checks passed'); sys.exit(0 if all(res) else 1)
