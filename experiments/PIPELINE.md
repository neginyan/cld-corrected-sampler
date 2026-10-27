# Reproducing every result from scratch

`verification/run_all.py` checks everything from the stored results and weights. This file lists the commands that regenerate those stored files. Run every command from the repository root. Each script writes into `results/<stage>/` (and the trainer into `weights/`), whatever the working directory, so **re-running overwrites the stored files**. Work on a copy if you want to compare.

The stage labels B1–B4 appear in some file names (`summary_B1.json`, `diag_B1seed0.json`, …). They stand for:

| stage | paper | experiments | results |
|---|---|---|---|
| B1 | Sec. III (exact score, threshold) | `experiments/threshold` | `results/threshold` |
| B2 | Sec. IV (map of the damage, mixtures) | `experiments/map` | `results/map` |
| B3 | Sec. V (learned score) | `experiments/learned` | `results/learned`, `weights/learned_eps` |
| B4 | Sec. VI (forgetting); Obs. 9 (score weighting) | `experiments/forgetting`, `experiments/learned` with `CLD_WEIGHTING=score` | `results/forgetting`, `results/score_weighting`, `weights/learned_score` |

Times are for one core of a 2-core CPU. `xargs -P 2` runs two jobs at a time.

## Sec. III — threshold with the exact score (B1)
```bash
python experiments/threshold/steps.py 0.5 0; python experiments/threshold/steps.py 0.9 0      # step-size study (~10 min)
cat experiments/threshold/jobs.txt | xargs -P 2 -L 1 python experiments/threshold/run_sweep.py   # 55 jobs, ~2.5 h
python experiments/threshold/analyze.py                                                       # summary_B1.json, threshold fit, figures
python experiments/threshold/mmd_null.py; python experiments/threshold/mmd_refcheck.py         # MMD reference values
for r in 0.5 0.8; do for s in 0 1 2; do python experiments/threshold/single_mode.py $r $s; done; done   # one-mode control
```

## Sec. IV — map of the damage (B2)
```bash
python experiments/map/map_one_mode.py; python experiments/map/plot_maps.py                   # one-mode maps (~40 min)
python experiments/map/decompose.py                                                           # where the damage is made
for s in A B C; do python experiments/map/steps.py $s; done                                   # step-size study
cat experiments/map/jobs.txt | xargs -P 2 -L 1 python experiments/map/run_mixture.py           # mixtures A, B, C x 5 seeds (~2.5 h)
cat experiments/map/jobs_spacing.txt | xargs -P 2 -L 1 python experiments/map/spacing.py       # mode-spacing test (~1.5 h)
python experiments/map/analyze.py; python experiments/map/plot_spacing.py
python experiments/map/single_mode_A.py 0; python experiments/map/single_mode_A.py 1           # velocity control for setting A
python experiments/map/diag_mode_switch.py 0; python experiments/map/diag_mode_switch.py 3     # mode switching between (i) and (ii)
```

## Sec. V — learned score (B3)
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 B3_CKPT=10000                                 # single thread: bit-reproducible training
for s in B C D; do for n in 1 2; do python experiments/learned/train.py $s 300000 $n; done; done   # 6 networks, ~23 min each
for s in B C D; do for n in 1 2; do python experiments/learned/score_error.py $s $n 300000; done; done
for s in B C; do for k in 10000 40000 100000; do python experiments/learned/score_error.py $s 1 $k; done; done
cat experiments/learned/jobs_samp.txt experiments/learned/jobs_ckpt.txt | xargs -P 2 -L 1 python experiments/learned/sample.py   # ~5 h
python experiments/learned/analyze.py; python experiments/learned/posthoc_trim.py
```
The trainer writes a checkpoint every `B3_CKPT` steps into `weights/learned_eps/`. Only the checkpoints used in the paper are stored in this repository.

## Obs. 9 — 'score' weighting (B4)
```bash
export CLD_WEIGHTING=score OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 B3_CKPT=10000
for n in 1 2; do python experiments/learned/train.py D 300000 $n; done                        # -> weights/learned_score
for n in 1 2; do python experiments/learned/score_error.py D $n 300000; done                  # -> results/score_weighting
for n in 1 2; do for s in 0 1 2; do python experiments/learned/sample.py D $n 300000 $s 20000; done; done
```

## Sec. VI — forgetting (B4)
```bash
python experiments/forgetting/theory_runs.py; python experiments/forgetting/plot.py
```

## Paper figures
```bash
python paper/figures/make_figures.py          # rebuilds paper/figures/fig{1,2,3,4}_*.{pdf,png} from results/
python verification/paper_numbers.py          # rewrites paper/numbers.md (every number in the paper and its source)
```
