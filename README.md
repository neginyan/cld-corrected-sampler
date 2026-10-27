# cld-corrected-sampler

**When the blur is slower than the flow: clipped and corrected samplers on the critically damped Langevin flow**

Code, stored results, trained networks and verification scripts for the paper by Taishi Namba (2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neginyan/cld-corrected-sampler/blob/main/demo/cld_corrected_sampler_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The result in brief

Critically damped Langevin diffusion (CLD) generates data in position–velocity phase space. If its flow is combined with an isotropic noise schedule specified by its marginals, σ(t) = σ_min e^{rt}, the standard reverse SDE exists only when the blur grows at least as fast as the flow stretches, r ≥ κ.

- **Below κ the usual repair (clipping the noise) is biased.** It shrinks each mode of the generated distribution.
- **A one-line change fixes it.** The corrected sampler replaces `G_plus` by `0.5*(G + G_plus)` in the drift. It is exact with the exact score and needs no retraining with a learned score.
- **The size of the effect is mapped.** The damage is made while the blur is close to σ_min. In typical settings it is a few per cent; it exceeds 10% only when σ_min is comparable to the data scale or the CLD mass is far from its default.
- **The probability-flow ODE does not damp score errors**, while the corrected SDE does (one Gaussian mode, noise-dominated regime).

The standard CLD, whose noise is given by its SDE, never meets this threshold. All data are four-dimensional Gaussian mixtures.

Every statement in the paper is labelled as proved (*Proposition*), derived and checked against direct numerical solution (*Derived result*), measured (*Observation*) or untested (*Conjecture*).

```diff
 def drift(z, s, G, G_plus):
-    return z @ B.T - s @ G_plus.T                  # (i) clipped
+    return z @ B.T - s @ (0.5 * (G + G_plus)).T    # (ii) corrected
```

## Demo

The [Colab notebook](demo/cld_corrected_sampler_demo.ipynb) runs on a CPU in a few minutes. It shows:
- the one-line difference between the two samplers;
- the distance from the mode centres for both samplers as r/κ changes, against the one-mode theory;
- optionally, a trained network: the probability-flow ODE against the corrected SDE.

## Verify

```bash
pip install -r requirements.txt
python verification/run_all.py              # about 6 minutes
python verification/run_all.py --recompute  # also re-runs parts of each section on a copy (about 13 minutes)
```

`run_all.py` checks:
- the SHA-256 hashes in `MANIFEST.sha256`;
- the checks of every section;
- 67 numbers of the paper against the stored results.

With `--recompute`, 286 checks run: 283 pass and 3 are known failures. The known failures belong to Sec. III: the shrinkage at r/κ = 0.95 is not significant with five seeds (as the paper states), plus two MMD diagnostics. They are reported, not hidden. The script exits with 0 when there is no unexpected failure.

No GPU is used. The score networks are a small NumPy MLP.

## Reproduce from scratch

`experiments/PIPELINE.md` lists the commands that regenerate every stored result. The full pipeline takes several hours on two CPU cores; training is bit-reproducible on a single thread.

## Repository layout

| folder | content |
|---|---|
| `src/` | shared code: CLD flow and samplers, one-mode theory, closed forms, learned-score setup, MLP |
| `experiments/` | scripts by paper section: `threshold` (Sec. III), `map` (Sec. IV), `learned` (Sec. V, Obs. 9), `forgetting` (Sec. VI) |
| `results/` | stored results of every run |
| `weights/` | trained score networks (final and intermediate checkpoints) |
| `verification/` | `run_all.py` and the checks it calls |
| `paper/` | LaTeX source and figures of the paper (`figures/make_figures.py` redraws them) |
| `preregistration/` | success criteria written before the runs (Japanese originals, unmodified, with English translations) |
| `demo/` | Colab notebook |

## Related work by the author

- T. Namba, *A stretching-rate threshold for phase-space diffusion models with isotropic marginal noise schedules*, Zenodo (2026), [doi:10.5281/zenodo.23181225](https://doi.org/10.5281/zenodo.23181225). The threshold and the exactness of the corrected sampler are proved there.
- T. Namba, *Sharp criterion for the coarse-grained arrow of time in Hamiltonian dynamics*, v1.2, Zenodo (2026), [doi:10.5281/zenodo.23258135](https://doi.org/10.5281/zenodo.23258135).
- T. Namba, *Can a learned score see the margin of the arrow of time?*, Zenodo (2026), [doi:10.5281/zenodo.23408213](https://doi.org/10.5281/zenodo.23408213).

## Use of AI assistants

The study was carried out with AI assistants: Anthropic Claude (planning, computations, code, verification and drafting), Google Gemini and OpenAI ChatGPT (review of the manuscript). Their roles are described in the acknowledgments of the paper. The author directed the research and takes full responsibility for it.

## Citation

Please cite the paper and this repository; see `CITATION.cff`.

## License

MIT © 2026 Taishi Namba
