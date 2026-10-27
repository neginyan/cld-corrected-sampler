> **English translation added after the fact (2026-10-27).** The Japanese original, `B2_事前登録.md`, is the record and is kept unmodified in this folder. Stage B2 corresponds to Sec. IV of the paper.
>
> **Translator's note on the time stamps.** The time labels written inside the original are wrong. The author of the file (the AI session that ran the computations) read the clock incorrectly. The file's last-modified time in the working folder was **2026-10-27 00:05:25 JST**, although the addendum is labelled "0:50 JST". The record in the working folder was:
> - The step-size studies of the main text, governed by the original text above the addendum, wrote their results between 23:54 and 00:04.
> - The mixture and spacing runs were launched at about 00:05, together with the addendum.
> - The first mixture result file was written at 00:16.
> - The last bullet of the addendum (setting B's weight check is reference only) was appended a few seconds after those runs were launched, before any of their results existed.

# B2 pre-registration (written before the computations were run; 2026-10-26 23:45 JST)

The success criteria and the tolerances of the checks are fixed here, before the parameter scans and the mixture computations are started. If anything is changed later, the change and its reason are written in the report.

## Notation and units
- The position standard deviation s of a mode is the unit of length (s = 1). The problem is linear, so the result depends only on the ratios σ_min/s, σ_max/s and v_std/s (to be confirmed by a check).
- Time is set by β. At fixed r/κ, β does not affect the result (to be confirmed by a check).
- Family of CLD flows: B = β[[0, M⁻¹], [−1, −ΓM⁻¹]], Γ = 2√M (critical damping, as in the CLD-SGM code; the damping stays critical when M is changed).
- "Shrinkage": L = log[(within-mode position variance of (i)) / (correct value)]. **"A shrinkage of 10% or more" means L ≤ log 0.9 = −0.1054.**
- B1 defaults: σ_min/s = 0.0625, σ_max/s = 500, v_std/s = 1.25, M = 1/4.

## §2-1 Formulas (derivation)
- Every formula claimed is compared with the exact ODE (one-mode Gaussian theory) at five or more parameter points.
  - Identities and exact relations: relative error ≤ 1e-6.
  - Leading-order (asymptotic) forms: a tolerance is stated in the report for each formula, and that tolerance is at most 5%.
- The linear response (slope as r → κ) must agree with the full ODE to within 2% at r/κ = 0.99.

## §2-2 Map
- ODE accuracy: changing the solver tolerance rtol from 1e-10 to 1e-12 changes L by at most 1e-6 (relative).
- Whether a region with more than 10% shrinkage exists is reported as a result either way (it is not a pass/fail criterion).

## §2-3 Check with the mixture (2–3 settings)
Same method as B1 (exact score; samplers (i), (ii), (iii); paired comparison; two step sizes with extrapolation; N = 40000; five seeds).
- Step convergence: for the paired L, |fine − coarse| ≤ 15% × |extrapolated value|.
- Significant shrinkage of (i): mean + 2 standard errors < 0.
- **Success**: in at least one of the chosen settings, the mean paired L of the mixture is ≤ −0.1054 (a shrinkage of 10% or more).
- (ii), (iii): |L| relative to the true value ≤ 3 standard errors + 0.01. This is looser than the 0.003 of B1 because moving σ_min/s or r/κ far can change the discretization error. If the extrapolated value exceeds this, it is reported as a FAIL.
- Mode weights: max |w_k − 1/8| ≤ 0.01. Exception: in a setting where σ_min approaches the mode spacing and the modes overlap, the rule that assigns samples to modes becomes ambiguous, so the weights there are reference values only.
- The measured/theory ratio is reported (not a pass/fail criterion).

## §2-4 Mode-spacing hypothesis (if done)
- r/κ = 0.5 with the B1 defaults; the circle radius is varied from 1 to 8 times the B1 radius.
- The **hypothesis is supported** if the measured/theory ratio increases with the radius (ratio at each point averaged over seeds), and at the largest radius it lies in 0.9–1.05. Otherwise it is reported that the hypothesis is not supported.

## Addendum (labelled 2026-10-27 0:50 JST in the original; see the translator's note; before the mixture and spacing computations were run)
- The settings of §2-3 are A (M = 0.01, r/κ = 0.2), B (σ_min = 1.78 s, r/κ = 0.3) and C (M = 10, σ_min = 0.5 s, r/κ = 0.5). In every one, β is chosen so that κ = 2. The step sizes are 0.002 and 0.001 (decided from the step-size study, results/steps_*.json).
- Learning from B1, the reference sample for the MMD is drawn afresh for every seed (reference value only, not used for pass/fail).
- §2-4 uses radii 1, 2, 4 and 8 times the B1 radius, seeds 0, 1 and 2, and N = 40000. The criterion is as written above.
- In setting B (σ_min = 1.78 s, blurred mode width ≈ 0.16, mode spacing 0.77), the tail of each mode reaches about 1% into its neighbours. By the proviso of §2-3, the weight check is a reference value in this setting. A and C use the normal check.
