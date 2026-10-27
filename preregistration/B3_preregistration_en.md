> **English translation added after the fact (2026-10-27).** The Japanese original, `B3_事前登録.md`, is the record and is kept unmodified in this folder. Stage B3 corresponds to Sec. V of the paper.
>
> **Translator's note.**
> - **Time stamp.** The time label inside the original ("07:20 JST") is wrong; the author of the file read the clock incorrectly. The file's last-modified time in the working folder was **2026-10-27 06:46:11 JST**. The first training checkpoint was written at 06:47:01, and the first score-error file at 06:48.
> - **Deviations, as reported at the time.**
>   - Checkpoints were saved every 10k steps, a superset of the planned "10k, 20k, 40k, …"; the checkpoints used for S3 are the pre-registered ones.
>   - The bit-for-bit retraining check (`verification/check_learned.py`) retrains 10 000 steps instead of the 2000 written here, which is stricter.
>   - In the verification-tolerance section, `verification/run_all_B3.py` is now `verification/check_learned.py`.

# B3 pre-registration (written before training and sampling were run; labelled 2026-10-27 07:20 JST in the original, see the note)

## Settings
- D: the B1 defaults (M = 1/4, σ_min = s/16, r/κ = 0.5; shrinkage with the exact score about −1.5%), for comparison.
- B: setting B of B2 (M = 1/4, σ_min = 1.78 s, r/κ = 0.3; −11.1% in the mixture), the main case.
- C: setting C of B2 (M = 10, σ_min = s/2, r/κ = 0.5; −18.6% in the mixture), the accompanying example.
- Data: a fixed training set of 200 000 samples of the 8-mode mixture (seed 0).

## Training (the recipe of paper D)
- Denoising score matching in the ε-prediction form, weight 1 (paper D's 'eps'). MLP 13 → 128 → 128 → 128 → 4, SiLU, Adam, learning rate 2e-3 (1000 warm-up steps, cosine decay to 1e-5), EMA 0.9995 (the EMA weights are used for sampling), batch 1024.
- Input: (z − m(σ))/√(diag(MΣ_dataMᵀ) + σ²) (a preconditioning that uses only the data moments), plus Fourier features of the normalized u = log σ.
- Half of each batch draws log σ uniformly on [σ_min, σ_max], the other half uniformly on [σ_min, 10σ_min]. This corresponds to paper D's "put half of the batch where the quantity is measured", and follows B2's finding that the shrinkage is made near σ_min.
- 300 000 steps per network. Network seeds 1 and 2. Checkpoints at 10k, 20k, 40k, …, 300k (every 20k, plus 10k).
- **Once fixed, the hyperparameters are not changed after seeing results.** Only if training clearly fails (for example, the loss diverges) is it redone, with the reason written down.

## How the score error is measured
- At each σ (40 points, uniform in log σ), 20 000 fresh samples are drawn from the forward marginal at that σ. The relative error e(σ) = E|s_θ − s|² / E|s|² is measured separately for the position and velocity components (s is the exact mixture score).
- Representative value of "the error near σ_min": e averaged over σ ∈ [σ_min, 2σ_min] (following §3.2 of B2).

## Sampling
- The three samplers (same equations as B1) are run with the final EMA weights. N = 20 000; step h = T/⌈T/0.002⌉; sampling seeds 0, 1, 2 (prior from rng(100 + seed), Brownian increments from rng(200 + seed)).
- **The exact score is run on the same prior samples and the same Brownian increments.** The discretization error enters both in the same way, so the difference Δ = learned − exact is due to the score error alone.
- The target is the data blurred by σ_min (as in B2). L = log(variance / correct value) is measured on the within-mode position variance.

## Success criteria (judged on B and C; D is reported only)
- **S1 ((i) also breaks with the learned score)**: for the paired L ((i)/(ii)) with the learned score, mean + 2 standard errors < 0, and its ratio to the paired L with the exact score lies in 0.75–1.25. The standard error is computed from the 6 values of 2 networks × 3 sampling seeds.
- **S2 ((ii) also fixes it with the learned score)**: for (ii) with the learned score, |Δ_ii| = |L_ii(learned) − L_ii(exact)| ≤ 0.03 (3%), and |Δ_ii| < |paired L with the learned score| / 3 (the remainder is less than 1/3 of the shrinkage to be corrected).
- The same quantities are reported for (iii) (not pass/fail).
- **S3 (relation between the score error and the deviations)**: for B and C, the checkpoints 10k, 40k, 100k and 300k of network seed 1 are run with sampling seed 0 and N = 10 000 (learned and exact paired). The relation of Δ_i, Δ_ii and Δ_iii to the error near σ_min is shown. It is not pass/fail; the relation (monotone or not, proportional or not) is reported.
- Prediction (expectation before checking): with the learned score, a score error of the same sign enters (i) and (ii), so the paired L should be close to its value with the exact score. The remainder of (ii) should be smaller when the score error is smaller.

## Tolerances of the checks (verification/run_all_B3.py)
- Interpolation of the preconditioning moments: relative 1e-4.
- The exact-score path reproduces the B2 result: for the same setting and step, the paired L differs from the value of B2's step-size study by at most 0.005.
- Training reproducibility: retraining 2000 steps from the same seed reproduces the stored weights bit for bit (single thread).
