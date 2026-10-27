> **English translation added after the fact (2026-10-27).** The Japanese original, `B4_事前登録.md`, is the record and is kept unmodified in this folder. Stage B4 corresponds to Sec. VI and Observation 9 of the paper.
>
> **Translator's note.**
> - **Time stamp.** The time label inside the original ("14:00 JST") is wrong; the author of the file read the clock incorrectly. The file's last-modified time in the working folder was **2026-10-27 13:48:41 JST**. The first training checkpoint of the 'score'-weighted networks was written at 13:51:26, and the theory results (`results/forgetting/theory_b4.json`) at 14:19.
> - **Paths.** In the verification-tolerance section, `verification/run_all_B4.py` is now `verification/check_forgetting.py`.

# B4 pre-registration (written before the computations were run; labelled 2026-10-27 14:00 JST in the original, see the note)

## §2-2 D with the 'score' weighting (numerical)
- Setting D only (the B1 defaults: M = 1/4, σ_min = s/16, r/κ = 0.5).
- Training is exactly as in B3 (data, network, preconditioning, batch sampling, 300 000 steps, network seeds 1 and 2). The only difference is the loss weight:
  **w(σ) = (σ_min/σ)²** (paper D's 'score' weight (σ₀/σ)², with σ₀ = σ_min; applied to the ε loss, it corresponds to plain score matching).
- Expected problem (expectation before running): in this schedule σ ranges over a factor of 8000 (about 4 in paper D). The weight therefore drops to about 1e-8 at large σ, and the score at large σ will probably remain almost unlearned. The large-σ error is recorded too. The setup is not changed.
- What is measured: as in B3 (relative error e(σ) by σ, error near σ_min, and 6 sampling runs: 2 networks × seeds 0–2, N = 20 000, step ≈ 0.002, paired with the exact score).
- Comparison (reported, not pass/fail; but the following two conditions for "improved" are fixed in advance):
  - (a) The error averaged over σ_min–2σ_min is at most half of that with the 'eps' weight (11.8%).
  - (b) |Δ_ii| is at most half of that with the 'eps' weight (+0.039).
  - If both (a) and (b) hold, report "the 'score' weight improves D"; if only one holds, "partly improves"; if neither, "does not improve".
- Whether the learned/exact ratio of the paired L ((i)/(ii)) lies in 0.75–1.25, as in B3, is reported.
- For divergence of the ODE, the number of diverging trajectories is reported.

## §2-1 How errors are forgotten (theory, no sampling)
- For one Gaussian mode, the learned score is modelled as ŝ = (1 − η(σ))s (an attraction that is slightly too weak), with η(σ) = √e_pos(σ) taken from the shape of the position error measured in B3.
- The output covariances of (ii) and (iii) are solved exactly by the moment equations, giving Δ_ii, Δ_iii and the ratio Δ_iii/Δ_ii.
- Compared with the B3 observations:
  - The ratio Δ_iii/Δ_ii (B3 final: 6.5 in B, 1.1 in C; checkpoints: 6–14 in B, 2–19 in C).
  - The condition for "the hypothesis follows from the equations": the theoretical ratio is well above 1 (at least 2) in both B and C. Whether its size is of the same order as B3 (within 1/3–3 times) is reported (not pass/fail).
- Check by formula: in the noise-dominated regime (C ≈ σ²I), the eigenvalues of the matrix that drives the relative error F = E/σ² are computed for (i), (ii) and (iii), and the "forgetting rates" are compared.

## Tolerances of the checks (verification/run_all_B4.py)
- Moment equations: with η = 0 they reproduce the B2 theory to a relative 1e-6.
- Generator in the noise-dominated regime: agrees with direct numerical differentiation to a relative 1e-6.
- Training reproducibility: with the 'score' weight too, the weights at 10 000 steps agree bit for bit.
