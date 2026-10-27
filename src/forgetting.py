"""Sec. 2-1: how samplers carry or forget a score error (one Gaussian mode, no sampling).

Score-error model: s_hat = (1 - eta(sigma)) s  (attraction slightly too weak), eta = sqrt(e_pos(sigma)) with e_pos the
relative position score error measured in B3.  For a sampler with score coefficient K and noise D (forward-time drift
Bz - K s_hat, reverse integration), the output covariance obeys in reverse time tau
    dSigma/dtau = A Sigma + Sigma A^T + D,   A = -B - K (1 - eta) P,   P = C^-1.
  (i)   K = G_+,            D = G_+
  (ii)  K = (G + G_+)/2,    D = G_+
  (iii) K = G/2,            D = 0
Delta_x = log(Sigma_xx(eta) / Sigma_xx(eta = 0))  (learned minus exact, as in B3).

Noise-dominated generator (C = sigma^2 I): the relative error F = E / sigma^2 obeys
    dF/dtau = (A + rI) F + F (A + rI)^T + ...,   A + rI = -Skew(B) - D / (2 sigma^2) - (K - (G + D)/2) / sigma^2.
For (ii) and (iii) K = (G + D)/2, so A + rI = -Skew(B) - D/(2 sigma^2): the injected noise D alone sets the forgetting rate;
for the ODE (D = 0) it is a pure rotation (nothing is forgotten)."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from one_mode import Theory

def run(th, mode, eta_fn, rtol=1e-10):
    """Output covariance Sigma = C(0) + E(0), with the deviation E from the exact marginal covariance integrated as in B2:
    dE/dtau = A E + E A^T + R,  R = D + G - (1 - eta)(K + K^T)   (R = 0 for (ii)/(iii) with eta = 0; R = -G_- for (i))."""
    def rhs(tau, y):
        t = th.T - tau; E = y.reshape(2, 2); P = np.linalg.inv(th.C(t))
        Gp, Gm = th.Gparts(t); G = Gp - Gm; eta = eta_fn(np.sqrt(th.sig2(t)))
        K = {'naive': Gp, 'fixed': 0.5 * (G + Gp), 'ode': 0.5 * G}[mode]; D = np.zeros((2, 2)) if mode == 'ode' else Gp
        A = -th.b - (1 - eta) * K @ P
        return (A @ E + E @ A.T + D + G - (1 - eta) * (K + K.T)).ravel()
    E = solve_ivp(rhs, [0, th.T], np.zeros(4), method='DOP853', rtol=rtol, atol=1e-14).y[:, -1].reshape(2, 2)
    return th.C(0) + (E + E.T) / 2

def deltas(th, eta_fn):
    """Delta = log(Sigma_xx(eta) / Sigma_xx(eta = 0)).  With eta = 0, (ii) and (iii) are exact (Sigma = C(0), proved:
    R = 0 and E(0) = 0), and (i) is the B2 theory; so only the eta runs are integrated."""
    out = {}; C0 = th.C(0)[0, 0]; Li = th.L()[0]
    for md in ['naive', 'fixed', 'ode']:
        S1 = run(th, md, eta_fn)[0, 0]; ref = C0 * (np.exp(Li) if md == 'naive' else 1.0)
        out[md] = float(np.log(S1 / ref)); out[md + '_exactL'] = float(np.log(ref / C0))
    return out

def generator(th, mode, sig):
    """A + rI for the relative error, noise-dominated limit C = sigma^2 I (exact algebra)."""
    s2 = sig**2; d = th.r - th.w
    Gp = 2 * s2 * th.V @ np.diag(np.clip(d, 0, None)) @ th.V.T; G = 2 * s2 * th.V @ np.diag(d) @ th.V.T
    K = {'naive': Gp, 'fixed': 0.5 * (G + Gp), 'ode': 0.5 * G}[mode]
    return -th.b - K / s2 + th.r * np.eye(2)

def forgetting_rates(th, mode):
    """Eigenvalues of the Lyapunov operator X -> M X + X M^T (M = generator): decay rates of F (per unit time)."""
    M = generator(th, mode, 1.0); I = np.eye(2)
    L = np.kron(M, I) + np.kron(I, M)
    return np.sort(np.linalg.eigvals(L).real)
