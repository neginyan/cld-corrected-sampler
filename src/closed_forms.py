"""Closed forms and decompositions for the one-mode theory (report sec. 2).

(a) Exact integral form.  G_- = 2 (kappa - r) sigma^2 u u^T  (u: top eigenvector of Sym B; for the CLD 2x2 block only
    one eigenvalue exceeds r when 0 < r < kappa).  Hence E(0) = -(kappa - r) K(r),
        K(r) = int_0^T 2 sigma(t)^2 Phi(t) u u^T Phi(t)^T dt  >= 0,
    Phi(t): propagator of dE/dtau = A E + E A^T from time t back to 0.  K obeys dK/dtau = A K + K A^T + 2 sigma^2 u u^T.
    So  L = log(1 - (kappa - r) K_xx(r) / C_xx(0))  exactly.
(b) Linear response at the threshold:  dL/d(r/kappa) at r = kappa equals kappa K_xx(kappa) / C_xx(0).
(c) Noise-dominated limit (sigma_min >> s, v_std): C = sigma^2 I, E = sigma^2 F with F solving the constant
    Lyapunov equation  (A + rI) F + F (A + rI)^T = 2 (kappa - r) u u^T,  A + rI = -B + rI - 2 (r - w_min) v v^T,
    L_inf = log(1 + F_xx).  Depends on r/kappa and M only.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from one_mode import Theory, b_cld, kappa2

def K_xx(th):
    """K(r) of (a) for a Theory instance, by its ODE."""
    w, V = np.linalg.eigh((th.b + th.b.T) / 2); u = V[:, -1]
    def rhs(tau, y):
        t = th.T - tau; Kk = y.reshape(2, 2); Gp, _ = th.Gparts(t)
        A = -th.b - Gp @ np.linalg.inv(th.C(t))
        return (A @ Kk + Kk @ A.T + 2 * th.sig2(t) * np.outer(u, u)).ravel()
    y = solve_ivp(rhs, [0, th.T], np.zeros(4), method='DOP853', rtol=th.rtol, atol=th.atol).y[:, -1]
    return y.reshape(2, 2)

def L_integral_form(rk, **kw):
    th = Theory(rk, **kw); Kk = K_xx(th); C0 = th.C(0)
    return float(np.log(1 - (th.k - th.r) * Kk[0, 0] / C0[0, 0]))

def slope_at_threshold(**kw):
    th = Theory(1.0, **kw); return float(th.k * K_xx(th)[0, 0] / th.C(0)[0, 0])

def L_noise_limit(rk, M=0.25, beta=4.0):
    b = b_cld(M, beta); k = kappa2(b); r = rk * k
    w, V = np.linalg.eigh((b + b.T) / 2); u, v = V[:, -1], V[:, 0]
    A = -b + r * np.eye(2) - 2 * (r - w[0]) * np.outer(v, v)
    F = solve_continuous_lyapunov(A, 2 * (k - r) * np.outer(u, u))
    return float(np.log(1 + F[0, 0])), float(np.log(1 + F[1, 1]))

def source_by_logsigma(rk, edges, **kw):
    """Where the damage is made: contribution to E_xx(0)/C_xx(0) from the source G_- restricted to
    log10(sigma/s) in each bin (E is linear in the source, so the bins add up to the total)."""
    th = Theory(rk, **kw); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        def rhs(tau, y):
            t = th.T - tau; E = y.reshape(2, 2); Gp, Gm = th.Gparts(t)
            ls = np.log10(np.sqrt(th.sig2(t)))
            A = -th.b - Gp @ np.linalg.inv(th.C(t))
            return (A @ E + E @ A.T - (Gm if lo <= ls < hi else 0 * Gm)).ravel()
        y = solve_ivp(rhs, [0, th.T], np.zeros(4), method='DOP853', rtol=1e-9, atol=1e-14, max_step=th.T / 2000).y[:, -1]
        out.append(float(y[0] / th.C(0)[0, 0]))
    return out

def L_r_to_zero(**kw):
    """(d) r -> 0 limit at fixed sigma_min: sigma frozen at sigma_min, G_- = 2 kappa sigma_min^2 u u^T,
    G_+ = 2 (-w_min) sigma_min^2 v v^T, C(t) = m(t) S0 m(t)^T + sigma_min^2 I; integrate the error equation over
    t in [0, t_end] with t_end large (the integrand decays).  Returns log ratio."""
    th = Theory(0.5, **kw); w, V = np.linalg.eigh((th.b + th.b.T) / 2); u, v = V[:, -1], V[:, 0]
    s2 = th.smin**2; Gm = 2 * th.k * s2 * np.outer(u, u); Gp = 2 * (-w[0]) * s2 * np.outer(v, v)
    t_end = 60.0 / th.k                       # many relaxation times
    def Cf(t):
        from scipy.linalg import expm
        m = expm(th.b * t); return m @ th.S0 @ m.T + s2 * np.eye(2)
    def rhs(tau, y):
        t = t_end - tau; E = y.reshape(2, 2); A = -th.b - Gp @ np.linalg.inv(Cf(t))
        return (A @ E + E @ A.T - Gm).ravel()
    E = solve_ivp(rhs, [0, t_end], np.zeros(4), method='DOP853', rtol=1e-10, atol=1e-14).y[:, -1].reshape(2, 2)
    return float(np.log((Cf(0)[0, 0] + E[0, 0]) / Cf(0)[0, 0]))
