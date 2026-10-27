"""One-mode Gaussian theory for the clipped sampler (paper Prop. 6), per position coordinate (2x2: x, v).

Units: mode position std s = 1.  Parameters: rk = r/kappa, smin = sigma_min/s, smax = sigma_max/s,
vstd = initial velocity std / s, M (CLD mass; Gamma = 2 sqrt(M), critical as in CLD-SGM), beta.

Error equation (derived in the report, sec. 2): with C(t) the true marginal covariance, P = C^-1,
G_- = G_+ - G >= 0 and A = -B - G_+ P, the clipped output covariance is Sigma = C + E where, in reverse
time tau = T - t,
    dE/dtau = A E + E A^T - G_-,      E(tau = 0) = 0          (exact, linear in E)
Shrinkage L = log[(C_xx(0) + E_xx(0)) / C_xx(0)].
"""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def b_cld(M, beta=4.0):
    G = 2 * np.sqrt(M)
    return beta * np.array([[0.0, 1.0 / M], [-1.0, -G / M]])

def kappa2(b):
    return np.linalg.eigvalsh((b + b.T) / 2).max()

class Theory:
    def __init__(self, rk, smin=0.0625, smax=500.0, vstd=1.25, M=0.25, beta=4.0, rtol=1e-10, atol=1e-14):
        self.b = b_cld(M, beta); self.k = kappa2(self.b); self.r = rk * self.k
        self.smin, self.smax, self.S0 = smin, smax, np.diag([1.0, vstd**2])
        self.T = np.log(smax / smin) / self.r
        w, V = np.linalg.eigh((self.b + self.b.T) / 2); self.w, self.V = w, V
        self.rtol, self.atol = rtol, atol
    def sig2(self, t): return (self.smin * np.exp(self.r * t))**2
    def C(self, t):
        m = expm(self.b * t); return m @ self.S0 @ m.T + self.sig2(t) * np.eye(2)
    def Gparts(self, t):
        d = self.r - self.w; s2 = 2 * self.sig2(t)
        Gp = s2 * self.V @ np.diag(np.clip(d, 0, None)) @ self.V.T
        Gm = s2 * self.V @ np.diag(np.clip(-d, 0, None)) @ self.V.T       # G_- = G_+ - G >= 0
        return Gp, Gm
    def E0(self):
        def rhs(tau, y):
            t = self.T - tau; E = y.reshape(2, 2); Gp, Gm = self.Gparts(t)
            A = -self.b - Gp @ np.linalg.inv(self.C(t))
            return (A @ E + E @ A.T - Gm).ravel()
        sol = solve_ivp(rhs, [0, self.T], np.zeros(4), method='DOP853', rtol=self.rtol, atol=self.atol)
        E = sol.y[:, -1].reshape(2, 2); return (E + E.T) / 2
    def L(self):
        E = self.E0(); C0 = self.C(0)
        return float(np.log((C0[0, 0] + E[0, 0]) / C0[0, 0])), float(np.log((C0[1, 1] + E[1, 1]) / C0[1, 1]))
