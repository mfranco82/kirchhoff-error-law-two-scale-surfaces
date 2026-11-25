"""
Backscattered (Littrow, r = -1) order of a sinusoidal grating with a sub-wavelength small scale: error of KA versus the height of the
small scale and versus eps.

Geometry modelled on the sinusoidal reference gratings measured at NIST (period 6.67 um, amplitude ~90 nm, HeNe: L ~ 10.5 lambda,
aK ~ 0.085), here as a lossless dielectric:
  z = a cos(Kx) + xi psi(x),  psi = cos(Lambda K x)/k  (so xi = k b),  L = 10.25 lambda (10.5 lambda would put an order at grazing),
  a = 0.0135 L (aK = 0.085),  sin(theta_i) = lambda/(2L) (theta_i = 2.8 deg): the order r = -1 is the backscattered one,
  Lambda = 15 (lambda_s = 0.68 lambda).
Sweeps (TE and TM):
  "xi"  : xi = kb in [0.02, 1.2] (40 values, log), eps = 2.25
  "eps" : sqrt(eps) = j lambda/L, j = 11 ... 41 (eps = 1.15 ... 16), half-way between the transmitted Wood anomalies, kb = 0.25
Models: C-method (reference, M = r_max), KA (full profile), SSA-1, smooth profile alone (exact), series n <= 1 and n <= 3
(KA and exact zeroth orders); derived in quantities(): Debye-Waller (scalar phase-screen) attenuation of the exact smooth solution,
e_DW = e_smooth J0(2 kb cos theta_i)^2 for the backscattered order (the deterministic counterpart of exp(-(4 pi sigma cos theta_i/lambda)^2),
sigma = b/sqrt(2), used to infer roughness from the loss of a specular or diffracted order).  Stored: reflected efficiencies e+_r, |r| <= 12, and the complex amplitude B+_{-1} of every model.
usage: python backscatter.py compute | summary      (compute: ~1 min with Pool(6); stores ../data/backscatter.npz)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
from scattering import *
from cmethod import c_method, spectral_derivative
from ssa import ssa1_propagating

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(HERE, "data", "backscatter.npz")
LR, A_L, LAM, R_BS = 10.25, 0.0135, 15, -1
WL = 1.0 / LR
SIN = WL / 2
POLS = ("TE", "TM")
XIS = np.geomspace(0.02, 1.2, 40)
SQEPS = np.arange(11, 42) / LR
MODELS = ("ref", "KA", "SSA1", "smooth", "KA0_n1", "KA0_n3", "ex0_n1", "ex0_n3")
RS = np.arange(-12, 13)


def solve(pol, eps, kb):
    th = np.arcsin(SIN)
    p0 = Problem(1.0, WL, th, eps, pol, 30, N=4096)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * LAM + 12, 57)
    p = Problem(1.0, WL, th, eps, pol, rmax, N=4096)
    f, f1 = cos_surface(p, A_L); z, z1 = cos_surface(p, 1.0 / p.k, LAM)
    F, F1 = f + kb * z, f1 + kb * z1
    i = p.rmax + R_BS
    assert abs(p.alpha_r[i] + p.alpha) < 1e-9 * p.k
    eff = np.full((len(MODELS), len(RS)), np.nan); amp = np.full(len(MODELS), np.nan, complex)
    with np.errstate(all="ignore"):
        op = SmoothOperator(p, f, f1); Bex = rayleigh_exact(p, f, f1)
        dpk, dmk = mvb_coefficients(p, f, f1, z, z1, ka_amplitudes_propagating(p, f, f1), 3, smooth=op)
        dpx, dmx = mvb_coefficients(p, f, f1, z, z1, Bex, 3, smooth=op)
        Bref = [np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)]
        Bs = [Bref, ka_amplitudes(p, F, F1), ssa1_propagating(p, F), Bex,
              mvb_sum(dpk, dmk, kb, 1), mvb_sum(dpk, dmk, kb, 3), mvb_sum(dpx, dmx, kb, 1), mvb_sum(dpx, dmx, kb, 3)]
        for j, B in enumerate(Bs):
            eff[j] = efficiencies(p, *B)[0][p.rmax + RS]; amp[j] = B[0][i]
    C = np.abs(spectral_derivative(p, F, 2)).max() / (2 * p.k * np.cos(th) ** 3)
    return eff, amp, C


def task(args):
    sweep, ip, ix = args
    if sweep == "xi":
        return args, solve(POLS[ip], 2.25, XIS[ix])
    return args, solve(POLS[ip], SQEPS[ix] ** 2, 0.25)


def compute():
    tasks = [("xi", ip, ix) for ip in range(2) for ix in range(len(XIS))] + [("eps", ip, ix) for ip in range(2) for ix in range(len(SQEPS))]
    n = {"xi": len(XIS), "eps": len(SQEPS)}
    E = {s: np.full((2, n[s], len(MODELS), len(RS)), np.nan) for s in n}
    A = {s: np.full((2, n[s], len(MODELS)), np.nan, complex) for s in n}
    Cv = {s: np.full((2, n[s]), np.nan) for s in n}
    t0 = time.time()
    with Pool(6) as pool:
        for (s, ip, ix), (eff, amp, C) in pool.imap_unordered(task, tasks):
            E[s][ip, ix], A[s][ip, ix], Cv[s][ip, ix] = eff, amp, C
    np.savez(FILE, eff_xi=E["xi"], eff_eps=E["eps"], amp_xi=A["xi"], amp_eps=A["eps"], C_xi=Cv["xi"], C_eps=Cv["eps"],
             xis=XIS, sqeps=SQEPS, models=MODELS, rs=RS, pols=POLS, geometry=[LR, A_L, LAM, R_BS])
    print(f"{len(tasks)} cases, {time.time() - t0:.0f} s -> {FILE}")


def debye_waller(kb):
    from scipy.special import j0
    return j0(2 * kb * np.sqrt(1 - SIN ** 2)) ** 2


def kb_from_attenuation(ratio):
    """Height kb inferred from the attenuation e_bs/e_smooth with the Debye-Waller factor (small-argument branch of J0)."""
    from scipy.optimize import brentq
    return np.array([brentq(lambda k: debye_waller(k) - q, 0, 1.2) if 0.2 < q < 1 else np.nan for q in ratio])


def quantities(d, sweep, ip):
    """Backscattered efficiency of each model, its relative error, the L1 error (Eq. (27)) and the fraction of the effect of the
    small scale on e+_{-1} captured by the model:  (e_smooth - e_model) / (e_smooth - e_ref)."""
    M = list(d["models"]); E = d["eff_" + sweep][ip]; ib = list(d["rs"]).index(R_BS)
    eb = {m: E[:, M.index(m), ib] for m in M}
    kb = d["xis"] if sweep == "xi" else np.full(len(d["sqeps"]), 0.25)
    eb["DW"] = eb["smooth"] * debye_waller(kb)
    ref = E[:, 0]
    L1 = {m: np.nansum(np.abs(E[:, M.index(m)] - ref), axis=1) / np.nansum(ref, axis=1) for m in M}
    rel = {m: (eb[m] - eb["ref"]) / eb["ref"] for m in eb}
    frac = {m: (eb["smooth"] - eb[m]) / (eb["smooth"] - eb["ref"]) for m in eb}
    return eb, rel, L1, frac


def summary():
    d = np.load(FILE); xis, sq = d["xis"], d["sqeps"]
    print(f"geometry L = {LR} lambda, aK = {2*np.pi*A_L:.3f}, Lambda = {LAM}, theta_i = {np.degrees(np.arcsin(SIN)):.2f} deg, backscattered order r = {R_BS}")
    for ip, pol in enumerate(POLS):
        eb, rel, L1, fr = quantities(d, "xi", ip)
        print(f"\n== {pol}, eps = 2.25: e_bs (ref), share of R, C, effect of the roughness on e_bs; per model: rel. error of e_bs [L1 error] f = fraction of the effect captured")
        for kb in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
            j = np.argmin(abs(xis - kb)); Rt = np.nansum(d["eff_xi"][ip, j, 0])
            print(f"  kb={xis[j]:.3f} e_bs={eb['ref'][j]:.3e} ({eb['ref'][j]/Rt:.2f} of R) C={d['C_xi'][ip, j]:.3f} effect {(eb['ref'][j]-eb['smooth'][j])/eb['smooth'][j]:+.3f}")
            print("     " + "  ".join(f"{m} {rel[m][j]:+.3f} [{L1[m][j]:.3f}] f={fr[m][j]:.2f}" for m in ("KA", "SSA1", "smooth", "ex0_n1", "ex0_n3", "KA0_n3")) + f"  DW {rel['DW'][j]:+.3f} f={fr['DW'][j]:.2f}")
    for ip, pol in enumerate(POLS):
        eb, rel, L1, fr = quantities(d, "eps", ip)
        print(f"\n== {pol}, kb = 0.25, versus eps")
        kinf = kb_from_attenuation(eb["ref"] / eb["smooth"])
        for j in range(0, len(sq), 3):
            print(f"  eps={sq[j]**2:5.2f} e_bs={eb['ref'][j]:.3e} effect {(eb['ref'][j]-eb['smooth'][j])/eb['smooth'][j]:+.3f} | " +
                  "  ".join(f"{m} {rel[m][j]:+.3f} f={fr[m][j]:.2f}" for m in ("KA", "SSA1", "DW", "ex0_n3")) + f"  kb from DW {kinf[j]:.3f}")


if __name__ == "__main__":
    cmds = dict(compute=compute, summary=summary)
    if len(sys.argv) < 2 or sys.argv[1] not in cmds: sys.exit(__doc__)
    cmds[sys.argv[1]]()
