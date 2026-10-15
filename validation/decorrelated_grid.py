"""
Is the error of KA a function of C or of the slope of the small scale?  (Sec. 4 of the paper, identifiability of the error law.)

For the cosine profiles z = a cos(Kx) + b cos(Lambda K x) with C_small = k b (lambda/lambda_s)^2/(2 cos^3) and s = k b lambda/lambda_s, any sweep in
(k b, Lambda) moves log C and log s together (log C = log kb + 2 log Lambda, log s = log kb + log Lambda: correlation >= 0.94 for ANY product grid).
Here the cells are chosen instead at fixed total C and Lambda varied, so that C is constant along a row and s = 16.6 C/Lambda changes by a factor of 5-8 (kb = 2 cos^3 (C - C_smooth)/(Lambda lambda/L)^2).
Profile as in phase_map.py (L = 10 lambda, a = 0.03 L, theta_i = 20 deg, eps = 2.25, TE and TM); reference: C-method; models: KA of the full profile and the smooth
profile alone.  Cells with k b > KB_MAX are skipped.
usage: python decorrelated_grid.py compute | fit      (compute: ~1 min on 8 cores; fit reads data/decorrelated_grid.npz)
"""
import os, sys, time, warnings
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data"); os.makedirs(DATA, exist_ok=True)
warnings.filterwarnings("ignore")
from scattering import *
from cmethod import c_method, spectral_derivative

WLr, A_L, TH, EPS, KB_MAX = 0.1, 0.03, 20.0, 2.25, 1.5
CTOT = [0.05, 0.1, 0.2, 0.4]
LAMS = [4, 5, 6, 8, 10, 12, 15, 18, 22, 25, 30]
POLS = ("TE", "TM")
FILE = os.path.join(DATA, "decorrelated_grid.npz")
CS = np.cos(np.deg2rad(TH)) ** 3
CSM = np.pi * A_L * WLr / CS                                   # validity parameter of the smooth cosine


def kb_of(C, Lam): return 2 * CS * (C - CSM) / (WLr * Lam) ** 2

def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()

def cell(args):
    ic, iL, ip = args
    C, Lam, pol = CTOT[ic], LAMS[iL], POLS[ip]; kb = kb_of(C, Lam)
    out = np.full(6, np.nan)           # KA full, smooth profile alone (exact), energy residual of the C-method, C-method(M) vs C-method(M+20) L1 distance, kb, s
    if kb > KB_MAX: return ic, iL, ip, out
    out[4], out[5] = kb, kb * Lam * WLr
    try:
        with np.errstate(all="ignore"):
            b = kb / (2 * np.pi / WLr)
            p0 = Problem(1.0, WLr, np.deg2rad(TH), EPS, pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, WLr, np.deg2rad(TH), EPS, pol, rmax, N=4096)
            f, f1 = cos_surface(p, A_L); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            Bc = c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)
            e = efficiencies(p, *[np.nan_to_num(x) for x in Bc]); ref = e[0]
            out[2] = e[0].sum() + e[1].sum() - 1
            out[0] = l1(efficiencies(p, *ka_amplitudes(p, F, F1))[0], ref, p.prop_p)
            out[1] = l1(efficiencies(p, *rayleigh_exact(p, f, f1))[0], ref, p.prop_p)
            p2 = Problem(1.0, WLr, np.deg2rad(TH), EPS, pol, rmax + 20, N=4096)
            f, f1 = cos_surface(p2, A_L); z, z1 = cos_surface(p2, b, Lam); F, F1 = f + z, f1 + z1
            ref2 = efficiencies(p2, *[np.nan_to_num(x) for x in c_method(p2, F, F1, spectral_derivative(p2, F, 2), M=rmax + 20, delta=1e-8)])[0]
            out[3] = np.abs(ref2[p2.prop_p] - ref[p.prop_p]).sum() / ref[p.prop_p].sum()
    except Exception as ex:
        print(pol, C, Lam, "failed:", ex, flush=True)
    return ic, iL, ip, out

def compute():
    tasks = [(ic, iL, ip) for ic in range(len(CTOT)) for iL in range(len(LAMS)) for ip in range(2)]
    data = np.full((len(CTOT), len(LAMS), 2, 6), np.nan); t0 = time.time()
    with Pool(8) as pool:
        for ic, iL, ip, out in pool.imap_unordered(cell, tasks): data[ic, iL, ip] = out
    np.savez(FILE, data=data, ctot=CTOT, lams=LAMS); print(f"saved {FILE} in {time.time()-t0:.0f} s; cells: {int(np.isfinite(data[..., 0]).sum())}")

def fit(boot=4000, seed=1):
    d = np.load(FILE)["data"]; rng = np.random.default_rng(seed)
    for ip, pol in enumerate(POLS):
        rows = []
        for ic, C in enumerate(CTOT):
            for iL, Lam in enumerate(LAMS):
                v = d[ic, iL, ip]
                if np.isfinite(v[0]): rows.append((C, v[5], v[0], v[1], v[2], v[3], Lam, v[4]))
        R = np.array(rows); C, s, e, es, en, dref, Lam, kb = R.T
        print(f"\n{pol}: {len(R)} cells; C-method |R+T-1| max {np.abs(en).max():.1e}, M vs M+20 max {dref.max():.1e};  s {s.min():.2f}-{s.max():.2f}, kb {kb.min():.2f}-{kb.max():.2f}, KA error {100*e.min():.2f}-{100*e.max():.0f} %")
        for label, m in (("all", np.ones(len(R), bool)), ("e<1", e < 1)):
            lc, ls, y, n = np.log10(C[m]), np.log10(s[m]), np.log10(e[m]), m.sum()
            def f(cols, idx):
                A = np.column_stack([np.ones(n)] + cols)[idx]; b = np.linalg.lstsq(A, y[idx], rcond=None)[0]; return b, (y[idx] - A @ b).std()
            all_ = np.arange(n); b1, s1 = f([lc], all_); b2, s2 = f([ls], all_); b3, s3 = f([lc, ls], all_)
            bs = []
            for _ in range(boot):
                idx = rng.integers(0, n, n)
                try: bs.append((f([lc], idx)[0][1], f([ls], idx)[0][1], *f([lc, ls], idx)[0][1:]))
                except Exception: pass
            q = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
            print(f"  {label:4s} n={n:2d} corr(logC,logs)={np.corrcoef(lc, ls)[0,1]:.2f}: C only {b1[1]:.2f} [{q[0,0]:.2f},{q[1,0]:.2f}] scatter {s1:.3f} | s only {b2[1]:.2f} [{q[0,1]:.2f},{q[1,1]:.2f}] {s2:.3f} | C^{b3[1]:.2f}[{q[0,2]:.2f},{q[1,2]:.2f}] s^{b3[2]:.2f}[{q[0,3]:.2f},{q[1,3]:.2f}] {s3:.3f}  (prefactor 10^{b3[0]:.2f})")
        print("  median KA error (%) by C (rows) over the cells of that C, and the ratio of the largest to the smallest KA error at fixed C:")
        for C0 in CTOT:
            w = C == C0
            print(f"    C={C0}: s {s[w].min():.2f}-{s[w].max():.2f}  error {100*e[w].min():.2f}-{100*e[w].max():.1f} %  ratio {e[w].max()/e[w].min():.1f}")

if __name__ == "__main__":
    {"compute": compute, "fit": fit}[sys.argv[1]]()
