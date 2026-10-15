"""
Convergence of the series with the perturbative order n across the parameter space (Sec. 4 of the paper), and Pade summation.

Cells: the configurations of robustness.py (L/lambda, incidence angle, eps, aK; grazing incidence and aK = 0.38 left out, as in Fig. 2), the 7 small-scale wavelengths of each
and kb = 0.2, 0.5, 1.0, TE and TM  (cosine profiles z = a cos(Kx) + b cos(Lambda K x)).  Reference: C-method.  For every cell it stores
  * the L1 error (reflected efficiencies) of the partial sums n = 0..NMAX with the exact zeroth order and with the Kirchhoff zeroth order (propagating orders),
  * the error of the diagonal Pade approximants [m/m], m = 1..4, of the series in xi (xi = 1) built from the exact-zeroth-order coefficients of the propagating orders,
  * the ratios ||d_n|| / ||d_{n-1}|| of the coefficients over the propagating reflected orders (exact zeroth order), n = 2..NMAX
    (a ratio below 1 at n = NMAX means that the series converges at xi = 1 if the ratio persists; 1/ratio estimates the radius of convergence in units of xi),
  * C (Eq. (10)), the small-scale slope and the condition number of the matrix of the smooth profile.
usage: python series_convergence.py compute | summary | rmax_check   (compute: ~8 min on 8 cores; stores ../data/series_convergence.npz; rmax_check: ~1 min)
`rmax_check`: base configuration, TE, kb = 0.5 and 1, Lambda = 5..20, partial sums n = 0..8 with r_max of the rule (3 Lambda + 12 or the transmitted orders + 12) and with r_max = 8 Lambda + 12 (all the couplings of order 8),
with the condition number: for Lambda <= 8 the series keeps converging to n = 8 even at kb = 1; for larger Lambda the larger r_max makes the partial sums diverge numerically (condition number up to 1e25).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scattering.py, cmethod.py, robustness.py live in the parent folder
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import pade
import robustness as R
from scattering import *
from cmethod import c_method, spectral_derivative

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE = os.path.join(HERE, "data", "series_convergence.npz")
NMAX = 8
KBS = [0.2, 0.5, 1.0]
PADE = (1, 2, 3, 4)
NAMES = [n for n in R.NAMES if n not in ("sin=0.75", "sin=0.85", "aK=0.38")]
def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


def cell(args):
    ic, ip, iL, ik = args
    name, c = NAMES[ic], R.CONF[NAMES[ic]]
    Lrel = c["L"]; Lam = R.LAMS[Lrel][iL]; kb = KBS[ik]; pol = R.POLS[ip]
    out = np.full(2 * (NMAX + 1) + len(PADE) + (NMAX - 1) + 4, np.nan)
    try:
        with np.errstate(all="ignore"):
            wl = 1.0 / Lrel; th = np.arcsin(c["sin"]); b = kb * wl / (2 * np.pi)
            p0 = Problem(1.0, wl, th, c["eps"], pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
            p = Problem(1.0, wl, th, c["eps"], pol, rmax, N=4096)
            f, f1 = cos_surface(p, c["a"]); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
            ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])[0]
            err = lambda B: l1(efficiencies(p, *B)[0], ref, p.prop_p)
            op = SmoothOperator(p, f, f1)
            k = 0
            for B0 in (rayleigh_exact(p, f, f1), ka_amplitudes_propagating(p, f, f1)):
                dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, NMAX, smooth=op)
                for n in range(NMAX + 1): out[k] = err(mvb_sum(dp, dm, 1, n)); k += 1
                if k == NMAX + 1:                                                     # exact zeroth order: Pade and ratios
                    Ap = np.array([d[p.prop_p] for d in dp]); Am = np.array([d[p.prop_m] for d in dm])
                    for m in PADE:
                        Bp, Bm = mvb_sum(dp, dm, 1, NMAX)                              # evanescent orders irrelevant for the efficiencies
                        Bp, Bm = Bp.copy(), Bm.copy()
                        for j in range(Ap.shape[1]):
                            q_, r_ = pade(Ap[:, j], m, m); Bp[np.flatnonzero(p.prop_p)[j]] = q_(1.0) / r_(1.0)
                        for j in range(Am.shape[1]):
                            q_, r_ = pade(Am[:, j], m, m); Bm[np.flatnonzero(p.prop_m)[j]] = q_(1.0) / r_(1.0)
                        out[k] = err((Bp, Bm)); k += 1
                    nrm = [np.linalg.norm(Ap[n]) for n in range(NMAX + 1)]
                    for n in range(2, NMAX + 1): out[k] = nrm[n] / nrm[n - 1]; k += 1
            cs3 = np.cos(th) ** 3
            out[k] = (abs(spectral_derivative(p, F, 2)).max()) / (2 * p.k * cs3); out[k + 1] = kb * Lam / Lrel; out[k + 2] = np.linalg.cond(system_matrix(p, f, f1)); out[k + 3] = rmax
    except Exception as e:
        pass
    return ic, ip, iL, ik, out


def compute():
    tasks = [(ic, ip, iL, ik) for ic in range(len(NAMES)) for ip in range(2) for iL in range(7) for ik in range(len(KBS))]
    data = None; t0 = time.time()
    with Pool(8) as pool:
        for k, (ic, ip, iL, ik, out) in enumerate(pool.imap_unordered(cell, tasks, chunksize=2)):
            if data is None: data = np.full((len(NAMES), 2, 7, len(KBS), len(out)), np.nan)
            data[ic, ip, iL, ik] = out
            if k % 50 == 0: print(f"{k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    np.savez(FILE, data=data, names=NAMES, kbs=KBS)
    print(f"saved {FILE} in {time.time()-t0:.0f} s; failed cells: {int(np.isnan(data[..., 0]).sum())}")


def load():
    d = np.load(FILE)["data"]; nx = NMAX + 1; npd = len(PADE)
    ex = d[..., :nx]; pd = d[..., nx:nx + npd]; rat = d[..., nx + npd:nx + npd + NMAX - 1]; ka = d[..., nx + npd + NMAX - 1:2 * nx + npd + NMAX - 1]
    C, slope, cond, rmax = [d[..., -4 + i] for i in range(4)]
    return ex, ka, pd, rat, C, slope, cond, rmax


def summary():
    ex, ka, pd, rat, C, slope, cond, rmax = load()
    ok = np.isfinite(ex).all(-1) & np.isfinite(C)
    print(f"{ok.sum()} cells of {ok.size} (failed: {(~ok).sum()}); n = 0..{NMAX}")
    print("\nmedian error (%) of the partial sums with the exact zeroth order, by kb (all configurations, TE and TM together) and by bin of C")
    bins = [0.02, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0, 30]
    for ik, kb in enumerate(KBS):
        m = ok[..., ik]
        print(f"  kb={kb}: n=" + " ".join(f"{n:8d}" for n in range(NMAX + 1)))
        print(f"          median " + " ".join(f"{100*np.median(ex[..., ik, n][m]):8.3g}" for n in range(NMAX + 1)) + f"   (cells {m.sum()})")
        print(f"          p90    " + " ".join(f"{100*np.percentile(ex[..., ik, n][m],90):8.3g}" for n in range(NMAX + 1)))
    print("\nconvergence: fraction of cells in which the error of the partial sum n=8 is below that of n=3 and below that of n=5 (exact zeroth order, error at n=3 above 1e-3 only)")
    for ik, kb in enumerate(KBS):
        m = ok[..., ik] & (ex[..., ik, 3] > 1e-3)
        print(f"  kb={kb}: cells {m.sum()}: e8<e3 in {100*np.mean(ex[..., ik, 8][m] < ex[..., ik, 3][m]):.0f}%, e8<e5 in {100*np.mean(ex[..., ik, 8][m] < ex[..., ik, 5][m]):.0f}%, e5<e3 in {100*np.mean(ex[..., ik, 5][m] < ex[..., ik, 3][m]):.0f}%, "
              f"median e3/e8 = {np.median(ex[..., ik, 3][m] / ex[..., ik, 8][m]):.3g}, cells with e8>100% : {np.sum(ex[..., ik, 8][m] > 1)}")
    print("\nratio ||d_n||/||d_{n-1}|| at n=8 (exact zeroth order; <1: converging at xi=1), by kb and by bin of C:")
    for ik, kb in enumerate(KBS):
        m = ok[..., ik]; r8 = rat[..., ik, -1][m]; rr = rat[..., ik, -3][m]
        print(f"  kb={kb}: median ratio n=8 {np.median(r8):.2f}, p10-p90 {np.percentile(r8,10):.2f}-{np.percentile(r8,90):.2f}, share above 1: {100*np.mean(r8>1):.0f}%   (n=6: median {np.median(rr):.2f})")
    print("\nmedian error (%) of the diagonal Pade approximants [m/m] (m=1..4, built from n<=2m) against the partial sum n=8, by kb")
    for ik, kb in enumerate(KBS):
        m = ok[..., ik] & np.isfinite(pd[..., ik, :]).all(-1)
        print(f"  kb={kb}: Pade " + " ".join(f"[{mm}/{mm}] {100*np.median(pd[..., ik, j][m]):8.3g}" for j, mm in enumerate(PADE)) + f"   Taylor n=8 {100*np.median(ex[..., ik, 8][m]):8.3g}   p90 Pade[4/4] {100*np.percentile(pd[..., ik, 3][m],90):.3g} vs Taylor n=8 {100*np.percentile(ex[..., ik, 8][m],90):.3g};  Pade[4/4]<Taylor8 in {100*np.mean(pd[..., ik, 3][m] < ex[..., ik, 8][m]):.0f}%")
    print("\nerror (%) at n=3 / n=5 / n=8 / Pade[4/4] (exact zeroth order, median over cells) by bin of C")
    Cc = C
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = ok & (Cc >= lo) & (Cc < hi)
        if m.sum() > 3:
            print(f"  C {lo:5.2f}-{hi:5.2f} (n={m.sum():3d}): " + " / ".join(f"{100*np.median(x[m]):8.3g}" for x in (ex[..., 3], ex[..., 5], ex[..., 8], pd[..., 3])) + f"   | share of cells with e3<1%: {100*np.mean(ex[..., 3][m] < .01):.0f}%, e8<1%: {100*np.mean(ex[..., 8][m] < .01):.0f}%, Pade<1%: {100*np.mean(pd[..., 3][m] < .01):.0f}%")


def reliable():
    """Cells in which the order-n partial sum is numerically meaningful: the retained Rayleigh orders contain every coupling of order n (n*Lambda <= r_max)
    and the matrix of the smooth profile has a condition number below 1e16 (the series is then limited by the truncation of the Rayleigh basis, not by its numerical stability)."""
    ex, ka, pd, rat, C, slope, cond, rmax = load()
    Lam = np.zeros(C.shape)
    for ic, name in enumerate(NAMES):
        Lam[ic] = np.array(R.LAMS[R.CONF[name]["L"]])[None, :, None]
    return ex, ka, pd, rat, C, slope, cond, rmax, Lam


def summary_reliable():
    ex, ka, pd, rat, C, slope, cond, rmax, Lam = reliable()
    print("\nreliable cells only (n*Lambda <= r_max and cond < 1e16): exact zeroth order, error of the partial sum n against n-1")
    for ik, kb in enumerate(KBS):
        for n in range(3, NMAX + 1):
            m = (n * Lam[..., ik] <= rmax[..., ik]) & (cond[..., ik] < 1e16) & np.isfinite(ex[..., ik, n]) & (ex[..., ik, n - 1] > 1e-6)
            if m.sum() > 5:
                q = ex[..., ik, n][m] / ex[..., ik, n - 1][m]
                print(f"  kb={kb} n={n}: cells {m.sum():3d}, error decreases in {100*np.mean(q < 1):3.0f}%, median ratio e_n/e_(n-1) {np.median(q):.3f}, p90 {np.percentile(q, 90):.3g}; median error {100*np.median(ex[..., ik, n][m]):.3g}%")
    print("\nreliable cells only: diagonal Pade [m/m] (n <= 2m) against the partial sum n = 2m (same coefficients)")
    for ik, kb in enumerate(KBS):
        for j, mm in enumerate(PADE):
            n = 2 * mm
            m = (n * Lam[..., ik] <= rmax[..., ik]) & (cond[..., ik] < 1e16) & np.isfinite(pd[..., ik, j])
            if n <= NMAX and m.sum() > 5:
                print(f"  kb={kb} [{mm}/{mm}]: cells {m.sum():3d}, Pade better in {100*np.mean(pd[..., ik, j][m] < ex[..., ik, n][m]):3.0f}%, median error Pade {100*np.median(pd[..., ik, j][m]):.3g}% against {100*np.median(ex[..., ik, n][m]):.3g}% (Taylor n={n}); p90 {100*np.percentile(pd[..., ik, j][m], 90):.3g}% against {100*np.percentile(ex[..., ik, n][m], 90):.3g}%")
    print("\nshare of cells with C < 0.8 / C in 0.8-1.6 in which the series with n = 4 (exact zeroth order, reliable cells) reaches 1 %")
    for lo, hi in ((0.02, 0.4), (0.4, 0.8), (0.8, 1.6)):
        for n in (3, 4, 5, 6):
            m = (C >= lo) & (C < hi) & (n * Lam <= rmax) & (cond < 1e16) & np.isfinite(ex[..., n])
            if m.sum() > 5: print(f"  C {lo}-{hi} n={n}: cells {m.sum():3d}, median error {100*np.median(ex[..., n][m]):.3g}%, below 1% in {100*np.mean(ex[..., n][m] < .01):.0f}%, below 5% in {100*np.mean(ex[..., n][m] < .05):.0f}%")


def _rmax_cell(a):
    pol, Lam, kb, mode = a
    c = R.CONF["base"]; wl = 0.1; th = np.arcsin(c["sin"]); b = kb * wl / (2 * np.pi)
    p0 = Problem(1.0, wl, th, c["eps"], pol, 30, N=4096); rm0 = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
    rmax = rm0 if mode == "rule" else max(rm0, NMAX * Lam + 12); N = 4096
    while N < 6 * rmax: N *= 2
    p = Problem(1.0, wl, th, c["eps"], pol, rmax, N=N)
    f, f1 = cos_surface(p, c["a"]); z, z1 = cos_surface(p, b, Lam); F, F1 = f + z, f1 + z1
    ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])[0]
    op = SmoothOperator(p, f, f1); dp, dm = mvb_coefficients(p, f, f1, z, z1, rayleigh_exact(p, f, f1), NMAX, smooth=op)
    return a, rmax, np.linalg.cond(system_matrix(p, f, f1)), [100 * l1(efficiencies(p, *mvb_sum(dp, dm, 1, n))[0], ref, p.prop_p) for n in range(NMAX + 1)]


def rmax_check():
    tasks = [("TE", Lam, kb, mode) for Lam in (5, 8, 10, 15, 20) for kb in (0.5, 1.0) for mode in ("rule", "big")]
    print("error (%) of the partial sums n = 0..8 (exact zeroth order)")
    with Pool(8) as pool:
        for a, rm, cn, e in pool.imap(_rmax_cell, tasks): print(a, f"r_max={rm} cond={cn:.0e}", " ".join(f"{x:9.3g}" for x in e), flush=True)


if __name__ == "__main__":
    {"compute": compute, "summary": lambda: (summary(), summary_reliable()), "rmax_check": rmax_check}[sys.argv[1]]()
