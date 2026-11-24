"""
First-order small-slope approximation (SSA-1, ssa.py) on the same profiles as the paper:
the cosine sweeps (robustness.py, phase_map.py), the random-phase profiles (random_stats.py) and the
decorrelated design (validation/decorrelated_grid.py).  The C-method reference is recomputed with the same
settings, and the error of KA on the full profile is recomputed as a check against the stored data.
Columns per case: 0 eR KA full | 1 eR SSA-1 | 2 eT KA full | 3 eT SSA-1 | 4 eR SSA-1 smooth profile vs its Rayleigh solution | 5 same for KA
usage: python ssa_sweeps.py compute [nproc]   (stores ../data/ssa_sweeps.npz) | summary
"""
import os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from scattering import Problem, cos_surface, ka_amplitudes, rayleigh_exact, efficiencies
from cmethod import c_method, spectral_derivative
import ssa
import robustness as R, phase_map as P, random_stats as RS, decorrelated_grid as D

FILE = os.path.join(HERE, "data", "ssa_sweeps.npz")
NC = 6


def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


def evaluate(p, f, f1, z, z1):
    out = np.full(NC, np.nan)
    F, F1 = f + z, f1 + z1
    ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=p.rmax, delta=1e-8)])
    eK = efficiencies(p, *ka_amplitudes(p, F, F1)); eS = efficiencies(p, *ssa.ssa1_propagating(p, F))
    out[0], out[1] = l1(eK[0], ref[0], p.prop_p), l1(eS[0], ref[0], p.prop_p)
    out[2], out[3] = l1(eK[1], ref[1], p.prop_m), l1(eS[1], ref[1], p.prop_m)
    rs = efficiencies(p, *rayleigh_exact(p, f, f1))[0]
    out[4] = l1(efficiencies(p, *ssa.ssa1_propagating(p, f))[0], rs, p.prop_p)
    out[5] = l1(efficiencies(p, *ka_amplitudes(p, f, f1))[0], rs, p.prop_p)
    return out


def problem(wl, th, eps, pol, Lam):
    p0 = Problem(1.0, wl, th, eps, pol, 30, N=4096)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam + 12)
    return Problem(1.0, wl, th, eps, pol, rmax, N=4096)


def task(t):
    kind, idx = t
    try:
        with np.errstate(all="ignore"):
            if kind == "rob":
                ic, ip, iL, ik = idx; c = R.CONF[R.NAMES[ic]]; Lr = c["L"]; Lam = R.LAMS[Lr][iL]; kb = R.KBS[ik]
                wl = 1.0 / Lr; p = problem(wl, np.arcsin(c["sin"]), c["eps"], R.POLS[ip], Lam)
                f, f1 = cos_surface(p, c["a"]); z, z1 = cos_surface(p, kb * wl / (2 * np.pi), Lam)
            elif kind == "pm":
                ip, iL, ik = idx; Lam = P.LAMS[iL]; p = problem(P.WLr, np.deg2rad(P.TH), P.EPS, P.POLS[ip], Lam)
                f, f1 = cos_surface(p, P.A_L); z, z1 = cos_surface(p, P.KBS[ik] / (2 * np.pi / P.WLr), Lam)
            elif kind == "rnd":
                ib, ik, ip, ir = idx; band = list(RS.BANDS)[ib]
                rng = np.random.default_rng(10_000 * (ib + 1) + 100 * ik + ir)
                p = problem(RS.WLr, np.arcsin(RS.SIN), RS.EPS, RS.POLS[ip], RS.BANDS[band][1])
                f, f1, z, z1, _ = RS.build(p, band, RS.KSIG[ik], rng)
            else:  # decorrelated design
                ic, iL, ip = idx; C, Lam = D.CTOT[ic], D.LAMS[iL]; kb = D.kb_of(C, Lam)
                if kb > D.KB_MAX: return t, np.full(NC, np.nan)
                p = problem(D.WLr, np.deg2rad(D.TH), D.EPS, D.POLS[ip], Lam)
                f, f1 = cos_surface(p, D.A_L); z, z1 = cos_surface(p, kb / (2 * np.pi / D.WLr), Lam)
            return t, evaluate(p, f, f1, z, z1)
    except Exception as ex:
        print(t, "failed:", ex, flush=True)
        return t, np.full(NC, np.nan)


SHAPES = {"rob": (len(R.NAMES), 2, 7, len(R.KBS)), "pm": (2, len(P.LAMS), len(P.KBS)),
          "rnd": (len(RS.BANDS), len(RS.KSIG), 2, RS.NREAL), "dec": (len(D.CTOT), len(D.LAMS), 2)}


def compute(nproc=6):
    tasks = [(k, idx) for k, sh in SHAPES.items() for idx in np.ndindex(*sh)]
    data = {k: np.full(sh + (NC,), np.nan) for k, sh in SHAPES.items()}
    t0 = time.time()
    with Pool(nproc) as pool:
        for n, ((k, idx), out) in enumerate(pool.imap_unordered(task, tasks, chunksize=2)):
            data[k][idx] = out
            if n % 200 == 0: print(f"{n}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    np.savez(FILE, **data)
    print(f"saved {FILE} in {time.time()-t0:.0f} s; failed: " + ", ".join(f"{k} {int(np.isnan(v[..., 0]).sum())}" for k, v in data.items()), flush=True)


def aligned(keep_all=False):
    """SSA columns in the row order of make_results_figures.gather (same exclusions); returns gather's output plus Y (rows x NC)."""
    import make_results_figures as M
    d = np.load(FILE); ex = ("sin=0.75", "sin=0.85", "aK=0.38")
    Y = [d["rob"][ic, ip, iL, ik] for ic, n in enumerate(R.NAMES) if n not in ex for ip in range(2)
         for iL in range(7) for ik in range(len(R.KBS))]
    Y += [d["pm"][idx] for idx in np.ndindex(*SHAPES["pm"])]
    Y += [d["rnd"][idx] for idx in np.ndindex(*SHAPES["rnd"])]
    Y = np.array(Y)
    pol, fam, X, ok = M.gather(keep_all=True)
    assert len(Y) == len(X)
    return pol[ok], fam[ok], X[ok], Y[ok]


def summary():
    import tolerance_Cmax as T
    pol, fam, X, Y = aligned()
    C, eK, eX, s = X[:, 0], X[:, 2], X[:, 4], X[:, 9]
    cos = fam == "cos"
    dif = np.abs(Y[:, 0] - eK) / eK
    print(f"check: KA recomputed vs stored, max relative difference {np.nanmax(dif):.1e}; SSA failures: {int(np.isnan(Y[:, 1]).sum())}")
    for pl in ("TE", "TM"):
        m = (pol == pl)
        print(f"\n=== {pl}: median error (%) of the cosine profiles in bins of C   [KA | SSA-1 | series ex0]   (n)")
        for lo, hi in zip(T.BINS[:-1], T.BINS[1:]):
            b = m & cos & (C >= lo) & (C < hi)
            if b.sum() > 3:
                print(f"  C {lo:4.2f}-{hi:4.2f}: {100*np.median(eK[b]):7.2f} | {100*np.nanmedian(Y[b,1]):7.2f} | {100*np.median(eX[b]):6.2f}   ({b.sum()})")
        b = m & cos
        print(f"  SSA better than KA in {100*np.mean(Y[b,1] < eK[b]):.0f}% of the cosines; median ratio SSA/KA {np.nanmedian(Y[b,1]/eK[b]):.2f}")
        for lo, hi in ((0, 0.05), (0.05, 0.2), (0.2, 0.8), (0.8, 9)):
            bb = b & (C >= lo) & (C < hi)
            print(f"     C {lo}-{hi}: SSA<KA in {100*np.mean(Y[bb,1] < eK[bb]):.0f}%, median SSA/KA {np.nanmedian(Y[bb,1]/eK[bb]):.2f}, SSA<series(ex0) in {100*np.mean(Y[bb,1] < eX[bb]):.0f}%  ({bb.sum()})")
        r = m & ~cos
        print(f"  random-phase: median KA {100*np.median(eK[r]):.2f}%, SSA {100*np.nanmedian(Y[r,1]):.2f}%, SSA<KA in {100*np.mean(Y[r,1] < eK[r]):.0f}%")
        print(f"  smooth profile alone (cosine, vs its Rayleigh solution): median KA {100*np.nanmedian(Y[b,5]):.2f}%, SSA {100*np.nanmedian(Y[b,4]):.2f}%")
        sel = b & (C > 0.03) & np.isfinite(Y[:, 1]) & (Y[:, 1] > 0)
        A = np.column_stack([np.ones(sel.sum()), np.log(C[sel]), np.log(s[sel])]); co, *_ = np.linalg.lstsq(A, np.log(Y[sel, 1]), rcond=None)
        res = np.std(np.log10(Y[sel, 1]) - (A @ co) / np.log(10))
        print(f"  SSA fit (C>0.03): {np.exp(co[0]):.2f} C^{co[1]:.2f} s^{co[2]:.2f}, scatter {res:.2f} decades")
        kk = b & (X[:, 10] <= T.KBMAX)
        for name, e in (("KA", 100 * eK), ("SSA-1", 100 * Y[:, 1]), ("series ex0", 100 * eX)):
            v = T.binned(pol[kk], fam[kk], C[kk], e[kk], pl, 50); vs = T.binned_s(pol[kk], fam[kk], s[kk], e[kk], pl, 50)
            print(f"  limits kb<=0.5 {name:10s}: C " + " ".join(f"{T.crossing(v, t):5.2f}" for t in T.TOLS) + "   s " + " ".join(f"{T.crossing(vs, t, T.MID_S):5.2f}" for t in T.TOLS))
    dd = np.load(FILE)["dec"]
    print("\ndecorrelated design: SSA vs KA at fixed C  (median over Lambda; range)")
    for ip, pl in enumerate(("TE", "TM")):
        for ic, Cv in enumerate(D.CTOT):
            v = dd[ic, :, ip]; ok = np.isfinite(v[:, 1])
            print(f"  {pl} C={Cv:4.2f}: KA {100*np.nanmin(v[:,0]):5.1f}-{100*np.nanmax(v[:,0]):5.1f}%   SSA {100*np.nanmin(v[:,1]):5.1f}-{100*np.nanmax(v[:,1]):5.1f}%   ({ok.sum()} cells)")


if __name__ == "__main__":
    if sys.argv[1] == "compute": compute(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    else: summary()
