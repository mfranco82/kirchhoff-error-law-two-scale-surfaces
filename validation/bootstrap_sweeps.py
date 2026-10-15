"""
Bootstrap over the parameter sweeps for the statistics of the cosine profiles (Sec. 4 of the paper; Fig. 2, Eqs. (28)-(29), Table 2).

The cosine profiles are not independent draws: they come from 12 sweeps (one parameter at a time around the base case: the 11 configurations of robustness.py, 7 small-scale
wavelengths x 4 kb each, and the (Lambda, kb) grid at 20 deg of phase_map.py).  The bootstrap resamples whole sweeps with replacement (B = 4000) and recomputes
  * the fit  eps_KA = alpha C^beta  (C > 0.03), TE and TM,
  * the two-variable fit  eps_KA ~ C^a s^c  (s = kb lambda/lambda_s), and the correlation of log C with log s (collinearity),
  * the limits of C of Table 2 (median and 90th percentile; first upward crossing of the binned curve, as in tolerance_Cmax.py) and the gains (ratio of limits), computed with the
    profiles of kb <= 0.5 (tolerance_Cmax.KBMAX) like the table, and the limits in the slope s (same profiles, bins of s of tolerance_Cmax.py),
and prints the median and the 95 % interval (2.5-97.5th percentiles) of each.  For comparison the same quantities are bootstrapped over single profiles (the usual, optimistic way).
Sweeps where a limit is not reached in the range (value inf) or where the bin is empty are counted separately.
usage: python bootstrap_sweeps.py          (~1-2 min)
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
import make_results_figures as M
import tolerance_Cmax as T

B = 4000
pol, fam, X, n_ex, n_all, clus = M.gather(return_cluster=True)
cos = fam == "cos"
pol, X, clus = pol[cos], X[cos], clus[cos]
C, eK, eM, eX, s, kb = X[:, 0], X[:, 2], X[:, 3], X[:, 4], X[:, 9], X[:, 10]
sweeps = np.unique(clus); idx_of = {c: np.flatnonzero(clus == c) for c in sweeps}
print(f"{len(C)} cosine profiles in {len(sweeps)} sweeps: " + ", ".join(f"{c} ({len(idx_of[c])})" for c in sweeps))
MODELS = (("KA", eK), ("KA0", eM), ("ex0", eX))
TOL = ((50, 1), (50, 2), (50, 5), (50, 10), (90, 5), (90, 10))


def stats(ix):
    out = {}
    for pl in ("TE", "TM"):
        m = ix[(pol[ix] == pl) & (C[ix] > 0.03)]
        k = np.polyfit(np.log(C[m]), np.log(eK[m]), 1); out[(pl, "alpha")] = np.exp(k[1]); out[(pl, "beta")] = k[0]
        A = np.column_stack([np.ones(len(m)), np.log(C[m]), np.log(s[m])]); co = np.linalg.lstsq(A, np.log(eK[m]), rcond=None)[0]
        out[(pl, "expC")] = co[1]; out[(pl, "exps")] = co[2]; out[(pl, "corr")] = np.corrcoef(np.log(C[m]), np.log(s[m]))[0, 1]
        mm = ix[(pol[ix] == pl) & (kb[ix] <= T.KBMAX)]
        for q, tol in TOL:
            lim = {}
            for name, e in MODELS:
                v = []
                for lo, hi in zip(T.BINS[:-1], T.BINS[1:]):
                    sel = mm[(C[mm] >= lo) & (C[mm] < hi)]
                    v.append(np.percentile(100 * e[sel], q) if len(sel) > 3 else np.nan)
                lim[name] = T.crossing(np.array(v), tol); out[(pl, q, tol, name)] = lim[name]
            out[(pl, q, tol, "gain")] = lim["KA0"] / lim["KA"] if lim["KA"] > 0 else np.nan
            ls = {}
            for name, e in MODELS:
                v = []
                for lo, hi in zip(T.BINS_S[:-1], T.BINS_S[1:]):
                    sel = mm[(s[mm] >= lo) & (s[mm] < hi)]
                    v.append(np.percentile(100 * e[sel], q) if len(sel) > 3 else np.nan)
                ls[name] = T.crossing(np.array(v), tol, T.MID_S); out[(pl, "s", q, tol, name)] = ls[name]
    return out


rng = np.random.default_rng(0)
full = stats(np.arange(len(C)))
res_sw, res_pr = [], []
for b in range(B):
    pick = rng.integers(0, len(sweeps), len(sweeps)); res_sw.append(stats(np.concatenate([idx_of[sweeps[i]] for i in pick])))
    res_pr.append(stats(rng.integers(0, len(C), len(C))))


def ci(res, key):
    v = np.array([r[key] for r in res], float); ok = np.isfinite(v)
    if not ok.any(): return np.full(3, np.inf), 100.0          # never reached in the range of the bins
    return np.percentile(v[ok], [2.5, 50, 97.5]), 100 * (1 - ok.mean())


print("\nfit eps_KA = alpha C^beta (cosine profiles, C>0.03): point estimate | 95 % over sweeps (median) | 95 % over profiles")
for pl in ("TE", "TM"):
    for key in ("alpha", "beta"):
        a, _ = ci(res_sw, (pl, key)); b_, _ = ci(res_pr, (pl, key))
        print(f"  {pl} {key:5s}: {full[(pl, key)]:.3f} | {a[0]:.3f}-{a[2]:.3f} ({a[1]:.3f}) | {b_[0]:.3f}-{b_[2]:.3f}")
print("\ntwo-variable fit eps ~ C^a s^c and collinearity: point | 95 % over sweeps | corr(log C, log s) point, 95 % over sweeps")
for pl in ("TE", "TM"):
    parts = []
    for key in ("expC", "exps", "corr"):
        a, _ = ci(res_sw, (pl, key)); parts.append(f"{key} {full[(pl, key)]:.2f} [{a[0]:.2f}, {a[2]:.2f}]")
    r = full[(pl, "corr")]; print(f"  {pl}: " + " | ".join(parts) + f" | variance inflation factor {1 / (1 - r ** 2):.1f}")
print("\nlimits of C (TE / TM) for a tolerance, with the 95 % interval over sweeps [and over profiles]; 'unr.' = % of resamples in which the limit is not reached or the bin is empty")
for q, tol in TOL:
    print(f"--- {'median' if q == 50 else '90th percentile'} error, tolerance {tol} %")
    for name in ("KA", "KA0", "ex0", "gain"):
        cells = []
        for pl in ("TE", "TM"):
            a, un = ci(res_sw, (pl, q, tol, name)); b_, _ = ci(res_pr, (pl, q, tol, name))
            cells.append(f"{full[(pl, q, tol, name)]:.2f} [{a[0]:.2f}-{a[2]:.2f}] ({b_[0]:.2f}-{b_[2]:.2f}) unr.{un:.0f}%")
        print(f"   {name:5s}: " + "  /  ".join(cells))

print("\nlimits of the slope s (TE / TM) for a tolerance, same profiles; 95 % interval over sweeps [and over profiles]")
for q, tol in TOL:
    print(f"--- {'median' if q == 50 else '90th percentile'} error, tolerance {tol} %")
    for name in ("KA", "KA0", "ex0"):
        cells = []
        for pl in ("TE", "TM"):
            a, un = ci(res_sw, (pl, "s", q, tol, name)); b_, _ = ci(res_pr, (pl, "s", q, tol, name))
            cells.append(f"{full[(pl, 's', q, tol, name)]:.2f} [{a[0]:.2f}-{a[2]:.2f}] ({b_[0]:.2f}-{b_[2]:.2f}) unr.{un:.0f}%")
        print(f"   {name:5s}: " + "  /  ".join(cells))
