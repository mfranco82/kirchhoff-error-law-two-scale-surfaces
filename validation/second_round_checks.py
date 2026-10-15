"""
Checks requested in the second referee round (numbers quoted in Sec. 4 of the paper):
  1. how much of the effect of the roughness each model recovers: rho = 1 - error_model / error_of_the_smooth_profile_alone (median per bin of C of the cosine profiles of Fig. 2),
  2. the Rayleigh-hypothesis criterion a K + b Lambda K < 0.6: number of cosine profiles above it, and the error of the series below and above it,
  3. Table 2 restricted to k b <= 0.5 (the range of the series) against the full set of profiles (k b up to 1),
  4. complex-amplitude metric vs efficiency metric: A+ against e+/2 (a relative amplitude error A in modulus gives a relative efficiency error 2A when there is no phase error),
  5. KA error on the smooth component alone of the random-phase profiles (smooth band of three harmonics) and of the cosines (the exact Rayleigh solution of the smooth component as reference).
usage: python second_round_checks.py        (reads the caches in ../data; ~30 s)
"""
import os, sys, warnings
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
import numpy as np
import make_results_figures as M, tolerance_Cmax as T, random_stats as RS, error_metrics as EM
from scattering import *

BINS = T.BINS
pol, fam, X, n_excl, n_all = M.gather()
C, eK, eM, eX, eS, cond, ray, slope, kb, aK = X[:, 0], X[:, 2], X[:, 3], X[:, 4], X[:, 7], X[:, 5], X[:, 6], X[:, 9], X[:, 10], X[:, 11]
cos = fam == "cos"

print("1. fraction of the roughness effect recovered, rho = 1 - e_model/e_smooth, median per bin of C (cosine profiles); TE | TM")
for name, e in (("KA full", eK), ("series KA0", eM), ("series exact0", eX)):
    line = f"  {name:14s}"
    for pl in ("TE", "TM"):
        v = []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            m = cos & (pol == pl) & (C >= lo) & (C < hi); v.append(np.median(1 - e[m] / eS[m]) if m.sum() > 3 else np.nan)
        line += " ".join(f"{x:6.2f}" for x in v) + "  | "
    print(line)
for pl in ("TE", "TM"):
    m = cos & (pol == pl) & (C >= 0.1) & (C < 0.8)
    print(f"  {pl} 0.1<C<0.8: median e_smooth {100*np.median(eS[m]):.1f} %, KA {100*np.median(eK[m]):.1f} %, series KA0 {100*np.median(eM[m]):.2f} %, exact0 {100*np.median(eX[m]):.2f} %; rho median: KA {np.median(1-eK[m]/eS[m]):.2f}, KA0 {np.median(1-eM[m]/eS[m]):.3f}, exact0 {np.median(1-eX[m]/eS[m]):.3f}; share of profiles where KA is worse than the smooth profile alone: {100*np.mean(eK[m]>eS[m]):.0f} %")

print("\n2. Rayleigh criterion aK + b Lambda K < 0.6 (cosine profiles of Fig. 2)")
crit = aK + slope
for lab, m in (("below 0.6", cos & (crit < 0.6)), ("above 0.6", cos & (crit >= 0.6))):
    print(f"  {lab}: {m.sum()} of {cos.sum()} profiles; Rayleigh(full) within 1e-3 of C-method: {100*np.mean(ray[m] < 1e-3):.0f} %; median error of the series exact0 {100*np.median(eX[m]):.2f} %, KA0 {100*np.median(eM[m]):.2f} %, KA {100*np.median(eK[m]):.1f} %, p90 exact0 {100*np.percentile(eX[m], 90):.1f} %")
m = cos & (crit >= 0.6) & (C < 0.8)
print(f"  above 0.6 and C<0.8: {m.sum()} profiles; median series exact0 {100*np.median(eX[m]):.2f} %, KA0 {100*np.median(eM[m]):.2f} %, p90 {100*np.percentile(eX[m], 90):.1f} / {100*np.percentile(eM[m], 90):.1f} %")
print(f"  maximum of aK + b Lambda K: {crit[cos].max():.2f}; maximum of aK alone {aK[cos].max():.2f}")

print("\n3. Table 2 limits of C, median (TE/TM): all profiles | k b <= 0.5")
res = {}
for lab, mk in (("all", np.ones(len(C), bool)), ("kb<=0.5", ~(kb > 0.5))):
    for q in (50, 90):
        for name, e in (("KA", eK), ("KA0", 100 * eM), ("ex0", 100 * eX)):
            ee = 100 * eK if name == "KA" else e
            res[(lab, q, name)] = [T.binned(pol[mk], fam[mk], C[mk], ee[mk], pl, q) for pl in ("TE", "TM")]
for q in (50, 90):
    for tol in ((1, 2, 5, 10) if q == 50 else (5, 10)):
        row = f"  q{q} {tol:2d}%: "
        for lab in ("all", "kb<=0.5"):
            for name in ("KA", "KA0", "ex0"):
                a = [T.crossing(res[(lab, q, name)][i], tol) for i in range(2)]; row += f"{name} {a[0]:.2f}/{a[1]:.2f}  "
            row += "|| "
        print(row)
print("  profiles per bin (TE) all | kb<=0.5:", [int((cos & (pol == 'TE') & (C >= lo) & (C < hi)).sum()) for lo, hi in zip(BINS[:-1], BINS[1:])], [int((cos & (pol == 'TE') & (C >= lo) & (C < hi) & ~(kb > 0.5)).sum()) for lo, hi in zip(BINS[:-1], BINS[1:])])

print("\n4. complex-amplitude metric A+ vs the efficiency error e+ (error_metrics cache), 0.1<C<0.8; models KA | smooth | KA0 | exact0")
d = np.load(EM.FILE); Y = d["data"].reshape(len(d["C"]), len(EM.MODELS), len(EM.METRICS)); Cm = d["C"]
ok = np.isfinite(Y).all(axis=(1, 2)) & (Cm >= 0.1) & (Cm < 0.8)
je, ja, jn, jan = EM.METRICS.index("e+"), EM.METRICS.index("A+"), EM.METRICS.index("ens"), EM.METRICS.index("Ans")
print("  median A+ / (e+/2):  " + "  ".join(f"{mod} {np.median(Y[ok, k, ja] / (Y[ok, k, je] / 2)):.2f}" for k, mod in enumerate(EM.MODELS)))
print("  median A+ / e+:      " + "  ".join(f"{mod} {np.median(Y[ok, k, ja] / Y[ok, k, je]):.2f}" for k, mod in enumerate(EM.MODELS)))
print("  median ens / e+:     " + "  ".join(f"{mod} {np.median(Y[ok, k, jn] / Y[ok, k, je]):.2f}" for k, mod in enumerate(EM.MODELS)) + "   (ens: L1 over r != 0 normalised by the reference efficiency of r != 0)")
print("  median Ans / A+:     " + "  ".join(f"{mod} {np.median(Y[ok, k, jan] / Y[ok, k, ja]):.2f}" for k, mod in enumerate(EM.MODELS)))
print("  share of the reflected energy in the specular order (median, 0.1<C<0.8): see check 5b below")

print("\n5. KA error on the smooth component alone (reference: Rayleigh solution of the smooth component, aK <= 0.28)")
K = 2 * np.pi; k = K / RS.WLr
out = []
for ib, (band, (m1, m2)) in enumerate(RS.BANDS.items()):
    for ik in range(len(RS.KSIG)):
        for ir in range(RS.NREAL):
            rng = np.random.default_rng(10_000 * (ib + 1) + 100 * ik + ir); ph = rng.uniform(0, 2 * np.pi, len(RS.M_S))
            for pl in ("TE", "TM"):
                with np.errstate(all="ignore"):
                    p = Problem(1.0, RS.WLr, np.arcsin(RS.SIN), RS.EPS, pl, 30, N=4096)
                    f = sum(RS.A_S * np.cos(m * p.K * p.x + a) for m, a in zip(RS.M_S, ph)); f1 = sum(-RS.A_S * m * p.K * np.sin(m * p.K * p.x + a) for m, a in zip(RS.M_S, ph))
                    ref = efficiencies(p, *rayleigh_exact(p, f, f1))[0]; ka = efficiencies(p, *ka_amplitudes(p, f, f1))[0]
                    out.append((pl == "TM", np.abs(ka - ref)[p.prop_p].sum() / ref[p.prop_p].sum(), np.abs(ref[p.prop_p][p.r[p.prop_p] == 0]).sum() / ref[p.prop_p].sum()))
out = np.array(out)
for i, pl in enumerate(("TE", "TM")):
    v = out[out[:, 0] == i, 1]; print(f"  random-phase smooth component, {pl}: KA error median {100*np.median(v):.2f} %, p10-p90 {100*np.percentile(v,10):.2f}-{100*np.percentile(v,90):.2f} %, max {100*v.max():.1f} %   (single cosine of the paper: 0.28 % TE, 0.49 % TM)")
