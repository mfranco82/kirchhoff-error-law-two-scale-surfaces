"""
Does the error of KA (full profile) depend on C alone?  (Sec. 4 of the paper, remark after Eq. (29).)

Same cosine profiles and same selection as Fig. 2 and Eqs. (28)-(29) (robustness.npz and phase_map.npz, C > 0.03), and the random
surfaces of random_stats.npz.  C combines the height and the length scale of the roughness (C_small = k b (lambda/lambda_s)^2 / (2 cos^3)); the
slope of the small scale s = k b lambda/lambda_s (random surfaces: k sigma_psi times the rms wavenumber of the band over k) is the other
combination.  For each polarization it prints, in decades of log10(error):
  * the scatter around the straight line in log C (Eq. (29)),
  * the scatter with the slope of the small scale alone, and with C and s together (log10 e = a + b log10 C + c log10 s),
  * for the cosine profiles, the median error in narrow windows of C split at the median slope (the C of the two halves are matched).
usage: python ka_error_second_variable.py          (reads the caches in ../data; a few seconds)
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np
import robustness as R, phase_map as P, random_stats as RS

# ---- cosine profiles: pol, C, error of KA (full profile), small-scale slope, cond, finite
rows = []
d = np.load(R.FILE)["data"]
for ic, name in enumerate(R.NAMES):
    if name in ("sin=0.75", "sin=0.85", "aK=0.38"): continue
    c = R.CONF[name]; Lr = c["L"]; cs = np.sqrt(1 - c["sin"] ** 2) ** 3
    for ip, pol in enumerate(R.POLS):
        for iL, Lam in enumerate(R.LAMS[Lr]):
            for ik, kb in enumerate(R.KBS):
                v = d[ic, ip, iL, ik]; Csm = np.pi * c["a"] / Lr / cs; Cl = np.pi * kb / (2 * np.pi * Lr) * Lam ** 2 / Lr / cs
                rows.append((pol, Csm + Cl, v[0], kb * Lam / Lr, v[6], np.isfinite(v[:6]).all()))
pm = np.load(P.FILE)["data"]; cs = np.cos(np.deg2rad(P.TH)) ** 3
for ip, pol in enumerate(P.POLS):
    for iL, Lam in enumerate(P.LAMS):
        for ik, kb in enumerate(P.KBS):
            v = pm[ip, iL, ik]; Csm = np.pi * P.A_L * P.WLr / cs; Cl = np.pi * (kb * P.WLr / (2 * np.pi)) * Lam ** 2 * P.WLr / cs
            rows.append((pol, Csm + Cl, v[1], kb * Lam * P.WLr, v[4], np.isfinite(v[1:5]).all()))
pol = np.array([r[0] for r in rows]); X = np.array([r[1:5] for r in rows], float); fin = np.array([r[5] for r in rows])
C, e, s, cond = X.T
ok = fin & (C > 0.03) & (e > 0)

# ---- random surfaces
rd = np.load(RS.FILE)["data"]; xb = [np.sqrt((np.arange(m1, m2 + 1) ** 2).mean()) * P.WLr for (m1, m2) in RS.BANDS.values()]   # rms wavenumber / k (L = 10 lambda)
rr = []
for ib in range(rd.shape[0]):
    for ik in range(rd.shape[1]):
        for ip, pl in enumerate(RS.POLS):
            for ir in range(rd.shape[3]):
                v = rd[ib, ik, ip, ir]; rr.append((pl, v[7], v[0], RS.KSIG[ik] * xb[ib], v[6]))
polr = np.array([r[0] for r in rr]); Y = np.array([r[1:] for r in rr], float); Cr, er, sr, cr = Y.T
okr = np.isfinite(er) & (er > 0) & (Cr > 0.03)


def report(label, pol_, C_, e_, s_, ok_):
    for pl in ("TE", "TM"):
        m = ok_ & (pol_ == pl); y = np.log10(e_[m]); lc = np.log10(C_[m]); ls = np.log10(s_[m])

        def fit(cols):
            A = np.column_stack([np.ones(m.sum())] + cols); b = np.linalg.lstsq(A, y, rcond=None)[0]; return b, (y - A @ b).std()
        b1, s1 = fit([lc]); b2, s2 = fit([ls]); b3, s3 = fit([lc, ls])
        print(f"{label} {pl}: n={m.sum():3d} | C only: exponent {b1[1]:.2f}, scatter {s1:.3f} | slope only: exponent {b2[1]:.2f}, {s2:.3f} | C and slope: {10 ** b3[0]:.2f} C^{b3[1]:.2f} s^{b3[2]:.2f}, {s3:.3f}")


print("small-scale slope of the cosine profiles: %.2f - %.2f;  random surfaces: %.2f - %.2f" % (s[ok].min(), s[ok].max(), sr[okr].min(), sr[okr].max()))
report("cosines", pol, C, e, s, ok)
report("random ", polr, Cr, er, sr, okr)
print("\ncosine profiles: median error (%) in narrow windows of C, split at the median small-scale slope (smaller slope | larger slope):")
for pl in ("TE", "TM"):
    m = ok & (pol == pl)
    for lo, hi in ((0.05, 0.08), (0.1, 0.14), (0.2, 0.3)):
        w = m & (C >= lo) & (C < hi); sm = np.median(s[w]); a = w & (s < sm); b = w & (s >= sm)
        print(f"  {pl} C in [{lo},{hi}): n={w.sum():3d}, split at s={sm:.2f}: {100 * np.median(e[a]):5.2f} | {100 * np.median(e[b]):5.2f}   ratio {np.median(e[b]) / np.median(e[a]):.2f}   (median C {np.median(C[a]):.3f} | {np.median(C[b]):.3f})")


# ---- second referee round: (1) the (Lambda, kb) grid of phase_map.py alone, (2) random-phase profiles against the law of the cosines with C_rms and with the peak-equivalent slope
print("\n(Lambda, kb) grid of phase_map.py alone (theta_i = 20 deg): log C, log s correlation and fits, 95 % intervals by bootstrap over cells")
rng = np.random.default_rng(1); gcs = np.cos(np.deg2rad(P.TH)) ** 3
for ip, pl in enumerate(P.POLS):
    R_ = np.array([(P.A_L * 0 + np.pi * P.A_L * P.WLr / gcs + kb / (2 * gcs) * (Lam * P.WLr) ** 2, kb * Lam * P.WLr, pm[ip, iL, ik][1]) for iL, Lam in enumerate(P.LAMS) for ik, kb in enumerate(P.KBS)])
    R_ = R_[(R_[:, 0] > 0.03) & np.isfinite(R_[:, 2])]; lc, ls, y = np.log10(R_[:, 0]), np.log10(R_[:, 1]), np.log10(R_[:, 2]); n = len(y)
    def fit_(cols, idx):
        A = np.column_stack([np.ones(n)] + cols)[idx]; b = np.linalg.lstsq(A, y[idx], rcond=None)[0]; return b, (y[idx] - A @ b).std()
    al = np.arange(n); b1, s1 = fit_([lc], al); b2, s2 = fit_([ls], al); b3, s3 = fit_([lc, ls], al)
    bs = np.array([(lambda idx: (fit_([lc], idx)[0][1], fit_([ls], idx)[0][1], *fit_([lc, ls], idx)[0][1:]))(rng.integers(0, n, n)) for _ in range(2000)]); q = np.percentile(bs, [2.5, 97.5], axis=0)
    print(f"  {pl}: n={n}, corr {np.corrcoef(lc, ls)[0, 1]:.2f} | C only {b1[1]:.2f} [{q[0, 0]:.2f},{q[1, 0]:.2f}] scatter {s1:.3f} | s only {b2[1]:.2f} [{q[0, 1]:.2f},{q[1, 1]:.2f}] {s2:.3f} | C^{b3[1]:.2f} [{q[0, 2]:.2f},{q[1, 2]:.2f}] s^{b3[2]:.2f} [{q[0, 3]:.2f},{q[1, 3]:.2f}] {s3:.3f}")

print("\nrandom-phase profiles against the law fitted to the cosines (C > 0.03): median of error / law, scatter of its log10, 10-90 % range")
K_ = 2 * np.pi; k_ = K_ / RS.WLr; c3 = 2 * k_ * np.sqrt(1 - RS.SIN ** 2) ** 3; rows_ = []
for ib, (m1, m2) in enumerate(RS.BANDS.values()):
    ms = np.arange(m1, m2 + 1)
    for ik, ks in enumerate(RS.KSIG):
        Ab = ks / k_ * np.sqrt(2 / len(ms)); curv2 = 0.5 * sum(RS.A_S ** 2 * (m * K_) ** 4 for m in RS.M_S) + 0.5 * (Ab ** 2 * (ms * K_) ** 4).sum()
        sl = ks * np.sqrt((ms ** 2).mean()) * K_ / k_            # rms slope of the small scale; the cosine s is the PEAK slope = sqrt(2) times the rms one
        for ip in range(2):
            for ir in range(rd.shape[3]):
                v = rd[ib, ik, ip, ir]; rows_.append((ip, v[0], v[7], np.sqrt(2 * curv2) / c3, np.sqrt(2) * sl))
rows_ = np.array(rows_)
for ip, pl in enumerate(("TE", "TM")):
    m = ok & (pol == pl); y = np.log10(e[m]); lc = np.log10(C[m]); ls = np.log10(s[m]); r_ = rows_[rows_[:, 0] == ip]; ok_r = np.isfinite(r_[:, 1]) & (r_[:, 1] > 0) & (r_[:, 2] > 0.03)
    for lab, cc, rr in (("C (max curvature)", [lc], [np.log10(r_[:, 2])]), ("C_rms", [lc], [np.log10(r_[:, 3])]), ("C, s", [lc, ls], [np.log10(r_[:, 2]), np.log10(r_[:, 4])])):
        A = np.column_stack([np.ones(m.sum())] + cc); b = np.linalg.lstsq(A, y, rcond=None)[0]; pred = np.column_stack([np.ones(len(r_))] + rr) @ b
        ratio = r_[ok_r, 1] / 10 ** pred[ok_r]
        print(f"  {pl} law in {lab:18s}: median ratio {np.median(ratio):.2f}, scatter {np.log10(ratio).std():.3f}, 10-90 %: {np.percentile(ratio, 10):.2f}-{np.percentile(ratio, 90):.2f}")
