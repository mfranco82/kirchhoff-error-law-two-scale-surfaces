"""
Results figures of the paper, built from the complementarity of the two expansions:
  KA            : expansion in the CURVATURE of the surface (valid for C = max|F''| / (2 k cos^3 theta_i) << 1)
  variation of boundaries about the smooth periodic surface : expansion in the HEIGHT of the small scale (k b << 1)
Both zeroth orders of the series are shown: the Kirchhoff amplitudes of the smooth profile ("KA zeroth order") and the exact (Rayleigh) solution
of the smooth profile ("exact zeroth order").  The baseline is KA applied to the FULL profile (the two-scale KA model); reference = C-method.
  figures/fig_spectra.pdf   (exploratory, not in the paper): spectra of a periodic and a random flat-band surface, deviation per order, convergence with the order n
  figures/fig_Caxis.pdf     (Fig. 2): error of each model versus the KA validity parameter C of the full profile (central figure)
  figures/fig_phasemap.pdf  (exploratory, not in the paper): error in the plane (Lambda, kb)
  numbers                   : prints the numbers quoted in Secs. 4 and 5 of the paper
usage: python make_results_figures.py spectra | caxis | phasemap | numbers
spectra recomputes three cases (~20 s); caxis, phasemap and numbers read the caches in data/ (robustness.npz, random_stats.npz, phase_map.npz),
which are produced by robustness.py, random_stats.py and phase_map.py.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from scattering import *
from cmethod import c_method, spectral_derivative
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{mathpazo}\usepackage{amsmath}",
                     "font.size": 8, "axes.grid": True, "grid.alpha": .3, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
OUT = os.path.join(HERE, "figures") + os.sep; os.makedirs(OUT, exist_ok=True)
# grayscale palette (colour art is charged in print); the models are also told apart by marker and line style
# colour (JQSRT: colour free online); categorical slots in fixed order, markers and line styles kept so the figure also reads in grey
C_KAF, C_KA0, C_EX0, C_REF, C_Z0, C_SM = "#2a78d6", "#1baf7a", "#eda100", "k", "0.72", "0.55"
C_SSA, C_SSA_LS = "#eb6834", (0, (3, 1, 1, 1))
S_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # sequential blue (steps 250-700) for the slope s
WL, SIN, EPS, A_SMOOTH = 0.1, 0.35, 2.25, 0.03          # lambda/L (L = 10 lambda), sin(theta_i), eps, smooth amplitude a/L
def l1(e, er, m): return np.abs(e - er)[m].sum() / er[m].sum()


def case(pol, builder, Lam_max):
    th = np.arcsin(SIN)
    p0 = Problem(1.0, WL, th, EPS, pol, 30, N=4096)
    rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30, 3 * Lam_max + 12)
    p = Problem(1.0, WL, th, EPS, pol, rmax, N=4096)
    f, f1, z, z1 = builder(p)
    F, F1 = f + z, f1 + z1
    ref = efficiencies(p, *[np.nan_to_num(x) for x in c_method(p, F, F1, spectral_derivative(p, F, 2), M=rmax, delta=1e-8)])
    B_ka0, B_ex0, B_kaf = ka_amplitudes_propagating(p, f, f1), rayleigh_exact(p, f, f1), ka_amplitudes(p, F, F1)
    op = SmoothOperator(p, f, f1)
    S = {}
    for key, B0 in (("ka", B_ka0), ("ex", B_ex0)):
        dp, dm = mvb_coefficients(p, f, f1, z, z1, B0, 4, smooth=op)
        S[key] = [efficiencies(p, *mvb_sum(dp, dm, 1, n)) for n in range(5)]
    return dict(p=p, ref=ref, kaf=efficiencies(p, *B_kaf), ka0=efficiencies(p, *B_ka0), ex0=efficiencies(p, *B_ex0), S=S)


def cos_builder(Lam, kb):
    def b(p):
        bb = kb / p.k
        f, f1 = cos_surface(p, A_SMOOTH); z, z1 = cos_surface(p, bb, Lam)
        return f, f1, z, z1
    return b


def random_builder(ib, ik, ir):
    import random_stats as RS
    def b(p):
        rng = np.random.default_rng(10_000 * (ib + 1) + 100 * ik + ir)
        f, f1, z, z1, _ = RS.build(p, list(RS.BANDS)[ib], RS.KSIG[ik], rng)
        return f, f1, z, z1
    return b


def pick_median_surface(ib=1, ik=2):
    """Index of the random surface (band ib, k*sigma index ik, TE) whose errors are closest to the medians of the 40 realisations."""
    import random_stats as RS
    d = np.load(RS.FILE)["data"][ib, ik, 0]
    e = np.log(d[:, [0, 1]]); med = np.median(e, axis=0)
    return int(np.argmin(((e - med) ** 2).sum(axis=1)))


def spectra(save=True):
    LAM, KB = 15, 0.25
    cases = {"cosTE": case("TE", cos_builder(LAM, KB), LAM), "cosTM": case("TM", cos_builder(LAM, KB), LAM)}
    ir = pick_median_surface(); import random_stats as RS
    cases["rndTE"] = case("TE", random_builder(1, 2, ir), max(RS.BANDS["B2: m=8-14"]))
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.2))
    titles = {"cosTE": r"(a) TE, periodic profile", "rndTE": r"(c) TE, random flat-band profile"}
    stats = {}
    for key in ("cosTE", "cosTM", "rndTE"):
        c = cases[key]; p = c["p"]; ref = c["ref"][0]; s = p.prop_p
        stats[key] = dict(ka0=l1(c["ka0"][0], ref, s), kaf=l1(c["kaf"][0], ref, s), ser_ka=[l1(c["S"]["ka"][n][0], ref, s) for n in range(5)],
                          ser_ex=[l1(c["S"]["ex"][n][0], ref, s) for n in range(5)], Ttr=[l1(c["kaf"][1], c["ref"][1], p.prop_m), l1(c["S"]["ka"][3][1], c["ref"][1], p.prop_m), l1(c["S"]["ex"][3][1], c["ref"][1], p.prop_m)])
    from matplotlib.ticker import MaxNLocator
    for ax, key in zip((axs[0, 0], axs[1, 0]), ("cosTE", "rndTE")):
        c = cases[key]; p = c["p"]; s = p.prop_p; r = p.r[s]
        ref = c["ref"][0]
        ax.semilogy(r, np.maximum(c["ka0"][0][s], 1e-14), ":", color=C_Z0, lw=1.0)
        ax.semilogy(r, c["kaf"][0][s], "--", color=C_KAF, lw=1.1)
        ax.semilogy(r, c["S"]["ka"][3][0][s], "-", color=C_KA0, lw=1.2)
        ax.semilogy(r, c["S"]["ex"][3][0][s], "-.", color=C_EX0, lw=1.2)
        ax.semilogy(r, np.maximum(ref[s], 1e-14), "o", color=C_REF, ms=2.6, mfc="none")
        ax.set_title(titles[key]); ax.set_ylim(1e-8, 0.1); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # (b): deviation from the reference order by order, for the periodic TE profile of (a)
    ax = axs[0, 1]; c = cases["cosTE"]; p = c["p"]; s = p.prop_p; r = p.r[s]; ref = c["ref"][0]
    dev = lambda e: np.maximum(np.abs(e[s] / ref[s] - 1), 1e-5)
    ax.semilogy(r, dev(c["kaf"][0]), "--s", color=C_KAF, lw=1.0, ms=2.6)
    ax.semilogy(r, dev(c["S"]["ka"][3][0]), "-o", color=C_KA0, lw=1.1, ms=2.6)
    ax.semilogy(r, dev(c["S"]["ex"][3][0]), "-.^", color=C_EX0, lw=1.1, ms=2.6)
    ax.set_title(r"(b) TE, periodic profile: $|e_r/e_r^{\rm ref}-1|$"); ax.set_ylim(1e-4, 30); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(r"Floquet order $r$")
    for ax in (axs[0, 0], axs[1, 0]):
        ax.set_xlabel(r"Floquet order $r$")
    axs[0, 0].set_ylabel(r"efficiency $e^+_r$"); axs[1, 0].set_ylabel(r"efficiency $e^+_r$")
    ax = axs[1, 1]
    n_ = np.arange(5)
    for key, ls, lab in (("cosTE", "-", "TE"), ("cosTM", "--", "TM")):
        ax.semilogy(n_, 100 * np.array(stats[key]["ser_ka"]), ls, color=C_KA0, marker="o", ms=3, lw=1.1)
        ax.semilogy(n_, 100 * np.array(stats[key]["ser_ex"]), ls, color=C_EX0, marker="^", ms=3, lw=1.1)
        ax.axhline(100 * stats[key]["kaf"], color=C_KAF, ls=ls, lw=0.9)
    ax.set_xlabel(r"perturbative order $n$ (sum up to $n$)"); ax.set_ylabel("relative error (\\%)"); ax.set_xticks(n_)
    ax.set_title(r"(d) convergence: periodic TE (solid), TM (dashed)"); ax.set_ylim(3e-3, 300)
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=C_REF, marker="o", mfc="none", ls="", ms=3.5), Line2D([0], [0], color=C_KAF, ls="--"), Line2D([0], [0], color=C_Z0, ls=":"),
         Line2D([0], [0], color=C_KA0), Line2D([0], [0], color=C_EX0, ls="-.")]
    fig.legend(h, ["reference (C-method)", "KA, full profile", r"zeroth order alone (KA, smooth profile)", r"MVB, $n\leq3$, about KA smooth solution", r"MVB, $n\leq3$, about exact smooth solution"],
               loc="upper center", ncol=3, frameon=False, handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    if save: fig.savefig(OUT + "fig_spectra.pdf")
    plt.close(fig)
    print("random surface used in (c): realisation index", ir)
    for k_, v in stats.items():
        print(f"  {k_}: KA(smooth, zeroth order alone) {100*v['ka0']:.1f}% | KA(full) {100*v['kaf']:.2f}% | series KA d0 n=0..4: {['%.2f' % (100*x) for x in v['ser_ka']]} | exact d0: {['%.3f' % (100*x) for x in v['ser_ex']]} | transmitted L1 (KA full, KA d0 n<=3, exact d0 n<=3): {['%.2f' % (100*x) for x in v['Ttr']]} %")
    return cases, stats


# ------------------------------------------------------------------------------------------------------------------------------------------------
def gather(cond_max=np.inf, keep_all=False, exclude=("sin=0.75", "sin=0.85", "aK=0.38"), return_cluster=False):
    """All cells/surfaces with their KA validity parameter C of the full profile.  Excluded: grazing incidence (sin>=0.75), the configurations in `exclude` and cond(M)>cond_max (default: none).
    keep_all=True returns instead every row considered (finite results) together with the mask of the rows used.
    X[:, 9] is the slope of the small scale k b lambda/lambda_s and X[:, 10] its height k b, X[:, 11] the smooth slope aK (cosines; nan for the random surfaces).  return_cluster=True appends the sweep (configuration of robustness.py,
    'grid20' for phase_map.py, 'random') of every row, for the bootstrap over sweeps."""
    import robustness as R, phase_map as P, random_stats as RS
    clus = []
    rows = []        # (pol, family, C_tot, C_smooth, eKAfull, eKA0 (Kirchhoff zeroth order of the propagating orders; evanescent amplitudes set to zero), eEX0, cond, Rayleigh-vs-C, eSmooth (exact n=0), eKA0 with the evanescent Kirchhoff amplitudes kept)
    d = np.load(R.FILE)["data"]
    for ic, name in enumerate(R.NAMES):
        if name in exclude: continue
        c = R.CONF[name]; Lr = c["L"]; cs = np.sqrt(1 - c["sin"] ** 2) ** 3
        for ip, pol in enumerate(R.POLS):
            for iL, Lam in enumerate(R.LAMS[Lr]):
                for ik, kb in enumerate(R.KBS):
                    v = d[ic, ip, iL, ik]; Csm = np.pi * c["a"] / Lr / cs; Cl = np.pi * kb / (2 * np.pi * Lr) * Lam ** 2 / Lr / cs
                    rows.append((pol, "cos", Csm + Cl, Csm, v[0], v[9], v[2], v[6], v[7], v[8], v[1], kb * Lam / Lr, kb, 2 * np.pi * c["a"])); clus.append(name)
    pm = np.load(P.FILE)["data"]; cs = np.cos(np.deg2rad(P.TH)) ** 3
    for ip, pol in enumerate(P.POLS):
        for iL, Lam in enumerate(P.LAMS):
            for ik, kb in enumerate(P.KBS):
                v = pm[ip, iL, ik]; Csm = np.pi * P.A_L * P.WLr / cs; Cl = np.pi * (kb * P.WLr / (2 * np.pi)) * Lam ** 2 * P.WLr / cs
                rows.append((pol, "cos", Csm + Cl, Csm, v[1], v[7], v[3], v[4], v[5], v[6], v[2], kb * Lam * P.WLr, kb, 2 * np.pi * P.A_L)); clus.append("grid20")
    rd = np.load(RS.FILE)["data"]
    for ib in range(rd.shape[0]):
        for ik in range(rd.shape[1]):
            for ip, pol in enumerate(RS.POLS):
                for ir in range(rd.shape[3]):
                    v = rd[ib, ik, ip, ir]; rows.append((pol, "rnd", v[7], v[8], v[0], v[15], v[2], v[6], v[11], v[14], v[1], np.nan, np.nan, np.nan)); clus.append("random")
    pol = np.array([r[0] for r in rows]); fam = np.array([r[1] for r in rows]); X = np.array([r[2:] for r in rows], float)
    ok = np.isfinite(X[:, [0, 1, 2, 3, 4, 5, 7, 8]]).all(axis=1) & (X[:, 5] < cond_max) & (X[:, 0] > 0)
    clus = np.array(clus)
    if keep_all: return (pol, fam, X, ok) + ((clus,) if return_cluster else ())
    return (pol[ok], fam[ok], X[ok], int((~ok).sum()), len(rows)) + ((clus[ok],) if return_cluster else ())


def caxis_v1(save=True):
    """Fig. 2 as submitted to JOSA A (all models in one panel per polarisation), kept for reference; saves fig_Caxis_v1.pdf."""
    pol, fam, X, n_excl, n_all = gather()
    C, eK, eM, eX, eS = X[:, 0], X[:, 2], X[:, 3], X[:, 4], X[:, 7]
    bins = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0]); mid = np.sqrt(bins[:-1] * bins[1:])
    fig, axs = plt.subplots(1, 2, figsize=(5.25, 3.5), sharey=True)   # printed at the text width of the template (5.25 in): fonts are 1:1
    out = {}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation"))
    import ssa_sweeps                                                      # first-order small-slope approximation, same rows as gather()
    eA = ssa_sweeps.aligned()[3][:, 1]
    for ax, pl in zip(axs, ("TE", "TM")):
        m = pol == pl
        sel_b = [m & (fam == "cos") & (C >= lo) & (C < hi) for lo, hi in zip(bins[:-1], bins[1:])]
        qa = np.array([np.median(100 * eA[s_]) if s_.sum() > 3 else np.nan for s_ in sel_b])
        ax.plot(mid, qa, ls=C_SSA_LS, marker="D", color=C_SSA, lw=1.2, ms=3.2, mec="w", mew=0.4)   # SSA-1, median only
        for e, col, mk, ls, dots in ((eK, C_KAF, "s", "-", True), (eM, C_KA0, "o", "--", False), (eX, C_EX0, "^", ":", True)):
            for fm, size, al in (("cos", 5, 0.22), ("rnd", 4, 0.16)):
                s_ = m & (fam == fm)
                if dots: ax.scatter(C[s_], np.clip(100 * e[s_], 2e-3, 1e4), s=size, marker=mk if fm == "cos" else ".", color=col, alpha=al, lw=0, rasterized=True)
            sel_b = [m & (fam == "cos") & (C >= lo) & (C < hi) for lo, hi in zip(bins[:-1], bins[1:])]
            q = np.array([np.percentile(100 * e[s_], [10, 50, 90]) if s_.sum() > 3 else [np.nan] * 3 for s_ in sel_b])
            ax.plot(mid, q[:, 2], color=col, ls=ls, lw=0.7)                                        # 90th percentile
            ax.plot(mid, q[:, 1], ls + mk, color=col, lw=1.6, ms=4, mec="w", mew=0.4)
        qs = np.array([np.median(100 * eS[m & (fam == "cos") & (C >= lo) & (C < hi)]) if (m & (fam == "cos") & (C >= lo) & (C < hi)).sum() > 3 else np.nan for lo, hi in zip(bins[:-1], bins[1:])])
        ax.plot(mid, qs, "-.", marker="x", color=C_SM, lw=1.3, ms=4)                       # smooth profile alone (roughness ignored)
        sel = m & (fam == "cos") & (C > 0.03)
        k = np.polyfit(np.log(C[sel]), np.log(eK[sel]), 1)
        cc = np.logspace(np.log10(0.03), np.log10(4), 40); ax.plot(cc, 100 * np.exp(k[1]) * cc ** k[0], color="k", ls=(0, (5, 1.5, 1, 1.5)), lw=0.9)
        out[pl] = (np.exp(k[1]), k[0], m.sum())
        ax.axvspan(0.01, 0.05, color="0.88", alpha=0.7, lw=0); ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.01, 4); ax.set_ylim(2e-3, 1e4)
        ax.set_xlabel(r"KA validity parameter $\mathcal{C}$"); ax.set_title(f"({'a' if pl == 'TE' else 'b'}) {pl}", fontsize=8, loc="left")
        ax.text(0.0105, 3e3, "both scales\nKA-valid", fontsize=6.5, va="top")
    axs[0].set_ylabel("relative $L^1$ error (\\%)")
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=C_KAF, marker="s", ms=4, ls="-"), Line2D([0], [0], color=C_KA0, marker="o", ms=4, ls="--"), Line2D([0], [0], color=C_EX0, marker="^", ms=4, ls=":"),
         Line2D([0], [0], color="k", ls=(0, (5, 1.5, 1, 1.5)), lw=0.9), Line2D([0], [0], color=C_SM, marker="x", ms=4, ls="-.", lw=1.3),
         Line2D([0], [0], color=C_SSA, marker="D", ms=3.2, ls=C_SSA_LS, lw=1.2)]
    fig.legend(h, ["KA, full profile", r"series ($n\leq3$) about the KA smooth solution", r"series ($n\leq3$) about the exact smooth solution", r"fit of KA, full profile", r"smooth profile alone (roughness ignored)", r"small-slope approximation (SSA-1)"], loc="upper center", ncol=2, frameon=False, fontsize=6.5, handlelength=1.8, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    if save: fig.savefig(OUT + "fig_Caxis_v1.pdf")
    plt.close(fig)
    print(f"cells/surfaces used: {len(C)} of {n_all} ({n_excl} excluded: none; grazing incidence and aK=0.38 are left out before); per polarisation TE {np.sum(pol=='TE')}, TM {np.sum(pol=='TM')}")
    for pl in ("TE", "TM"): print(f"  KA(full) fit, cosine profiles, {pl}: error = {out[pl][0]:.2f} * C^{out[pl][1]:.2f}")
    print("median error (%) by C bin  [KA full | KA+MVB KA d0 | KA+MVB exact d0 | median gain | fraction series<KA full | n | smooth profile alone (cosines only)]")
    for pl in ("TE", "TM"):
        print(" ", pl)
        for lo, hi in zip(bins[:-1], bins[1:]):
            s_ = (pol == pl) & (C >= lo) & (C < hi)
            if s_.sum() > 3: print(f"    C {lo:5.2f}-{hi:5.2f}: {100*np.median(eK[s_]):7.2f} | {100*np.median(eM[s_]):6.2f} | {100*np.median(eX[s_]):6.2f} | {np.median(eK[s_]/eM[s_]):5.1f} | {100*np.mean(eM[s_]<eK[s_]):3.0f}% | {s_.sum()} | {100*np.median(eS[s_ & (fam == 'cos')]) if (s_ & (fam == 'cos')).sum() else float('nan'):7.2f}")
    return out


def caxis(save=True):
    """Fig. 2.  Top (a, b): every profile, error of KA versus C, grey level = slope s of the small scale (cosines: squares,
    random-phase: dots), with the fit Eq. (28).  Bottom (c, d): medians of the cosine profiles in bins of C for all the models."""
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    pol, fam, X, n_excl, n_all = gather()
    C, eK, eM, eX, eS, s = X[:, 0], X[:, 2], X[:, 3], X[:, 4], X[:, 7], X[:, 9]
    sys.path.insert(0, os.path.join(HERE, "validation"))
    import ssa_sweeps
    eA = ssa_sweeps.aligned()[3][:, 1]
    bins = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0]); mid = np.sqrt(bins[:-1] * bins[1:])
    fig, axs = plt.subplots(2, 2, figsize=(5.25, 5.8), sharex=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("s", S_RAMP)
    norm = LogNorm(0.02, 1.5)
    for j, pl in enumerate(("TE", "TM")):
        m = pol == pl
        ax = axs[0, j]
        r_ = m & (fam == "rnd"); c_ = m & (fam == "cos")
        ax.scatter(C[r_], 100 * eK[r_], s=3, c=s[r_] if np.isfinite(s[r_]).all() else "0.6", marker=".", cmap=cmap, norm=norm, lw=0, rasterized=True) if np.isfinite(s[r_]).any() else \
            ax.scatter(C[r_], 100 * eK[r_], s=4, color="0.75", marker=".", lw=0, rasterized=True)
        sc = ax.scatter(C[c_], 100 * eK[c_], s=7, c=s[c_], marker="s", cmap=cmap, norm=norm, lw=0, rasterized=True)
        sel = c_ & (C > 0.03); k = np.polyfit(np.log(C[sel]), np.log(eK[sel]), 1)
        cc = np.logspace(np.log10(0.03), np.log10(4), 40); ax.plot(cc, 100 * np.exp(k[1]) * cc ** k[0], color="k", ls=(0, (5, 1.5, 1, 1.5)), lw=0.9)
        ax.set_title(f"({'ab'[j]}) KA, full profile, {pl}", fontsize=8, loc="left")
        ax = axs[1, j]
        for e, col, mk, ls, lab in ((eK, C_KAF, "s", "-", "KA"), (eM, C_KA0, "o", "--", "KA0"), (eX, C_EX0, "^", ":", "EX0"),
                                    (eA, C_SSA, "D", C_SSA_LS, "SSA"), (eS, C_SM, "x", "-.", "SM")):
            q = np.array([np.median(100 * e[c_ & (C >= lo) & (C < hi)]) if (c_ & (C >= lo) & (C < hi)).sum() > 3 else np.nan for lo, hi in zip(bins[:-1], bins[1:])])
            ax.plot(mid, q, ls=ls, marker=mk, color=col, lw=1.4 if lab != "SM" else 1.1, ms=4 if mk != "D" else 3.3, mec="w" if mk != "x" else col, mew=0.4)
        ax.set_title(f"({'cd'[j]}) medians of the cosine profiles, {pl}", fontsize=8, loc="left")
        ax.set_xlabel(r"KA validity parameter $\mathcal{C}$")
    for ax in axs.flat:
        ax.axvspan(0.01, 0.05, color="0.9", lw=0, zorder=0); ax.axhline(1, color="k", lw=0.5, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.01, 4)
    for ax in axs[0]: ax.set_ylim(5e-2, 2e3)
    for ax in axs[1]: ax.set_ylim(2e-2, 1e3)
    axs[0, 1].set_yticklabels([]); axs[1, 1].set_yticklabels([])
    axs[0, 0].set_ylabel("error of KA (\\%)"); axs[1, 0].set_ylabel("median error (\\%)")
    axs[0, 0].text(0.011, 1.2e3, "both scales\nKA-valid", fontsize=6.5, va="top")
    fig.subplots_adjust(left=0.1, right=0.87, top=0.96, bottom=0.155, hspace=0.18, wspace=0.06)
    p1 = axs[0, 1].get_position(); cax = fig.add_axes([0.885, p1.y0 + 0.02, 0.016, p1.height - 0.04])
    cb = fig.colorbar(sc, cax=cax); cb.set_label(r"slope $s$ of the small scale", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    h = [Line2D([0], [0], color=C_KAF, marker="s", ms=4, ls="-"), Line2D([0], [0], color=C_SSA, marker="D", ms=3.3, ls=C_SSA_LS),
         Line2D([0], [0], color=C_KA0, marker="o", ms=4, ls="--"), Line2D([0], [0], color=C_EX0, marker="^", ms=4, ls=":"),
         Line2D([0], [0], color=C_SM, marker="x", ms=4, ls="-.", lw=1.1), Line2D([0], [0], color="k", ls=(0, (5, 1.5, 1, 1.5)), lw=0.9)]
    fig.legend(h, ["KA, full profile", "SSA-1, full profile", r"series ($n\leq3$), KA zeroth order", r"series ($n\leq3$), exact zeroth order",
                   "smooth profile alone", r"fit of KA in $\mathcal{C}$ (panels a, b)"], loc="lower center", bbox_to_anchor=(0.48, 0.0), ncol=3, frameon=False, fontsize=6.5, handlelength=2.2, columnspacing=1.2)
    if save: fig.savefig(OUT + "fig_Caxis.pdf", dpi=400)
    plt.close(fig)
    print(f"fig_Caxis.pdf: {len(C)} rows; cosine per polarisation {np.sum((fam=='cos')&(pol=='TE'))}")


def phasemap():
    import phase_map as P
    from matplotlib.colors import LogNorm
    pm = np.load(P.FILE)["data"][0]                       # TE:  [Lam, kb, (KAsmooth, KAfull, MVB_KA, MVB_ex, cond, check)]
    Lams, kbs = np.array(P.LAMS, float), np.array(P.KBS, float)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    cs = np.cos(np.deg2rad(P.TH)) ** 3
    Cfun = lambda Lam, kb: np.pi * P.WLr / cs * (P.A_L + (kb * P.WLr / (2 * np.pi)) * Lam ** 2)
    LL, KK = np.meshgrid(np.geomspace(3, 25, 200), np.geomspace(0.05, 1.0, 200)); CC = Cfun(LL, KK)
    titles = ["(a) KA, full profile", r"(b) MVB about KA smooth solution", r"(c) MVB about exact smooth solution"]
    for ax, j, t in zip(axs, (1, 2, 3), titles):
        val = 100 * pm[:, :, j].T
        im = ax.pcolormesh(Lams, kbs, np.where(val > 0, val, np.nan), shading="nearest", cmap="Greys", norm=LogNorm(0.1, 100), rasterized=True)
        import matplotlib.patheffects as pe
        cs_ = ax.contour(LL, KK, CC, levels=[0.05, 0.4], colors=["k", "k"], linewidths=1.1, linestyles=["-", "--"])
        cs_.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([3, 5, 8, 12, 20]); ax.set_xticklabels(["3", "5", "8", "12", "20"])
        ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"]); ax.minorticks_off()
        ax.set_xlabel(r"$\Lambda$  ($\lambda_s/\lambda=10/\Lambda$)"); ax.set_title(t, fontsize=8, loc="left")
    axs[0].set_ylabel(r"$kb$")
    cb = fig.colorbar(im, ax=axs, pad=0.015, fraction=0.03); cb.set_label("error (\\%)", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    from matplotlib.lines import Line2D
    axs[0].legend([Line2D([0], [0], color="k", lw=1.1), Line2D([0], [0], color="k", lw=1.1, ls="--")], [r"$C=0.05$", r"$C=0.4$"], loc="lower left", fontsize=6.5, framealpha=0.9, handlelength=1.6)
    fig.savefig(OUT + "fig_phasemap.pdf", bbox_inches="tight"); plt.close(fig)
    print("saved fig_phasemap.pdf")


def per_family_tables():
    """Median errors by bin of C, separately for the cosine profiles and the random surfaces (Sec. 4)."""
    pol, fam, X, nex, nall = gather()
    C, eK, eM, eX = X[:, 0], X[:, 2], X[:, 3], X[:, 4]
    bins = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    for fm, name in (("cos", "cosine profiles"), ("rnd", "random flat-band surfaces")):
        print(f"  {name}: median error (%) KA full | series KA zeroth | series exact zeroth | median gain | n")
        for pl in ("TE", "TM"):
            for lo, hi in zip(bins[:-1], bins[1:]):
                s_ = (pol == pl) & (fam == fm) & (C >= lo) & (C < hi)
                if s_.sum() > 3: print(f"    {pl} C {lo:4.2f}-{hi:4.2f}: {100*np.median(eK[s_]):7.2f} | {100*np.median(eM[s_]):6.2f} | {100*np.median(eX[s_]):6.3f} | {np.median(eK[s_]/eM[s_]):5.1f} | {s_.sum()}")
    ok = (C <= 0.8) & np.isfinite(X[:, 6])
    d = X[ok, 6]
    print(f"  Rayleigh vs C-method, {ok.sum()} cases with C<=0.8 (cond<1e15): median {np.median(d):.1e}, 95th percentile {np.percentile(d, 95):.1e}, max {d.max():.1e}; below 1e-4: {100*np.mean(d<1e-4):.0f}%, below 1e-3: {100*np.mean(d<1e-3):.0f}%")


def stability_table():
    """Fraction of cases with series error < 5 % versus the condition number of the smooth-profile matrix (kb <= 0.3, no grazing incidence)."""
    import robustness as R, phase_map as P, random_stats as RS
    rows = []
    d = np.load(R.FILE)["data"]
    for ic, name in enumerate(R.NAMES):
        if name in ("sin=0.75", "sin=0.85"): continue
        for ip in range(2):
            for iL in range(7):
                for ik, kb in enumerate(R.KBS):
                    if kb <= 0.3: rows.append(d[ic, ip, iL, ik][[6, 1, 9, 2]])
    pm = np.load(P.FILE)["data"]
    for ip in range(2):
        for iL in range(len(P.LAMS)):
            for ik, kb in enumerate(P.KBS):
                if kb <= 0.3: rows.append(pm[ip, iL, ik][[4, 2, 7, 3]])
    rd = np.load(RS.FILE)["data"]
    for ib in range(3):
        for ik, ks in enumerate(RS.KSIG):
            if ks <= 0.2:
                for ip in range(2):
                    for ir in range(rd.shape[3]): rows.append(rd[ib, ik, ip, ir][[6, 1, 15, 2]])
    a = np.array(rows, float); a = a[np.isfinite(a).all(axis=1)]
    print(f"  {len(a)} cases; error < 5 % (KA zeroth with the evanescent Kirchhoff amplitudes kept | KA zeroth, propagating orders only | exact zeroth) by condition number of the zeroth-order matrix")
    for lo, hi in zip([1, 1e8, 1e11, 1e14, 1e15, 1e16, 1e18], [1e8, 1e11, 1e14, 1e15, 1e16, 1e18, 1e30]):
        s_ = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if s_.sum(): print(f"    {lo:7.0e}..{hi:7.0e}: n={s_.sum():4d}  {100*np.mean(a[s_, 1] < 0.05):4.0f}% | {100*np.mean(a[s_, 2] < 0.05):4.0f}% | {100*np.mean(a[s_, 3] < 0.05):4.0f}%")
    s_ = a[:, 0] < 1e16
    print(f"    below 1e16: n={s_.sum()}, KA zeroth (kept) {100*np.mean(a[s_, 1] < 0.05):.0f}%, (propagating) {100*np.mean(a[s_, 2] < 0.05):.0f}%, exact zeroth {100*np.mean(a[s_, 3] < 0.05):.0f}%")


def floor_table():
    """Floor of the series with the Kirchhoff zeroth order = error of KA on the smooth profile alone (parametric study, Sec. 4)."""
    import robustness as R
    d = np.load(R.FILE)["data"]
    print("  config, pol: KA vs exact on the smooth profile alone | floor of the series (KA zeroth, kb=0.1) | median exact-zeroth error in the core (kb 0.2-0.3, lambda_s<=1.25 lambda)")
    for ic, name in enumerate(R.NAMES):
        c = R.CONF[name]; Lr = c["L"]; th = np.arcsin(c["sin"])
        for ip, pol in enumerate(R.POLS):
            p0 = Problem(1.0, 1.0 / Lr, th, c["eps"], pol, 30, N=4096)
            rmax = max(int(np.abs(p0.r[p0.prop_m]).max()) + 12, 30)
            p = Problem(1.0, 1.0 / Lr, th, c["eps"], pol, rmax, N=4096); f, f1 = cos_surface(p, c["a"])
            es = l1(efficiencies(p, *ka_amplitudes(p, f, f1))[0], efficiencies(p, *rayleigh_exact(p, f, f1))[0], p.prop_p)
            D = d[ic, ip]; core = [i for i, L_ in enumerate(R.LAMS[Lr]) if Lr / L_ <= 1.25]
            print(f"    {name:>9} {pol}: {100*es:6.2f} % | {100*np.nanmedian(D[:3, 0, 1]):6.2f} % | {100*np.nanmedian(D[np.ix_(core, [1, 2])][..., 2]):6.3f} %")


def statistics():
    """Dispersion, fit, sampling design and control parameters of Fig. 2 (numbers of the abstract, Sec. 4 and the caption of Fig. 2)."""
    pol, fam, X, nex, nall = gather()
    C, eK, eM, eX = X[:, 0], X[:, 2], X[:, 3], X[:, 4]
    rng = np.random.default_rng(0)
    bins = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 4.0])
    print("--- cosine profiles: 10/50/90th percentiles (%) of the error in each bin of C")
    rng_med = {}
    for pl in ("TE", "TM"):
        print(f"  {pl}:  C bin | n | KA full | series, KA zeroth | series, exact zeroth")
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pol == pl) & (fam == "cos") & (C >= lo) & (C < hi)
            if m.sum() > 3:
                q = lambda e: "/".join(f"{100 * v:.2g}" for v in np.percentile(e[m], [10, 50, 90]))
                print(f"     {lo:.2f}-{hi:.2f} | {m.sum():3d} | {q(eK):>15s} | {q(eM):>15s} | {q(eX):>15s}")
                if 0.1 <= lo and hi <= 0.8: rng_med.setdefault(pl, []).append([np.median(e[m]) for e in (eK, eM, eX)] + [np.percentile(e[m], 90) for e in (eK, eM, eX)])
    r = np.array(rng_med["TE"] + rng_med["TM"]) * 100
    print("  over the bins 0.1<C<0.8 (both polarizations): range of the medians  KA full %.1f-%.1f %% | series KA zeroth %.2f-%.2f %% | series exact zeroth %.2f-%.2f %%" % (r[:, 0].min(), r[:, 0].max(), r[:, 1].min(), r[:, 1].max(), r[:, 2].min(), r[:, 2].max()))
    print("                                                  range of the 90th percentiles  series KA zeroth %.1f-%.1f %% | series exact zeroth %.1f-%.1f %%" % (r[:, 4].min(), r[:, 4].max(), r[:, 5].min(), r[:, 5].max()))
    print("--- fit of the KA error of the cosine profiles (least squares in log-log, C>0.03)")
    for pl in ("TE", "TM"):
        sel = (pol == pl) & (fam == "cos") & (C > 0.03); x, y = np.log(C[sel]), np.log(eK[sel]); k = np.polyfit(x, y, 1); n = sel.sum()
        bs = np.array([np.polyfit(x[i], y[i], 1) for i in (rng.integers(0, n, n) for _ in range(2000))])
        print(f"  {pl}: n={n}, error = {np.exp(k[1]):.2f} C^{k[0]:.2f}; bootstrap s.d. of the exponent {bs[:, 0].std():.2f}; rms residual {np.std(y - np.polyval(k, x)) / np.log(10):.2f} decades; "
              f"fit at C=0.3: {100 * np.exp(k[1]) * 0.3 ** k[0]:.1f} %, at C=0.6: {100 * np.exp(k[1]) * 0.6 ** k[0]:.1f} %")
        m = (pol == pl) & (fam == "cos") & (C >= 0.02) & (C < 0.05)
        print(f"     C in [0.02, 0.05): median KA error {100 * np.median(eK[m]):.2f} % against {100 * np.exp(k[1]) * np.sqrt(0.02 * 0.05) ** k[0]:.2f} % of the fit")
        m = (pol == pl) & (fam == "cos") & (C < 0.05); g = eK[m] / eM[m]
        ci = np.percentile([np.median(g[rng.integers(0, m.sum(), m.sum())]) for _ in range(2000)], [2.5, 97.5])
        print(f"     C<0.05 (cosine, n={m.sum()}): median gain {np.median(g):.2f} (95 % interval {ci[0]:.2f}-{ci[1]:.2f}); series better than KA in {100 * np.mean(eM[m] < eK[m]):.0f} % of the cases")
    print("--- random surfaces: C from %.3f to %.3f" % (C[fam == "rnd"].min(), C[fam == "rnd"].max()))
    P_, F_, X_, ok = gather(keep_all=True); bad = ok & (X_[:, 5] >= 1e15)
    print("--- sampling design: %d cosine cases (%d per polarization: one parameter at a time around the base case, and the plane (Lambda, kb) at 20 deg) and %d random surfaces; %d of %d used" % ((F_ == "cos").sum(), (F_ == "cos").sum() // 2, (F_ == "rnd").sum(), len(C), nall))
    print(f"--- cond>1e15 (included): {bad.sum()} cases ({(F_[bad] == 'cos').sum()} cosine, {(F_[bad] == 'rnd').sum()} random); median error and fraction below 5 %:")
    for lab, j in (("KA full", 2), ("smooth profile alone", 7), ("series, KA zeroth (evanescent kept)", 8), ("series, KA zeroth (propagating)", 3), ("series, exact zeroth", 4)):
        e = X_[bad, j]; print(f"     {lab:22s}: median {100 * np.median(e):6.1f} %, below 5 %: {100 * np.mean(e < 0.05):.0f} %")
    # which parameter controls the error of the series: kb, or the slope of the small scale kb*Lambda/(L/lambda)?
    import robustness as R, phase_map as P
    rows = []; dat = np.load(R.FILE)["data"]
    for ic, name in enumerate(R.NAMES):
        if name in ("sin=0.75", "sin=0.85", "aK=0.38"): continue
        c = R.CONF[name]
        for ip in range(2):
            for iL, Lm in enumerate(R.LAMS[c["L"]]):
                for ik, kb in enumerate(R.KBS): rows.append((ip, c["L"], Lm, kb) + tuple(dat[ic, ip, iL, ik][[9, 2, 6]]))
    pm = np.load(P.FILE)["data"]
    for ip in range(2):
        for iL, Lm in enumerate(P.LAMS):
            for ik, kb in enumerate(P.KBS): rows.append((ip, 1 / P.WLr, Lm, kb) + tuple(pm[ip, iL, ik][[7, 3, 4]]))
    A = np.array(rows, float); A = A[np.isfinite(A).all(axis=1) & (A[:, 6] < 1e15)]      # cond<1e15: the regime where the Rayleigh matrix is not the limiting factor
    ip_, Lr, Lm, kb, e_ka, e_ex = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4], A[:, 5]; slope = kb * Lm / Lr
    print("--- cosine profiles, kb>=0.2: least squares of log10(error) on log10(kb) and log10(small-scale slope kb*Lambda*lambda/L)")
    for lab, e, floor in (("exact zeroth", e_ex, 3e-4), ("KA zeroth", e_ka, 3e-3)):
        for ipol, pl in enumerate(("TE", "TM")):
            m = (ip_ == ipol) & (kb >= 0.2) & (e > floor); y = np.log10(e[m]); out = []
            for cols in ([np.log10(kb[m])], [np.log10(slope[m])], [np.log10(kb[m]), np.log10(slope[m])]):
                M_ = np.column_stack([np.ones(m.sum())] + cols); co = np.linalg.lstsq(M_, y, rcond=None)[0]; out.append((co[1:], np.std(y - M_ @ co)))
            print(f"     {lab:12s} {pl}: n={m.sum()}  kb only: exponent {out[0][0][0]:.1f}, residual {out[0][1]:.2f} | slope only: {out[1][0][0]:.1f}, {out[1][1]:.2f} | both: kb^{out[2][0][0]:.1f} slope^{out[2][0][1]:.1f}, {out[2][1]:.2f} (decades)")
    print("--- median error (%) of the series by small-scale slope, kb<=0.5 (exact | KA zeroth; TE, TM together)")
    for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.3)):
        m = (kb <= 0.5) & (slope >= lo) & (slope < hi)
        if m.sum(): print(f"     slope {lo:.1f}-{hi:.1f}: n={m.sum():3d}  median {100 * np.median(e_ex[m]):.2f} | {100 * np.median(e_ka[m]):.2f}   90th percentile {100 * np.percentile(e_ex[m], 90):.1f} | {100 * np.percentile(e_ka[m], 90):.1f}")
    # local versus global validity parameter, base-case profile
    k = 2 * np.pi / WL; K = 2 * np.pi; b = 0.25 / k; x = np.linspace(0, 1, 200001)
    F1 = -A_SMOOTH * K * np.sin(K * x) - b * 15 * K * np.sin(15 * K * x); F2 = -A_SMOOTH * K ** 2 * np.cos(K * x) - b * (15 * K) ** 2 * np.cos(15 * K * x)
    ci_ = np.sqrt(1 - SIN ** 2); den = ci_ + SIN * F1; il = den > 0
    Cg = np.abs(F2).max() / (2 * k * ci_ ** 3); Cl = np.max(np.abs(F2[il]) / (2 * k * den[il] ** 3))
    print(f"--- base-case profile: C of Eq. (10) {Cg:.3f}; maximum of the local value (curvature / (2 k cos^3 theta_l)) over the illuminated points {Cl:.3f} (x{Cl / Cg:.2f}); maximum slope {np.abs(F1).max():.2f}")


def numbers():
    """Print the numbers quoted in Secs. 4 and 5 of the paper."""
    print("=== base case (periodic profile Lambda=15, kb=0.25; median random surface)")
    cases, stats = spectra(save=False)
    c = cases["cosTE"]; p = c["p"]; s = p.prop_p; r = p.r[s]; ref = c["ref"][0][s]
    dev = lambda e: np.abs(e[s] / ref - 1); big = ref > 1e-4
    for name, e in (("KA full", c["kaf"][0]), ("series, KA zeroth", c["S"]["ka"][3][0]), ("series, exact zeroth", c["S"]["ex"][3][0])):
        dd = dev(e); print(f"  per-order deviation, {name}: median {100*np.median(dd[big]):.1f} %, max {100*dd[big].max():.1f} % (e>1e-4), max overall {100*dd.max():.1f} % at r={r[np.argmax(dd)]}")
    dk = dev(c["kaf"][0]); print(f"  KA full: {100*dk[r >= -8].min():.0f}-{100*dk[r >= -8].max():.0f} % for r>=-8; model/reference at r=-13: {c['kaf'][0][s][r == -13][0] / ref[r == -13][0]:.2f}")
    cs3 = np.cos(np.arcsin(SIN)) ** 3; Cs = np.pi * WL * A_SMOOTH / cs3
    print(f"  C_smooth = {Cs:.3f}, C_small = kb (lambda/lambda_s)^2 / (2 cos^3 theta) = {0.25 * (15 * WL) ** 2 / (2 * cs3):.3f}")
    print("\n=== Fig. 2 (error versus C)")
    caxis(save=False); per_family_tables()
    print("\n=== plane Lambda-kb, TE and TM (exploratory figure, not in the paper)")
    import phase_map as P
    pm = np.load(P.FILE)["data"]
    for ip, pol in enumerate(P.POLS):
        v = pm[ip]; ka, mk, mx = 100 * v[:, :, 1], 100 * v[:, :, 2], 100 * v[:, :, 3]
        g = (ka / mk)[3:, 2:5]
        print(f"  {pol}: gain KA full / series (KA zeroth), kb=0.2-0.3, Lambda>=10: min {g.min():.1f}, median {np.median(g):.1f}, max {g.max():.1f}")
        for ik, kb in enumerate(P.KBS):
            print(f"      kb={kb:4.2f}: series KA zeroth {mk[:, ik].min():6.2f}-{mk[:, ik].max():6.2f} %, exact zeroth {mx[:, ik].min():6.3f}-{mx[:, ik].max():6.2f} %, KA full {ka[:, ik].min():6.2f}-{ka[:, ik].max():6.1f} %")
    print("\n=== floor of the Kirchhoff zeroth order and incidence angle (parametric study)")
    floor_table()
    print("\n=== stability: condition number")
    stability_table()
    print("\n=== dispersion, fit, sampling, control parameters (Fig. 2)")
    statistics()


if __name__ == "__main__":
    cmds = {"spectra": spectra, "caxis": caxis, "caxis_v1": caxis_v1, "phasemap": phasemap, "numbers": numbers}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds: sys.exit("usage: python make_results_figures.py " + " | ".join(cmds))
    cmds[sys.argv[1]]()
