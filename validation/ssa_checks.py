"""
Checks of the SSA-1 implementation (ssa.py):
  1. flat interface: B+(alpha) = -i Q+_0 R and B-(alpha) = -i Q-_0 T (consistency with a displaced plane);
  2. SPM-1 kernels against the first-order MVB coefficients about a FLAT surface (f = 0, exact zeroth order),
     an independent computation (scattering.mvb_coefficients), TE and TM, several eps, angles and roughness spectra;
  3. small-amplitude limit: SSA-1 vs the Rayleigh solution, error ~ b^2 (log-log slope ~2).
usage: python ssa_checks.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import scattering as S
import ssa

ok = True
print("1. flat-interface consistency |B(alpha) + i Q R| / |Q R|")
for pol in ("TE", "TM"):
    for eps in (2.25, 9.0):
        for th in (0.0, 20.0, 50.0):
            p = S.Problem(L=1.0, wavelength=0.1, theta_i=np.deg2rad(th), eps=eps, pol=pol, rmax=30)
            R, T, ga = ssa.fresnel(p)
            Bp, Bm = ssa.spm1_kernels(p)
            i0 = p.rmax
            e1 = abs(Bp[i0] + 1j * (p.bp[i0] + p.beta) * R) / abs((p.bp[i0] + p.beta) * R)
            e2 = abs(Bm[i0] + 1j * (p.beta - p.bm[i0]) * T) / abs((p.beta - p.bm[i0]) * T)
            ok &= max(e1, e2) < 1e-12
            print(f"   {pol} eps={eps:4.2f} th={th:4.1f}:  {e1:.1e}  {e2:.1e}")

print("2. SPM-1 kernel vs first-order MVB about a flat surface: max relative difference over all orders")
rng = np.random.default_rng(1)
for pol in ("TE", "TM"):
    for eps, th, kind in ((2.25, 20.0, "cos15"), (9.0, 33.0, "cos4"), (1.5, 10.0, "band"), (2.25, 50.0, "band")):
        p = S.Problem(L=1.0, wavelength=0.1, theta_i=np.deg2rad(th), eps=eps, pol=pol, rmax=60)
        if kind.startswith("cos"):
            L_ = int(kind[3:]); psi, psi1 = S.cos_surface(p, 1.0, L_)
        else:
            psi = np.zeros(p.N); psi1 = np.zeros(p.N)
            for m in range(8, 15):
                ph = rng.uniform(0, 2 * np.pi)
                psi += np.cos(m * p.K * p.x + ph); psi1 -= m * p.K * np.sin(m * p.K * p.x + ph)
        f = np.zeros(p.N)
        d0 = S.rayleigh_exact(p, f, f)
        dp, dm = S.mvb_coefficients(p, f, f, psi, psi1, d0, 1)
        hr = np.fft.fft(psi)[p.r % p.N] / p.N
        Bp, Bm = ssa.spm1_kernels(p)
        m = np.abs(hr) > 1e-8
        ep = np.max(np.abs(dp[1][m] - Bp[m] * hr[m]) / np.abs(dp[1][m]))
        em = np.max(np.abs(dm[1][m] - Bm[m] * hr[m]) / np.abs(dm[1][m]))
        ok &= max(ep, em) < 1e-8
        print(f"   {pol} eps={eps:4.2f} th={th:4.1f} {kind:5s}:  reflected {ep:.1e}  transmitted {em:.1e}  ({m.sum()} orders)")

print("3. small amplitude: L1 error of SSA-1 reflected efficiencies vs Rayleigh, cos(15Kx), L=10 lambda")
for pol in ("TE", "TM"):
    p = S.Problem(L=1.0, wavelength=0.1, theta_i=np.arcsin(0.35), eps=2.25, pol=pol, rmax=60)
    kbs = np.array([0.01, 0.02, 0.04, 0.08])
    err = []
    for kb in kbs:
        h, h1 = S.cos_surface(p, kb / p.k, 15)
        ref = S.efficiencies(p, *S.rayleigh_exact(p, h, h1))[0]
        e = S.efficiencies(p, *ssa.ssa1_propagating(p, h))[0]
        err.append(np.abs(e - ref).sum() / ref.sum())
    sl = np.polyfit(np.log(kbs), np.log(err), 1)[0]
    print(f"   {pol}: errors " + " ".join(f"{x:.1e}" for x in err) + f"   slope {sl:.2f}")
print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
