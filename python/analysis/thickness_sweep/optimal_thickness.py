"""
Extraction error propagation across (κ, ρ) parameter space.

For each point on the (κ, ρ) grid, derives D₁, S₁ from the dimensionless
ratios (given fixed D₂, S₂ and equal thicknesses), then:
  1. Computes exact analytical τ and J_ss
  2. Perturbs τ and J_ss by a measurement uncertainty δ
  3. Runs Method 1 extraction on the perturbed values
  4. Reports the resulting error in D₁, S₁, Φ₁

This shows where in (κ, ρ) space small measurement errors get amplified
into large extraction errors — the same axes as sensitivity_sweep.py.

Run with: conda run -n festim2-env python python/analysis/thickness_sweep/optimal_thickness.py
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Fixed reference parameters
# ---------------------------------------------------------------------------
D2_REF = 1e-9      # m²/s
S2_REF = 1e19      # H/m³/Pa^0.5
ELL = 1e-3          # 1mm for both layers
P_UP = 1e5          # Pa
DELTA = 0.02        # 2% measurement uncertainty in τ and J_ss


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------

def composite_timelag(ell1, ell2, D1, D2, S1, S2):
    K = S2 / S1
    num = (ell1**2 * ell2 / 2
           + ell2**3 * D1 / (6 * D2)
           + K * ell1**3 * D2 / (6 * D1)
           + K * ell1 * ell2**2 / 2)
    den = ell2 * D1 + K * ell1 * D2
    return num / den


def jss_analytical(ell1, ell2, D1, D2, S1, S2, P):
    Phi1 = D1 * S1
    Phi2 = D2 * S2
    return np.sqrt(P) / (ell1 / Phi1 + ell2 / Phi2)


def method1_extract(tau_bi, Jss_bi, D_known, S_known,
                    ell_known, ell_unknown, known_is_layer1=True):
    Phi_known = D_known * S_known
    R_total = np.sqrt(P_UP) / Jss_bi
    R_known = ell_known / Phi_known
    R_unknown = R_total - R_known
    if R_unknown <= 0:
        return np.nan, np.nan
    Phi_unknown = ell_unknown / R_unknown

    if known_is_layer1:
        ell1, ell2, D1 = ell_known, ell_unknown, D_known
        def obj(log_D2):
            D2 = np.exp(log_D2)
            K = (Phi_unknown / D2) / S_known
            num = (ell1**2*ell2/2 + ell2**3*D1/(6*D2) + K*ell1**3*D2/(6*D1)
                   + K*ell1*ell2**2/2)
            den = ell2*D1 + K*ell1*D2
            return (num/den - tau_bi)**2
    else:
        ell1, ell2, D2 = ell_unknown, ell_known, D_known
        def obj(log_D1):
            D1 = np.exp(log_D1)
            K = S_known / (Phi_unknown / D1)
            num = (ell1**2*ell2/2 + ell2**3*D1/(6*D2) + K*ell1**3*D2/(6*D1)
                   + K*ell1*ell2**2/2)
            den = ell2*D1 + K*ell1*D2
            return (num/den - tau_bi)**2

    res = minimize_scalar(obj, bounds=(np.log(1e-20), np.log(1e2)),
                          method="bounded")
    D_ext = np.exp(res.x)
    S_ext = Phi_unknown / D_ext
    return D_ext, S_ext


# ---------------------------------------------------------------------------
# Error propagation sweep
# ---------------------------------------------------------------------------

def compute_error_grid(kappa_vals, rho_vals, delta=DELTA):
    """
    For each (κ, ρ) point, compute worst-case extraction error
    from a δ fractional perturbation in τ and J_ss.

    Returns D₁, S₁, Φ₁ error grids (%).
    """
    n_kap = len(kappa_vals)
    n_rho = len(rho_vals)

    D1_err = np.full((n_rho, n_kap), np.nan)
    S1_err = np.full((n_rho, n_kap), np.nan)
    Phi1_err = np.full((n_rho, n_kap), np.nan)

    for j, kap in enumerate(kappa_vals):
        for i, rho in enumerate(rho_vals):
            K = kap * rho
            D1 = rho * D2_REF
            S1 = S2_REF / K
            Phi1 = D1 * S1

            tau_exact = composite_timelag(ELL, ELL, D1, D2_REF, S1, S2_REF)
            jss_exact = jss_analytical(ELL, ELL, D1, D2_REF, S1, S2_REF, P_UP)

            if tau_exact <= 0 or jss_exact <= 0:
                continue

            # Try all 4 sign combinations of perturbation and take worst case
            worst_D1 = 0.0
            worst_S1 = 0.0
            worst_Phi1 = 0.0

            for s_tau in [1 - delta, 1 + delta]:
                for s_jss in [1 - delta, 1 + delta]:
                    tau_pert = tau_exact * s_tau
                    jss_pert = jss_exact * s_jss

                    D1_ext, S1_ext = method1_extract(
                        tau_pert, jss_pert, D2_REF, S2_REF,
                        ELL, ELL, known_is_layer1=False
                    )

                    if np.isnan(D1_ext) or D1_ext <= 0:
                        continue

                    de = abs(D1_ext - D1) / D1 * 100
                    se = abs(S1_ext - S1) / S1 * 100
                    pe = abs(D1_ext * S1_ext - Phi1) / Phi1 * 100

                    worst_D1 = max(worst_D1, de)
                    worst_S1 = max(worst_S1, se)
                    worst_Phi1 = max(worst_Phi1, pe)

            if worst_D1 > 0:
                D1_err[i, j] = worst_D1
                S1_err[i, j] = worst_S1
                Phi1_err[i, j] = worst_Phi1

    return D1_err, S1_err, Phi1_err


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_error_heatmaps(kappa_vals, rho_vals, D1_err, S1_err, Phi1_err,
                         delta, out_dir="python/analysis/thickness_sweep"):
    KK, RR = np.meshgrid(kappa_vals, rho_vals)

    plot_specs = [
        (D1_err, r"$D_1$ extraction error from %g%% measurement noise (%%)" % (delta*100),
         "D1_error_propagation"),
        (S1_err, r"$S_1$ extraction error from %g%% measurement noise (%%)" % (delta*100),
         "S1_error_propagation"),
        (Phi1_err, r"$\Phi_1$ extraction error from %g%% measurement noise (%%)" % (delta*100),
         "Phi1_error_propagation"),
    ]

    fixed_vmin, fixed_vmax = 0.01, 100
    error_cmap = LinearSegmentedColormap.from_list("error_green", [
        (0.0,  "#006400"),
        (0.50, "#2ca02c"),
        (0.75, "#a8d08d"),
        (0.85, "#ffdd44"),
        (1.0,  "#d62728"),
    ])

    for data, label, fname in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 7))

        valid = np.isfinite(data) & (data > 0)
        if valid.sum() == 0:
            print(f"No valid data for {fname}, skipping")
            plt.close()
            continue

        pcm = ax.pcolormesh(KK, RR, np.clip(data, fixed_vmin, fixed_vmax),
                             norm=LogNorm(vmin=fixed_vmin, vmax=fixed_vmax),
                             cmap=error_cmap, shading="nearest")
        fig.colorbar(pcm, ax=ax, label=label)

        # Error contours
        try:
            cs = ax.contour(KK, RR, data, levels=[5, 10],
                             colors=["white", "white"],
                             linewidths=[1, 2.5],
                             linestyles=[":", "--"])
            ax.clabel(cs, fmt="%g%%", fontsize=10)
        except Exception:
            pass

        # Mark NaN points
        for ii in range(len(rho_vals)):
            for jj in range(len(kappa_vals)):
                if not np.isfinite(data[ii, jj]):
                    ax.plot(kappa_vals[jj], rho_vals[ii], "kx",
                            markersize=8, markeredgewidth=1.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa$ (resistance ratio $R_1/R_2$)", fontsize=12)
        ax.set_ylabel(r"$\rho$ (diffusion speed ratio)", fontsize=12)
        ax.set_title(label, fontsize=11)

        plt.tight_layout()
        path = f"{out_dir}/{fname}.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Same κ, ρ grid as sensitivity_sweep.py
    kappa_vals = np.logspace(-6, 6, 13)
    rho_vals = np.logspace(-4, 4, 9)

    print(f"=== Error Propagation Sweep ===")
    print(f"Measurement noise: δ = {DELTA*100:.0f}%")
    print(f"Reference: D₂ = {D2_REF:.1e}, S₂ = {S2_REF:.1e}")
    print(f"Thicknesses: ℓ₁ = ℓ₂ = {ELL*1e3:.0f} mm")
    print(f"Grid: {len(kappa_vals)} κ × {len(rho_vals)} ρ = "
          f"{len(kappa_vals)*len(rho_vals)} points")
    print()

    D1_err, S1_err, Phi1_err = compute_error_grid(
        kappa_vals, rho_vals, delta=DELTA)

    print("Sample results (worst-case error from 2% noise):")
    for j, kap in enumerate(kappa_vals[::3]):
        jj = list(kappa_vals).index(kap)
        for i, rho in enumerate(rho_vals[::2]):
            ii = list(rho_vals).index(rho)
            d = D1_err[ii, jj]
            s = S1_err[ii, jj]
            if np.isfinite(d):
                print(f"  κ={kap:.0e}, ρ={rho:.0e}: "
                      f"D₁_err={d:.1f}%, S₁_err={s:.1f}%")
    print()

    plot_error_heatmaps(kappa_vals, rho_vals, D1_err, S1_err, Phi1_err,
                         delta=DELTA)
    print("\nDone.")


if __name__ == "__main__":
    main()
