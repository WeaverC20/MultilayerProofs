"""
FESTIM-based 2D validation sweep across (κ, ρ) parameter space.

FESTIM is the ground truth. For each grid point, we:
  1. Run a FESTIM bilayer simulation with known D₁, D₂, S₁, S₂
  2. Extract τ and J_ss from the FESTIM flux curve
  3. Compare to analytical predictions (τ formula, J_ss series resistance)
  4. Run Method 1 extraction (substrate cal + bilayer → coating properties)
  5. Compare extracted D₁, S₁ to the known input values

Produces heatmaps showing where the analytical formulas and extraction work.

Run with: conda run -n festim2-env python python/analysis/sensitivity_sweep.py
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import festim as F

# ---------------------------------------------------------------------------
# Fixed reference substrate
# ---------------------------------------------------------------------------
D2_REF = 1e-9     # m²/s (desired value at T_SIM)
S2_REF = 1e19     # H/m³/Pa^0.5 (desired value at T_SIM)
ELL = 1e-3         # 1mm for both layers
P_UP = 1e5         # Pa
T_SIM = 600.0      # K
E_DUMMY = 0.1      # eV — non-trivial Arrhenius energy for FESTIM stability
K_B_EV = 8.617333262145e-5

# Pre-compute Arrhenius factor: D = D_0 * exp(-E/(k_B*T))
_ARR_FACTOR = np.exp(-E_DUMMY / (K_B_EV * T_SIM))


def to_arrhenius(D_target):
    """Convert desired D (or S) at T_SIM to D_0 for FESTIM Arrhenius."""
    return D_target / _ARR_FACTOR


# ---------------------------------------------------------------------------
# Analytical formulas (to compare against FESTIM truth)
# ---------------------------------------------------------------------------

def composite_timelag_analytical(ell1, ell2, D1, D2, S1, S2):
    """Physical composite timelag from the derivation."""
    K = S2 / S1
    num = (ell1**2 * ell2 / 2
           + ell2**3 * D1 / (6 * D2)
           + K * ell1**3 * D2 / (6 * D1)
           + K * ell1 * ell2**2 / 2)
    den = ell2 * D1 + K * ell1 * D2
    return num / den


def jss_analytical(ell1, ell2, D1, D2, S1, S2, P):
    """Steady-state flux from series resistance."""
    Phi1 = D1 * S1
    Phi2 = D2 * S2
    return np.sqrt(P) / (ell1 / Phi1 + ell2 / Phi2)


def extract_timelag_and_Jss(time, flux, frac_start=0.85, frac_end=0.95):
    Q = np.zeros_like(time)
    for i in range(1, len(time)):
        Q[i] = Q[i - 1] + 0.5 * (flux[i] + flux[i - 1]) * (time[i] - time[i - 1])
    mask = (time >= frac_start * time[-1]) & (time <= frac_end * time[-1])
    if mask.sum() < 3:
        return np.nan, np.nan
    coeffs = np.polyfit(time[mask], Q[mask], 1)
    J_ss = coeffs[0]
    if J_ss <= 0:
        return np.nan, np.nan
    tau = -coeffs[1] / J_ss
    return tau, J_ss


# ---------------------------------------------------------------------------
# FESTIM simulations
# ---------------------------------------------------------------------------

def run_single_layer(D, S, thickness, final_time, n_cells=500):
    """Run single-layer. D and S are desired values at T_SIM."""
    D_0 = float(to_arrhenius(D))
    S_0 = float(to_arrhenius(S))

    model = F.HydrogenTransportProblem()
    model.mesh = F.Mesh1D(vertices=np.linspace(0, thickness, n_cells + 1))
    mat = F.Material(D_0=D_0, E_D=E_DUMMY, K_S_0=S_0, E_K_S=E_DUMMY,
                     solubility_law="sieverts")
    vol = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, thickness])
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=thickness)
    model.subdomains = [vol, left, right]
    H = F.Species(name="H", mobile=True, subdomains=[vol])
    model.species = [H]
    model.surface_to_volume = {left: vol, right: vol}
    model.temperature = T_SIM
    model.boundary_conditions = [
        F.SievertsBC(subdomain=left, S_0=S_0, E_S=E_DUMMY,
                      pressure=P_UP, species=H),
        F.FixedConcentrationBC(subdomain=right, value=0, species=H),
    ]
    flux_right = F.SurfaceFlux(field=H, surface=right)
    model.exports = [flux_right]
    model.settings = F.Settings(
        atol=1e8, rtol=1e-10, transient=True, final_time=final_time,
        stepsize=F.Stepsize(initial_value=final_time / 2000,
                            growth_factor=1.05, cutback_factor=0.5,
                            target_nb_iterations=4,
                            max_stepsize=final_time / 200),
    )
    model.initialise()
    model.run()
    return np.array(flux_right.t), np.abs(np.array(flux_right.data))


def run_bilayer(D1, S1, ell1, D2, S2, ell2, final_time, n_cells=500):
    """Run bilayer. D1, S1, D2, S2 are desired values at T_SIM."""
    D1_0 = float(to_arrhenius(D1))
    S1_0 = float(to_arrhenius(S1))
    D2_0 = float(to_arrhenius(D2))
    S2_0 = float(to_arrhenius(S2))

    model = F.HydrogenTransportProblemDiscontinuous()
    pts1 = np.linspace(0, ell1, n_cells + 1)
    pts2 = np.linspace(ell1, ell1 + ell2, n_cells + 1)
    model.mesh = F.Mesh1D(vertices=np.concatenate([pts1, pts2[1:]]))
    mat1 = F.Material(D_0=D1_0, E_D=E_DUMMY, K_S_0=S1_0, E_K_S=E_DUMMY,
                      solubility_law="sieverts")
    mat2 = F.Material(D_0=D2_0, E_D=E_DUMMY, K_S_0=S2_0, E_K_S=E_DUMMY,
                      solubility_law="sieverts")
    barrier = F.VolumeSubdomain1D(id=1, material=mat1, borders=[0, ell1])
    substrate = F.VolumeSubdomain1D(id=2, material=mat2,
                                     borders=[ell1, ell1 + ell2])
    left = F.SurfaceSubdomain1D(id=3, x=0)
    right = F.SurfaceSubdomain1D(id=4, x=ell1 + ell2)
    interface = F.Interface(id=5, subdomains=[barrier, substrate],
                            penalty_term=1e17)
    model.interfaces = [interface]
    model.subdomains = [barrier, substrate, left, right]
    H = F.Species(name="H", mobile=True, subdomains=[barrier, substrate])
    model.species = [H]
    model.surface_to_volume = {left: barrier, right: substrate}
    model.temperature = T_SIM
    model.boundary_conditions = [
        F.SievertsBC(subdomain=left, S_0=S1_0, E_S=E_DUMMY,
                      pressure=P_UP, species=H),
        F.FixedConcentrationBC(subdomain=right, value=0, species=H),
    ]
    flux_right = F.SurfaceFlux(field=H, surface=right)
    model.exports = [flux_right]

    model.settings = F.Settings(
        atol=1e8, rtol=1e-10, transient=True, final_time=final_time,
        stepsize=F.Stepsize(initial_value=final_time / 2000,
                            growth_factor=1.05, cutback_factor=0.5,
                            target_nb_iterations=4,
                            max_stepsize=final_time / 200),
    )
    model.initialise()
    model.run()
    return np.array(flux_right.t), np.abs(np.array(flux_right.data))


# ---------------------------------------------------------------------------
# Method 1 extraction
# ---------------------------------------------------------------------------

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

    res = minimize_scalar(obj, bounds=(np.log(1e-20), np.log(1e0)),
                          method="bounded")
    D_ext = np.exp(res.x)
    S_ext = Phi_unknown / D_ext
    return D_ext, S_ext


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Phi2 = D2_REF * S2_REF

    # --- Single-layer substrate calibration (run once) ---
    tau_2_ana = ELL**2 / (6 * D2_REF)
    print("Running single-layer substrate calibration...")
    t_s, f_s = run_single_layer(D2_REF, S2_REF, ELL, 15 * tau_2_ana)
    tau_2_num, Jss_2_num = extract_timelag_and_Jss(t_s, f_s)
    D2_cal = ELL**2 / (6 * tau_2_num)
    S2_cal = (Jss_2_num * ELL / np.sqrt(P_UP)) / D2_cal
    print(f"  D2 = {D2_cal:.3e} (true: {D2_REF:.3e}, "
          f"err: {abs(D2_cal - D2_REF)/D2_REF*100:.1f}%)")
    print(f"  S2 = {S2_cal:.3e} (true: {S2_REF:.3e}, "
          f"err: {abs(S2_cal - S2_REF)/S2_REF*100:.1f}%)")
    print()

    # --- 2D grid ---
    kappa_vals = np.logspace(-1, 3, 7)
    rho_vals = np.logspace(-1, 1, 7)

    n_kap = len(kappa_vals)
    n_rho = len(rho_vals)

    # Result grids
    tau_err = np.full((n_rho, n_kap), np.nan)    # analytical τ vs FESTIM τ
    jss_err = np.full((n_rho, n_kap), np.nan)    # analytical J vs FESTIM J
    D1_err = np.full((n_rho, n_kap), np.nan)     # Method 1 D₁ vs true D₁
    S1_err = np.full((n_rho, n_kap), np.nan)     # Method 1 S₁ vs true S₁
    Phi1_err = np.full((n_rho, n_kap), np.nan)   # Method 1 Φ₁ vs true Φ₁
    converged = np.full((n_rho, n_kap), False)

    total = n_kap * n_rho
    count = 0

    for j, kap in enumerate(kappa_vals):
        for i, rho in enumerate(rho_vals):
            count += 1
            K = kap * rho

            # Derive Layer 1 properties
            D1 = rho * D2_REF
            S1 = S2_REF / K
            Phi1 = D1 * S1

            tau_ana = composite_timelag_analytical(ELL, ELL, D1, D2_REF,
                                                    S1, S2_REF)
            jss_ana = jss_analytical(ELL, ELL, D1, D2_REF, S1, S2_REF, P_UP)

            t_final = 15 * tau_ana

            print(f"[{count}/{total}] κ={kap:.1e}, ρ={rho:.1e}, "
                  f"K={K:.1e}, D1={D1:.1e}, S1={S1:.1e}, "
                  f"τ_ana={tau_ana:.0f}s", end="", flush=True)

            try:
                t_b, f_b = run_bilayer(D1, S1, ELL, D2_REF, S2_REF, ELL,
                                        t_final)
                tau_fem, jss_fem = extract_timelag_and_Jss(t_b, f_b)

                if np.isnan(tau_fem) or np.isnan(jss_fem) or jss_fem <= 0:
                    print(" → NaN/zero flux")
                    continue

                converged[i, j] = True

                # Analytical vs FESTIM
                te = abs(tau_fem - tau_ana) / tau_ana * 100
                je = abs(jss_fem - jss_ana) / jss_ana * 100
                tau_err[i, j] = te
                jss_err[i, j] = je

                # Method 1 extraction
                D1_ext, S1_ext = method1_extract(
                    tau_fem, jss_fem, D2_cal, S2_cal,
                    ELL, ELL, known_is_layer1=False
                )

                if not np.isnan(D1_ext) and D1_ext > 0:
                    D1_err[i, j] = abs(D1_ext - D1) / D1 * 100
                    S1_err[i, j] = abs(S1_ext - S1) / S1 * 100
                    Phi1_err[i, j] = abs(D1_ext * S1_ext - Phi1) / Phi1 * 100

                    print(f" → τ_err={te:.1f}%, J_err={je:.1f}%, "
                          f"D1={D1_err[i,j]:.1f}%, S1={S1_err[i,j]:.1f}%")
                else:
                    print(f" → τ_err={te:.1f}%, J_err={je:.1f}%, "
                          f"extraction NaN")

            except Exception as e:
                print(f" → FAILED: {e}")

    # --- Plots ---
    KK, RR = np.meshgrid(kappa_vals, rho_vals)

    plot_specs = [
        (tau_err, r"$\tau$ error: analytical vs FESTIM (%)",
         "tau_error_2D", "RdYlGn_r"),
        (jss_err, r"$J_{ss}$ error: analytical vs FESTIM (%)",
         "Jss_error_2D", "RdYlGn_r"),
        (D1_err, r"$D_1$ extraction error: Method 1 vs true (%)",
         "D1_error_2D", "RdYlGn_r"),
        (S1_err, r"$S_1$ extraction error: Method 1 vs true (%)",
         "S1_error_2D", "RdYlGn_r"),
        (Phi1_err, r"$\Phi_1$ extraction error: Method 1 vs true (%)",
         "Phi1_error_2D", "RdYlGn_r"),
    ]

    for data, label, fname, cmap in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 7))

        valid = np.isfinite(data) & (data > 0)
        if valid.sum() == 0:
            print(f"No valid data for {fname}, skipping plot")
            plt.close()
            continue

        vmin = max(data[valid].min(), 0.01)
        vmax = min(data[valid].max(), 1000)

        pcm = ax.pcolormesh(KK, RR, np.clip(data, vmin, vmax),
                             norm=LogNorm(vmin=vmin, vmax=vmax),
                             cmap=cmap, shading="nearest")
        fig.colorbar(pcm, ax=ax, label=label)

        # 10% contour
        try:
            cs = ax.contour(KK, RR, data, levels=[5, 10],
                             colors=["white", "white"],
                             linewidths=[1, 2.5],
                             linestyles=[":", "--"])
            ax.clabel(cs, fmt="%g%%", fontsize=10)
        except Exception:
            pass

        # Mark non-converged points
        for ii in range(n_rho):
            for jj in range(n_kap):
                if not converged[ii, jj]:
                    ax.plot(kappa_vals[jj], rho_vals[ii], "kx",
                            markersize=10, markeredgewidth=2)

        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa$ (resistance ratio $R_1/R_2$)", fontsize=12)
        ax.set_ylabel(r"$\rho$ (diffusion speed ratio)", fontsize=12)
        ax.set_title(label, fontsize=12)

        plt.tight_layout()
        path = f"python/analysis/{fname}.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
