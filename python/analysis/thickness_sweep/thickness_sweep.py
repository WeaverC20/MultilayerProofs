"""
FESTIM-based 2D validation sweep across (ℓ₁, ℓ₂) thickness space
for a tungsten barrier + 316L stainless steel substrate bilayer.

Material properties are imported from h_transport_materials (htm).
For each grid point we:
  1. Run a FESTIM bilayer simulation with known W/SS properties
  2. Extract τ and J_ss from the FESTIM flux curve
  3. Compare to analytical predictions (τ formula, J_ss series resistance)
  4. Run Method 1 extraction (substrate cal + bilayer → coating properties)
  5. Compare extracted D₁, S₁ to the known input values

Dimensionless parameters (include thickness):
  ρ = (D₁ℓ₂)/(D₂ℓ₁)
  κ = (ℓ₁D₂S₂)/(ℓ₂D₁S₁)

Run with: conda run -n festim2-env python python/analysis/thickness_sweep/thickness_sweep.py
"""

import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

import festim as F
import h_transport_materials as htm

# ---------------------------------------------------------------------------
# Material properties from htm
# ---------------------------------------------------------------------------
_D_W_prop = htm.diffusivities.filter(material="tungsten").filter(isotope="h").filter(author="zhou")[0]
_S_W_prop = htm.solubilities.filter(material="tungsten").filter(isotope="h").filter(author="esteban")[0]

_D_SS_prop = htm.diffusivities.filter(material=htm.STEEL_316L, author="reiter")[0]
_S_SS_prop = htm.solubilities.filter(material=htm.STEEL_316L, author="reiter")[0]

# Arrhenius parameters — D = D_0 * exp(-E_D / (k_B * T))
D1_0 = float(_D_W_prop.pre_exp.magnitude)   # m²/s
E_D1 = float(_D_W_prop.act_energy.magnitude) # eV
S1_0 = float(_S_W_prop.pre_exp.magnitude)    # H/m³/Pa^0.5
E_S1 = float(_S_W_prop.act_energy.magnitude) # eV

D2_0 = float(_D_SS_prop.pre_exp.magnitude)
E_D2 = float(_D_SS_prop.act_energy.magnitude)
S2_0 = float(_S_SS_prop.pre_exp.magnitude)
E_S2 = float(_S_SS_prop.act_energy.magnitude)

# ---------------------------------------------------------------------------
# Simulation conditions
# ---------------------------------------------------------------------------
P_UP = 1e5        # Pa
T_SIM = 600.0     # K
K_B_EV = 8.617333262145e-5  # eV/K

# Compute effective D, S at T_SIM
D1_EFF = D1_0 * np.exp(-E_D1 / (K_B_EV * T_SIM))
S1_EFF = S1_0 * np.exp(-E_S1 / (K_B_EV * T_SIM))
D2_EFF = D2_0 * np.exp(-E_D2 / (K_B_EV * T_SIM))
S2_EFF = S2_0 * np.exp(-E_S2 / (K_B_EV * T_SIM))

PHI1_EFF = D1_EFF * S1_EFF
PHI2_EFF = D2_EFF * S2_EFF


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------

def composite_timelag_analytical(ell1, ell2, D1, D2, S1, S2):
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

def run_single_layer(D_0, E_D, S_0, E_S, thickness, final_time, n_cells=500):
    model = F.HydrogenTransportProblem()
    model.mesh = F.Mesh1D(vertices=np.linspace(0, thickness, n_cells + 1))
    mat = F.Material(D_0=D_0, E_D=E_D, K_S_0=S_0, E_K_S=E_S,
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
        F.SievertsBC(subdomain=left, S_0=S_0, E_S=E_S,
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


def run_bilayer(D1_0_loc, E_D1_loc, S1_0_loc, E_S1_loc, ell1,
                D2_0_loc, E_D2_loc, S2_0_loc, E_S2_loc, ell2,
                final_time, n_cells=500):
    model = F.HydrogenTransportProblemDiscontinuous()
    pts1 = np.linspace(0, ell1, n_cells + 1)
    pts2 = np.linspace(ell1, ell1 + ell2, n_cells + 1)
    model.mesh = F.Mesh1D(vertices=np.concatenate([pts1, pts2[1:]]))
    mat1 = F.Material(D_0=D1_0_loc, E_D=E_D1_loc,
                      K_S_0=S1_0_loc, E_K_S=E_S1_loc,
                      solubility_law="sieverts")
    mat2 = F.Material(D_0=D2_0_loc, E_D=E_D2_loc,
                      K_S_0=S2_0_loc, E_K_S=E_S2_loc,
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
        F.SievertsBC(subdomain=left, S_0=S1_0_loc, E_S=E_S1_loc,
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

    res = minimize_scalar(obj, bounds=(np.log(1e-20), np.log(1e2)),
                          method="bounded")
    D_ext = np.exp(res.x)
    S_ext = Phi_unknown / D_ext
    return D_ext, S_ext


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"=== Tungsten / 316L SS Thickness Sweep ===")
    print(f"T = {T_SIM} K,  P_up = {P_UP:.0e} Pa")
    print(f"W  (barrier):   D_0={D1_0:.3e}, E_D={E_D1:.4f} eV, "
          f"S_0={S1_0:.3e}, E_S={E_S1:.4f} eV")
    print(f"SS (substrate): D_0={D2_0:.3e}, E_D={E_D2:.4f} eV, "
          f"S_0={S2_0:.3e}, E_S={E_S2:.4f} eV")
    print(f"At {T_SIM}K: D_W={D1_EFF:.3e}, S_W={S1_EFF:.3e}, "
          f"D_SS={D2_EFF:.3e}, S_SS={S2_EFF:.3e}")
    print()

    # --- Thickness grids ---
    ell1_vals = np.logspace(-6, -1, 11)   # 1μm to 100mm (W coating)
    ell2_vals = np.logspace(-4, -1, 7)    # 0.1mm to 100mm (SS substrate)

    n_ell1 = len(ell1_vals)
    n_ell2 = len(ell2_vals)

    # --- Single-layer substrate calibrations (one per ℓ₂) ---
    D2_cal = np.full(n_ell2, np.nan)
    S2_cal = np.full(n_ell2, np.nan)

    for j, ell2 in enumerate(ell2_vals):
        tau_2_ana = ell2**2 / (6 * D2_EFF)
        t_final = 15 * tau_2_ana
        print(f"Substrate calibration ℓ₂={ell2*1e3:.3f} mm, "
              f"τ_ana={tau_2_ana:.2e} s ...", end="", flush=True)
        t_s, f_s = run_single_layer(D2_0, E_D2, S2_0, E_S2, ell2, t_final)
        tau_2_num, Jss_2_num = extract_timelag_and_Jss(t_s, f_s)
        D2_cal[j] = ell2**2 / (6 * tau_2_num)
        S2_cal[j] = (Jss_2_num * ell2 / np.sqrt(P_UP)) / D2_cal[j]
        print(f" D2_err={abs(D2_cal[j]-D2_EFF)/D2_EFF*100:.2f}%, "
              f"S2_err={abs(S2_cal[j]-S2_EFF)/S2_EFF*100:.2f}%")

    print()

    # --- Result grids ---
    tau_err = np.full((n_ell2, n_ell1), np.nan)
    jss_err = np.full((n_ell2, n_ell1), np.nan)
    D1_err = np.full((n_ell2, n_ell1), np.nan)
    S1_err = np.full((n_ell2, n_ell1), np.nan)
    Phi1_err = np.full((n_ell2, n_ell1), np.nan)
    converged = np.full((n_ell2, n_ell1), False)

    total = n_ell1 * n_ell2
    count = 0

    for j, ell2 in enumerate(ell2_vals):
        for i, ell1 in enumerate(ell1_vals):
            count += 1
            rho = (D1_EFF * ell2) / (D2_EFF * ell1)
            kappa = (ell1 * D2_EFF * S2_EFF) / (ell2 * D1_EFF * S1_EFF)

            tau_ana = composite_timelag_analytical(
                ell1, ell2, D1_EFF, D2_EFF, S1_EFF, S2_EFF)
            jss_ana = jss_analytical(
                ell1, ell2, D1_EFF, D2_EFF, S1_EFF, S2_EFF, P_UP)

            t_final = 15 * tau_ana

            print(f"[{count}/{total}] ℓ₁={ell1*1e3:.3f}mm, ℓ₂={ell2*1e3:.1f}mm, "
                  f"κ={kappa:.1e}, ρ={rho:.1e}, τ_ana={tau_ana:.0f}s",
                  end="", flush=True)

            try:
                t_b, f_b = run_bilayer(
                    D1_0, E_D1, S1_0, E_S1, ell1,
                    D2_0, E_D2, S2_0, E_S2, ell2,
                    t_final)
                tau_fem, jss_fem = extract_timelag_and_Jss(t_b, f_b)

                if np.isnan(tau_fem) or np.isnan(jss_fem) or jss_fem <= 0:
                    print(" → NaN/zero flux")
                    continue

                converged[j, i] = True

                te = abs(tau_fem - tau_ana) / tau_ana * 100
                je = abs(jss_fem - jss_ana) / jss_ana * 100
                tau_err[j, i] = te
                jss_err[j, i] = je

                D1_ext, S1_ext = method1_extract(
                    tau_fem, jss_fem, D2_cal[j], S2_cal[j],
                    ell2, ell1, known_is_layer1=False
                )

                if not np.isnan(D1_ext) and D1_ext > 0:
                    D1_err[j, i] = abs(D1_ext - D1_EFF) / D1_EFF * 100
                    S1_err[j, i] = abs(S1_ext - S1_EFF) / S1_EFF * 100
                    Phi1_err[j, i] = (abs(D1_ext * S1_ext - PHI1_EFF)
                                      / PHI1_EFF * 100)
                    print(f" → τ_err={te:.1f}%, J_err={je:.1f}%, "
                          f"D1={D1_err[j,i]:.1f}%, S1={S1_err[j,i]:.1f}%")
                else:
                    print(f" → τ_err={te:.1f}%, J_err={je:.1f}%, "
                          f"extraction NaN")

            except Exception as e:
                print(f" → FAILED: {e}")

    # --- Plots: κ vs ρ axes (same as sensitivity_sweep.py) ---
    # Compute κ and ρ for each (ℓ₁, ℓ₂) grid point
    ELL1_m, ELL2_m = np.meshgrid(ell1_vals, ell2_vals)
    KAPPA_grid = (ELL1_m * D2_EFF * S2_EFF) / (ELL2_m * D1_EFF * S1_EFF)
    RHO_grid = (D1_EFF * ELL2_m) / (D2_EFF * ELL1_m)

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

    fixed_vmin, fixed_vmax = 0.01, 100
    error_cmap = LinearSegmentedColormap.from_list("error_green", [
        (0.0,  "#006400"),
        (0.50, "#2ca02c"),
        (0.75, "#a8d08d"),
        (0.85, "#ffdd44"),
        (1.0,  "#d62728"),
    ])

    out_dir = "python/analysis/thickness_sweep"

    for data, label, fname, _cmap in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 7))

        valid = np.isfinite(data) & (data > 0)
        if valid.sum() == 0:
            print(f"No valid data for {fname}, skipping plot")
            plt.close()
            continue

        pcm = ax.pcolormesh(KAPPA_grid, RHO_grid,
                             np.clip(data, fixed_vmin, fixed_vmax),
                             norm=LogNorm(vmin=fixed_vmin, vmax=fixed_vmax),
                             cmap=error_cmap, shading="nearest")
        fig.colorbar(pcm, ax=ax, label=label)

        # Error contours
        try:
            cs = ax.contour(KAPPA_grid, RHO_grid, data, levels=[5, 10],
                             colors=["white", "white"],
                             linewidths=[1, 2.5],
                             linestyles=[":", "--"])
            ax.clabel(cs, fmt="%g%%", fontsize=10)
        except Exception:
            pass

        # Mark non-converged
        for jj in range(n_ell2):
            for ii in range(n_ell1):
                if not converged[jj, ii]:
                    ax.plot(KAPPA_grid[jj, ii], RHO_grid[jj, ii], "kx",
                            markersize=10, markeredgewidth=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa$ (resistance ratio $R_1/R_2$)", fontsize=12)
        ax.set_ylabel(r"$\rho$ (diffusion speed ratio)", fontsize=12)
        ax.set_title(label, fontsize=12)

        plt.tight_layout()
        path = f"{out_dir}/{fname}.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()

    # Save calibration results
    import csv
    cal_path = f"{out_dir}/single_layer_calibrations.csv"
    with open(cal_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ell2_m", "ell2_mm", "D2_true", "D2_cal", "D2_err_pct",
                     "S2_true", "S2_cal", "S2_err_pct"])
        for j, ell2 in enumerate(ell2_vals):
            w.writerow([f"{ell2:.6e}", f"{ell2*1e3:.4f}",
                         f"{D2_EFF:.6e}", f"{D2_cal[j]:.6e}",
                         f"{abs(D2_cal[j]-D2_EFF)/D2_EFF*100:.4f}",
                         f"{S2_EFF:.6e}", f"{S2_cal[j]:.6e}",
                         f"{abs(S2_cal[j]-S2_EFF)/S2_EFF*100:.4f}"])
    print(f"Saved: {cal_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
