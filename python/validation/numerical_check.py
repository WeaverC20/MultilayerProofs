"""
FESTIM-based bilayer permeation simulation for validating analytical timelag
expressions and demonstrating the single-layer calibration extraction method.

Uses two test cases:
  1. Moderate system (κ~10, ρ~0.1): clean validation of the extraction method
  2. W/SS system: demonstrates conditioning limitations at extreme κ

Material properties for Case 2 from h_transport_materials (htm).

Run with: conda run -n festim2-env python python/validation/numerical_check.py
"""

import numpy as np
from scipy.optimize import minimize_scalar

import h_transport_materials as htm
import festim as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_B_EV = 8.617333262145e-5  # eV/K

# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------

def composite_timelag(ell1, ell2, D1, D2, K):
    """τ = [ℓ₁²ℓ₂/2 + ℓ₂³D₁/(6D₂) + K·ℓ₁³D₂/(6D₁) + K·ℓ₁ℓ₂²/2]
          / [ℓ₂D₁ + K·ℓ₁D₂]"""
    num = (ell1**2 * ell2 / 2
           + ell2**3 * D1 / (6 * D2)
           + K * ell1**3 * D2 / (6 * D1)
           + K * ell1 * ell2**2 / 2)
    den = ell2 * D1 + K * ell1 * D2
    return num / den


def extract_timelag_and_Jss(time, flux, frac_start=0.85, frac_end=0.95):
    """Extract timelag and J_ss from flux(t) via Q(t) asymptote."""
    Q = np.zeros_like(time)
    for i in range(1, len(time)):
        Q[i] = Q[i - 1] + 0.5 * (flux[i] + flux[i - 1]) * (time[i] - time[i - 1])

    mask = (time >= frac_start * time[-1]) & (time <= frac_end * time[-1])
    coeffs = np.polyfit(time[mask], Q[mask], 1)
    J_ss = coeffs[0]
    tau = -coeffs[1] / J_ss
    return tau, J_ss


# ---------------------------------------------------------------------------
# FESTIM simulations (E_D=0, E_S=0 bypass — pass D and S directly)
# ---------------------------------------------------------------------------

def run_single_layer(D, S, thickness, final_time, T=300, n_cells=800):
    """Transient single-layer permeation. Returns (time, flux)."""
    model = F.HydrogenTransportProblem()
    model.mesh = F.Mesh1D(vertices=np.linspace(0, thickness, n_cells + 1))

    # Use a tiny nonzero E to avoid UFL issues, but effectively constant
    mat = F.Material(D_0=float(D), E_D=1e-10, K_S_0=float(S), E_K_S=1e-10,
                     solubility_law="sieverts")
    vol = F.VolumeSubdomain1D(id=1, material=mat, borders=[0, thickness])
    left = F.SurfaceSubdomain1D(id=2, x=0)
    right = F.SurfaceSubdomain1D(id=3, x=thickness)
    model.subdomains = [vol, left, right]

    H = F.Species(name="H", mobile=True, subdomains=[vol])
    model.species = [H]
    model.surface_to_volume = {left: vol, right: vol}
    model.temperature = T

    P_up = 1e5
    model.boundary_conditions = [
        F.SievertsBC(subdomain=left, S_0=float(S), E_S=1e-10,
                      pressure=P_up, species=H),
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


def run_bilayer(D1, S1, ell1, D2, S2, ell2, final_time, T=300,
                n_cells_per_layer=800):
    """Transient bilayer permeation. Returns (time, flux)."""
    model = F.HydrogenTransportProblemDiscontinuous()

    pts1 = np.linspace(0, ell1, n_cells_per_layer + 1)
    pts2 = np.linspace(ell1, ell1 + ell2, n_cells_per_layer + 1)
    model.mesh = F.Mesh1D(vertices=np.concatenate([pts1, pts2[1:]]))

    mat1 = F.Material(D_0=float(D1), E_D=1e-10, K_S_0=float(S1), E_K_S=1e-10,
                      solubility_law="sieverts")
    mat2 = F.Material(D_0=float(D2), E_D=1e-10, K_S_0=float(S2), E_K_S=1e-10,
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
    model.temperature = T

    P_up = 1e5
    model.boundary_conditions = [
        F.SievertsBC(subdomain=left, S_0=float(S1), E_S=1e-10,
                      pressure=P_up, species=H),
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
# Method 1 extraction (least-squares, robust)
# ---------------------------------------------------------------------------

def method1_extract(tau_bi, Jss_bi, D_known, S_known,
                    ell_known, ell_unknown, P_up, known_is_layer1=True):
    """Extract D, S for unknown layer from bilayer τ + J_ss + known layer."""
    Phi_known = D_known * S_known
    R_total = np.sqrt(P_up) / Jss_bi
    R_known = ell_known / Phi_known
    R_unknown = R_total - R_known
    if R_unknown <= 0:
        raise ValueError("Negative unknown resistance — calibration error too large")
    Phi_unknown = ell_unknown / R_unknown

    if known_is_layer1:
        ell1, ell2, D1 = ell_known, ell_unknown, D_known
        def obj(log_D2):
            D2 = np.exp(log_D2)
            K = (Phi_unknown / D2) / S_known
            return (composite_timelag(ell1, ell2, D1, D2, K) - tau_bi)**2
    else:
        ell1, ell2, D2 = ell_unknown, ell_known, D_known
        def obj(log_D1):
            D1 = np.exp(log_D1)
            K = S_known / (Phi_unknown / D1)
            return (composite_timelag(ell1, ell2, D1, D2, K) - tau_bi)**2

    res = minimize_scalar(obj, bounds=(np.log(1e-18), np.log(1e-4)),
                          method="bounded")
    D_ext = np.exp(res.x)
    S_ext = Phi_unknown / D_ext
    return D_ext, S_ext


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    P_up = 1e5

    # ==================================================================
    # CASE 1: Moderate system — clean validation
    # ==================================================================
    # Layer 1 (coating): D=1e-10, S=1e18 → Φ=1e8
    # Layer 2 (substrate): D=1e-9, S=1e19 → Φ=1e10
    # ℓ₁ = ℓ₂ = 1mm
    # K = S2/S1 = 10, κ = 100, ρ = 0.1

    D1 = 1e-10; S1 = 1e18
    D2 = 1e-9;  S2 = 1e19
    ell1 = 1e-3; ell2 = 1e-3
    K = S2 / S1
    kappa = (ell1 / (D1 * S1)) / (ell2 / (D2 * S2))
    rho = D1 * ell2 / (D2 * ell1)

    tau_1_ana = ell1**2 / (6 * D1)
    tau_2_ana = ell2**2 / (6 * D2)
    tau_c_ana = composite_timelag(ell1, ell2, D1, D2, K)
    Phi1 = D1 * S1; Phi2 = D2 * S2
    Jss_c_ana = np.sqrt(P_up) / (ell1 / Phi1 + ell2 / Phi2)

    print("=" * 70)
    print("CASE 1: Moderate system (κ=100, ρ=0.1)")
    print(f"  Layer 1: D={D1:.0e}, S={S1:.0e}, ℓ={ell1*1e3:.0f}mm")
    print(f"  Layer 2: D={D2:.0e}, S={S2:.0e}, ℓ={ell2*1e3:.0f}mm")
    print(f"  K={K:.0f}, κ={kappa:.0f}, ρ={rho:.1f}")
    print(f"  τ_1={tau_1_ana:.1f}s, τ_2={tau_2_ana:.1f}s, "
          f"τ_comp={tau_c_ana:.1f}s, J_ss={Jss_c_ana:.3e}")
    print()

    t_final = 15 * tau_c_ana

    # --- Single-layer substrate (Layer 2) ---
    print("  Running single-layer Layer 2 (substrate)...")
    t_s, f_s = run_single_layer(D2, S2, ell2, 15 * tau_2_ana)
    tau_2_num, Jss_2_num = extract_timelag_and_Jss(t_s, f_s)
    D2_cal = ell2**2 / (6 * tau_2_num)
    S2_cal = (Jss_2_num * ell2 / np.sqrt(P_up)) / D2_cal

    print(f"    τ = {tau_2_num:.2f}s  (ana: {tau_2_ana:.2f}, "
          f"err: {abs(tau_2_num - tau_2_ana)/tau_2_ana*100:.1f}%)")
    print(f"    D = {D2_cal:.3e}  (true: {D2:.3e}, "
          f"err: {abs(D2_cal - D2)/D2*100:.1f}%)")
    print(f"    S = {S2_cal:.3e}  (true: {S2:.3e}, "
          f"err: {abs(S2_cal - S2)/S2*100:.1f}%)")

    # --- Bilayer ---
    print("  Running bilayer...")
    t_b, f_b = run_bilayer(D1, S1, ell1, D2, S2, ell2, t_final)
    tau_c_num, Jss_c_num = extract_timelag_and_Jss(t_b, f_b)

    print(f"    τ = {tau_c_num:.2f}s  (ana: {tau_c_ana:.2f}, "
          f"err: {abs(tau_c_num - tau_c_ana)/tau_c_ana*100:.1f}%)")
    print(f"    J = {Jss_c_num:.3e}  (ana: {Jss_c_ana:.3e}, "
          f"err: {abs(Jss_c_num - Jss_c_ana)/Jss_c_ana*100:.1f}%)")

    # --- Method 1 extraction ---
    print("  Extracting Layer 1 properties from substrate cal + bilayer...")
    D1_ext, S1_ext = method1_extract(
        tau_c_num, Jss_c_num, D2_cal, S2_cal,
        ell2, ell1, P_up, known_is_layer1=False
    )
    print(f"    D1 = {D1_ext:.3e}  (true: {D1:.3e}, "
          f"err: {abs(D1_ext - D1)/D1*100:.1f}%)")
    print(f"    S1 = {S1_ext:.3e}  (true: {S1:.3e}, "
          f"err: {abs(S1_ext - S1)/S1*100:.1f}%)")
    print(f"    Φ1 = {D1_ext * S1_ext:.3e}  (true: {Phi1:.3e}, "
          f"err: {abs(D1_ext*S1_ext - Phi1)/Phi1*100:.1f}%)")
    print()

    # ==================================================================
    # CASE 2: W/SS system from htm
    # ==================================================================
    _w_d = htm.diffusivities.filter(material="tungsten").filter(
        isotope="h", author="frauenfelder")
    _w_s = htm.solubilities.filter(material="tungsten").filter(
        isotope="h", author="frauenfelder")
    _ss_d = htm.diffusivities.filter(material="steel").filter(
        isotope="h", author="esteban")
    _ss_s = htm.solubilities.filter(material="steel").filter(
        isotope="h", author="esteban")

    T_sim = 600.0
    D_W = float(_w_d[0].pre_exp.magnitude) * np.exp(
        -float(_w_d[0].act_energy.magnitude) / (K_B_EV * T_sim))
    S_W = float(_w_s[0].pre_exp.magnitude) * np.exp(
        -float(_w_s[0].act_energy.magnitude) / (K_B_EV * T_sim))
    D_SS = float(_ss_d[0].pre_exp.magnitude) * np.exp(
        -float(_ss_d[0].act_energy.magnitude) / (K_B_EV * T_sim))
    S_SS = float(_ss_s[0].pre_exp.magnitude) * np.exp(
        -float(_ss_s[0].act_energy.magnitude) / (K_B_EV * T_sim))

    ell_W = 0.5e-3; ell_SS = 1e-3
    K_ws = S_SS / S_W
    kappa_ws = (ell_W / (D_W * S_W)) / (ell_SS / (D_SS * S_SS))
    rho_ws = D_W * ell_SS / (D_SS * ell_W)
    tau_ws_ana = composite_timelag(ell_W, ell_SS, D_W, D_SS, K_ws)
    Phi_W = D_W * S_W; Phi_SS = D_SS * S_SS
    Jss_ws_ana = np.sqrt(P_up) / (ell_W / Phi_W + ell_SS / Phi_SS)

    # Analytical sensitivity
    dD = D_W * 1e-6
    tau_pert = composite_timelag(ell_W, ell_SS, D_W + dD, D_SS, K_ws)
    sens = abs((tau_pert - tau_ws_ana) / dD * D_W / tau_ws_ana)

    print("=" * 70)
    print(f"CASE 2: W/SS system at T={T_sim:.0f}K")
    print(f"  W:  D={D_W:.3e}, S={S_W:.3e}, ℓ={ell_W*1e3:.1f}mm")
    print(f"  SS: D={D_SS:.3e}, S={S_SS:.3e}, ℓ={ell_SS*1e3:.0f}mm")
    print(f"  K={K_ws:.1e}, κ={kappa_ws:.1e}, ρ={rho_ws:.2f}")
    print(f"  τ_comp={tau_ws_ana:.1f}s, J_ss={Jss_ws_ana:.3e}")
    print(f"  D_W sensitivity fraction: {sens:.4f} ({sens*100:.2f}%)")
    print(f"  NOTE: κ={kappa_ws:.0e} is extreme — D_W/S_W separation will be")
    print(f"  ill-conditioned. Φ_W extraction should still work.")
    print()

    tau_ss_ana = ell_SS**2 / (6 * D_SS)
    t_final_ws = 15 * max(tau_ws_ana, tau_ss_ana)

    print("  Running single-layer SS...")
    t_s2, f_s2 = run_single_layer(D_SS, S_SS, ell_SS, 15 * tau_ss_ana)
    tau_ss_num, Jss_ss_num = extract_timelag_and_Jss(t_s2, f_s2)
    D_SS_cal = ell_SS**2 / (6 * tau_ss_num)
    S_SS_cal = (Jss_ss_num * ell_SS / np.sqrt(P_up)) / D_SS_cal
    print(f"    D_SS = {D_SS_cal:.3e} (true: {D_SS:.3e}, "
          f"err: {abs(D_SS_cal - D_SS)/D_SS*100:.1f}%)")
    print(f"    S_SS = {S_SS_cal:.3e} (true: {S_SS:.3e}, "
          f"err: {abs(S_SS_cal - S_SS)/S_SS*100:.1f}%)")

    print("  Running bilayer W+SS...")
    t_b2, f_b2 = run_bilayer(D_W, S_W, ell_W, D_SS, S_SS, ell_SS, t_final_ws)
    tau_ws_num, Jss_ws_num = extract_timelag_and_Jss(t_b2, f_b2)
    print(f"    τ = {tau_ws_num:.1f}s (ana: {tau_ws_ana:.1f}, "
          f"err: {abs(tau_ws_num - tau_ws_ana)/tau_ws_ana*100:.1f}%)")
    print(f"    J = {Jss_ws_num:.3e} (ana: {Jss_ws_ana:.3e}, "
          f"err: {abs(Jss_ws_num - Jss_ws_ana)/Jss_ws_ana*100:.1f}%)")

    print("  Extracting W properties...")
    try:
        D_W_ext, S_W_ext = method1_extract(
            tau_ws_num, Jss_ws_num, D_SS_cal, S_SS_cal,
            ell_SS, ell_W, P_up, known_is_layer1=False
        )
        print(f"    Φ_W = {D_W_ext * S_W_ext:.3e} (true: {Phi_W:.3e}, "
              f"err: {abs(D_W_ext*S_W_ext - Phi_W)/Phi_W*100:.1f}%)")
        print(f"    D_W = {D_W_ext:.3e} (true: {D_W:.3e}, "
              f"err: {abs(D_W_ext - D_W)/D_W*100:.1f}%)")
        print(f"    S_W = {S_W_ext:.3e} (true: {S_W:.3e}, "
              f"err: {abs(S_W_ext - S_W)/S_W*100:.1f}%)")
    except Exception as e:
        print(f"    Extraction failed: {e}")

    print()
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
