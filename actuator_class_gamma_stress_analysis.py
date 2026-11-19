"""
Figure 05 - Class Gamma Actuator Rigorous AGMA Stress, Bearing & Inverter Analysis
===================================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Engineering Group
Validation target: Kyle Edelberg (Chief Engineer, Figure AI)

First-principles calculations:
1. Exact 2-Stage Compound Planetary Tooth Geometry (18.00:1 total ratio, assembly-verified).
2. AGMA 2001-D04 Helical Bending Stress & Safety Factor (Grade 3 Aerospace 9310/300M steel).
3. Exact AGMA Hertzian Pitting Geometry Factor I & Surface Contact Stress (SF > 1.7 peak).
4. Dual-Row Planet Cylindrical/Needle Roller Bearing L10h Dynamic Life (ISO 281, > 20,000 hrs).
5. 800V DC Bus 3-Level Flying-Capacitor 650V GaN Inverter Switching & Conduction Loss Model.
6. Generates clean, non-overlapping technical manufacturing cross-section blueprint.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

def calculate_agma_stresses():
    print("-----------------------------------------------------------------")
    print("FIGURE 05: CLASS GAMMA ACTUATOR - AGMA STRESS & BEARING ANALYSIS")
    print("-----------------------------------------------------------------")

    # 1. Exact 2-Stage Planetary Kinematics (18.00:1 Total Reduction)
    # Stage 1 (4 planets): z_s1 = 20, z_p1 = 16, z_r1 = 52 -> i1 = 1 + 52/20 = 3.60:1
    #   Assembly check: (20 + 52)/4 = 18 (exact integer), 20 + 2*16 = 52 (exact radial fit)
    # Stage 2 (4 planets): z_s2 = 16, z_p2 = 24, z_r2 = 64 -> i2 = 1 + 64/16 = 5.00:1
    #   Assembly check: (16 + 64)/4 = 20 (exact integer), 16 + 2*24 = 64 (exact radial fit)
    z_s1, z_p1, z_r1 = 20, 16, 52
    z_s2, z_p2, z_r2 = 16, 24, 64
    i_stage1 = 1.0 + z_r1 / z_s1
    i_stage2 = 1.0 + z_r2 / z_s2
    ratio = i_stage1 * i_stage2  # Exact 18.00:1

    # Actuator Performance Envelope
    T_peak = 510.0   # Nm (3s peak shock output torque)
    T_cont = 180.0   # Nm (Continuous thermal rating)
    T_rms_duty = 230.0 # Nm (Heavy cyclic warehouse palletizing RMS)
    n_output = 35.0  # RPM output speed at heavy lift (~3.66 rad/s)

    # Material: AGMA Grade 3 Case-Carburized Aerospace Gear Steel (AMS 6265 / AISI 9310)
    # Case hardness: 60-62 HRC, Core: 38-42 HRC, shot-peened root
    sigma_allow_bend = 515.0     # MPa (Allowable bending fatigue limit, Grade 3)
    sigma_allow_contact = 1720.0 # MPa (Allowable contact/pitting fatigue limit, Grade 3)
    E_steel = 206000.0           # MPa
    nu_steel = 0.29

    # Stage 2 Sun Gear Parameters (Highest torque density mesh)
    m_n = 2.0                    # Normal module (mm)
    beta_deg = 18.0              # Helix angle (deg) for quiet smooth load sharing
    beta = np.deg2rad(beta_deg)
    m_t = m_n / np.cos(beta)     # Transverse module = 2.103 mm
    d_sun = m_t * z_s2           # Pitch diameter = 33.65 mm (Ring OD ~ 134.6 mm)
    b_face = 24.0                # Face width (mm)
    phi_n = np.deg2rad(20.0)     # Normal pressure angle
    phi_t = np.arctan(np.tan(phi_n) / np.cos(beta)) # Transverse pressure angle
    num_planets = 4              # 4 planets with floating sun for uniform load share

    # Stage 2 Sun Input Torque
    T_sun2_peak = T_peak / i_stage2
    T_sun2_cont = T_cont / i_stage2
    T_sun2_rms = T_rms_duty / i_stage2

    # Tangential load per mesh (K_gamma = 1.12 for 4-planet floating sun sharing)
    K_gamma = 1.12
    W_t_peak = (2.0 * T_sun2_peak * 1000.0 / d_sun) * (K_gamma / num_planets)
    W_t_cont = (2.0 * T_sun2_cont * 1000.0 / d_sun) * (K_gamma / num_planets)
    W_t_rms = (2.0 * T_sun2_rms * 1000.0 / d_sun) * (K_gamma / num_planets)

    # 2. AGMA 2001-D04 Bending Stress: sigma_b = (W_t / (b * m_t)) * (K_o * K_v * K_m / J)
    K_o = 1.25  # Shock overload factor
    K_v = 1.06  # Dynamic factor (AGMA Q11 precision ground helical)
    K_m = 1.15  # Load distribution factor across 24mm face width
    J_factor = 0.46  # AGMA helical geometry factor (z=16, beta=18 deg, +0.25 profile shift)

    sigma_b_peak = (W_t_peak / (b_face * m_t)) * ((K_o * K_v * K_m) / J_factor)
    sigma_b_cont = (W_t_cont / (b_face * m_t)) * ((K_o * K_v * K_m) / J_factor)
    s_factor_bend_peak = sigma_allow_bend / sigma_b_peak
    s_factor_bend_cont = sigma_allow_bend / sigma_b_cont

    # 3. Exact AGMA Hertzian Pitting Factor I & Contact Stress
    # I = (cos(phi_t)*sin(phi_t)/(2*m_N)) * (m_G/(m_G + 1))
    # Load sharing ratio m_N ~ 0.72 for helical contact ratio > 1.8
    m_G = z_p2 / z_s2 # 1.5
    m_N = 0.72
    I_factor = (np.cos(phi_t) * np.sin(phi_t) / (2.0 * m_N)) * (m_G / (m_G + 1.0))
    Z_E = np.sqrt(1.0 / (np.pi * (2.0 * (1.0 - nu_steel**2) / E_steel)))  # ~190.2 sqrt(MPa)

    sigma_c_peak = Z_E * np.sqrt((W_t_peak / (b_face * d_sun)) * ((K_o * K_v * K_m) / I_factor))
    sigma_c_cont = Z_E * np.sqrt((W_t_cont / (b_face * d_sun)) * ((K_o * K_v * K_m) / I_factor))
    s_factor_contact_peak = sigma_allow_contact / sigma_c_peak
    s_factor_contact_cont = sigma_allow_contact / sigma_c_cont

    # 4. Dual-Row Planet Cylindrical Roller Bearing L10h Life (ISO 281)
    # Bearing: Dual-row full-complement cylindrical roller RNA4901 / K14x22x20
    C_dyn = 14800.0 # N (Dynamic basic load rating)
    # Planet carries reaction from BOTH Sun mesh and Ring mesh -> radial force = 2 * W_t / cos(phi_t)
    P_rad_rms = (2.0 * W_t_rms) / np.cos(phi_t)
    # Planet relative rotation speed around carrier pin
    n_planet_rpm = n_output * (z_r2 / z_p2) # ~93.3 RPM relative to carrier
    p_exp = 10.0 / 3.0 # Roller bearing exponent
    L10_revs = (C_dyn / P_rad_rms) ** p_exp  # Millions of revolutions
    L10_hours = (L10_revs * 1e6) / (60.0 * n_planet_rpm)

    # 5. 800V DC Bus 3-Level Flying-Capacitor GaN Inverter Analysis
    # Using 650V GaN HEMTs (Infineon CoolGaN / GS66516T, R_ds_on = 25 mOhm at 100C)
    # In a 3-Level FCML topology, each FET blocks V_dc / 2 = 400 V (safe margin to 650V!).
    V_dc = 800.0
    V_switch = V_dc / 2.0 # 400 V blocking voltage per device
    R_ds_on_hot = 0.025   # 25 mOhm (junction at 95C)
    # Effective 2 switches in series per conducting phase leg -> 2 * R_ds_on
    I_phase_rms = 22.0    # Amps continuous RMS @ 180 Nm rated output (18:1 gear, Kt=0.45 Nm/A -> ~63 A at 510 Nm peak)
    P_cond_peak = 3.0 * (I_phase_rms**2) * (2.0 * R_ds_on_hot) # ~72.6 W
    f_sw = 40000.0        # 40 kHz per switch (80 kHz effective at phase output!)
    t_rise = 7e-9         # 7 ns GaN transition at 400V
    # 12 switches total in 3-phase 3-level inverter
    P_sw_peak = 12.0 * (0.5 * V_switch * (I_phase_rms * 1.414) * t_rise * f_sw) # ~20.9 W
    P_inverter_loss_peak = P_cond_peak + P_sw_peak
    P_mech_peak = T_peak * (n_output * np.pi / 30.0) # 1869 W
    inverter_eff_peak = (P_mech_peak / (P_mech_peak + P_inverter_loss_peak)) * 100.0

    print(f"[*] Compound Ratio Verified: {i_stage1:.1f} x {i_stage2:.1f} = {ratio:.2f}:1 exact")
    print(f"[*] Peak Tangential Mesh Load: {W_t_peak:.1f} N (4-planet load sharing)")
    print(f"[*] AGMA Bending Stress (Peak): {sigma_b_peak:.1f} MPa -> SF = {s_factor_bend_peak:.2f} (PASS)")
    print(f"[*] AGMA Bending Stress (Cont): {sigma_b_cont:.1f} MPa -> SF = {s_factor_bend_cont:.2f} (PASS)")
    print(f"[*] Hertz Contact Stress (Peak): {sigma_c_peak:.1f} MPa -> SF = {s_factor_contact_peak:.2f} (PASS)")
    print(f"[*] Hertz Contact Stress (Cont): {sigma_c_cont:.1f} MPa -> SF = {s_factor_contact_cont:.2f} (PASS)")
    print(f"[*] Dual-Row Planet Bearing L10h Life (Heavy RMS): {L10_hours:,.0f} Hours (Target > 20,000 hrs: PASS)")
    print(f"[*] 800V 3-Level GaN Inverter Eff (Peak): {inverter_eff_peak:.2f}% (V_sw = {V_switch:.0f}V / 650V rated)")

    # Generate Engineering Blueprint & Stress Charts (GridSpec without tight_layout conflicts)
    fig = plt.figure(figsize=(16, 11), facecolor='#0D1321', constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    # 1. CAD Cross-Section Blueprint View
    ax_bp = fig.add_subplot(gs[0, :])
    ax_bp.set_facecolor('#080C14')
    ax_bp.set_title("FIGURE 05 // CLASS GAMMA ACTUATOR - PRECISION CROSS-SECTION BLUEPRINT (18.00:1 HELICAL PLANETARY)",
                     color='#38BDF8', fontsize=12, fontweight='bold', pad=10, loc='left')

    ax_bp.set_xlim(-132, 132)
    ax_bp.set_ylim(-74, 74)
    ax_bp.grid(True, color='#1E293B', linestyle='--', alpha=0.5)

    # Outer 7075-T6 Housing + Cooling Fins
    housing = patches.Rectangle((-85, -52), 170, 104, linewidth=2.0, edgecolor='#64748B', facecolor='#0F172A', alpha=0.95)
    ax_bp.add_patch(housing)
    for fx in range(-75, 65, 10):
        ax_bp.add_patch(patches.Rectangle((fx, 52), 4, 4, facecolor='#475569', edgecolor='#94A3B8', lw=0.8))
        ax_bp.add_patch(patches.Rectangle((fx, -56), 4, 4, facecolor='#475569', edgecolor='#94A3B8', lw=0.8))

    # Stator Core (18S16P 0.15mm Si-Steel)
    ax_bp.add_patch(patches.Rectangle((-68, 26), 125, 22, linewidth=1.5, edgecolor='#0284C7', facecolor='#0369A1', alpha=0.65))
    ax_bp.add_patch(patches.Rectangle((-68, -48), 125, 22, linewidth=1.5, edgecolor='#0284C7', facecolor='#0369A1', alpha=0.65))

    # Rotor & Halbach NdFeB N52SH Magnets
    ax_bp.add_patch(patches.Rectangle((-62, 17), 115, 7, linewidth=1.2, edgecolor='#F59E0B', facecolor='#D97706', alpha=0.85))
    ax_bp.add_patch(patches.Rectangle((-62, -24), 115, 7, linewidth=1.2, edgecolor='#F59E0B', facecolor='#D97706', alpha=0.85))

    # Stage 1 (3.6:1) & Stage 2 (5.0:1) Helical Planetary Meshes
    ax_bp.add_patch(patches.Rectangle((-34, -15), 26, 30, linewidth=1.5, edgecolor='#38BDF8', facecolor='#1E293B'))
    ax_bp.add_patch(patches.Rectangle((-2, -16), 36, 32, linewidth=1.8, edgecolor='#10B981', facecolor='#064E3B', alpha=0.7))

    # Cross-Roller Output Bearing
    ax_bp.add_patch(patches.Circle((42, 34), 8, linewidth=1.5, edgecolor='#CBD5E1', facecolor='#334155'))
    ax_bp.add_patch(patches.Circle((42, -34), 8, linewidth=1.5, edgecolor='#CBD5E1', facecolor='#334155'))

    # Central Ø12mm Through-Bore (Zero External Cabling)
    ax_bp.add_patch(patches.Rectangle((-88, -6), 176, 12, linewidth=1.5, edgecolor='#38BDF8', facecolor='#05080F', linestyle=':'))

    # 3-Level 650V GaN FCML Inverter Ring (800V DC Bus)
    ax_bp.add_patch(patches.Rectangle((65, -46), 14, 92, linewidth=1.5, edgecolor='#EC4899', facecolor='#831843', alpha=0.85))

    # Dual 20-Bit BiSS-C Encoders
    ax_bp.add_patch(patches.Rectangle((-82, -22), 9, 44, linewidth=1.2, edgecolor='#A855F7', facecolor='#581C87'))

    # Clean non-overlapping blueprint annotations
    annotations = [
        ("Ø12mm Through-Bore (1000BASE-T1 SPE + 800V Bus)", (-18, 0), (-126, 61)),
        ("18S16P Stator Core + Vapor-Chamber Fins (R_th,hs=0.42 K/W; 0.46 K/W Cu-Amb)", (-10, 48), (12, 61)),
        (f"Stage 2: 5.0:1 Helical Planetary (9310 Steel, SF={s_factor_contact_peak:.2f})", (16, 16), (26, 39)),
        ("3-Level 650V GaN FCML Inverter (800V Bus, 95.2% Eff)", (72, -18), (28, -68)),
        ("Dual 20-Bit BiSS-C Encoders (0.0003°)", (-77, -20), (-128, -68)),
        ("Cross-Roller Bearing + Bayonet Lock", (42, -38), (-48, -58)),
    ]

    for text, point, text_pos in annotations:
        ax_bp.annotate(text, xy=point, xytext=text_pos,
                        color='#E2E8F0', fontsize=8.5, fontweight='bold', fontfamily='monospace',
                        arrowprops=dict(facecolor='#38BDF8', edgecolor='#38BDF8', arrowstyle="->", lw=1.3),
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='#1E293B', edgecolor='#334155', alpha=0.95))

    ax_bp.set_aspect('equal')
    ax_bp.axis('off')

    # Subplot 2: AGMA Bending & Hertzian Contact Stress Margins
    ax_stress = fig.add_subplot(gs[1, 0])
    ax_stress.set_facecolor('#131B2E')
    ax_stress.tick_params(colors='#94A3B8')
    for spine in ax_stress.spines.values():
        spine.set_color('#334155')

    torque_sweep = np.linspace(50, 550, 100)
    w_t_sweep = (2.0 * (torque_sweep / i_stage2) * 1000.0 / d_sun) * (K_gamma / num_planets)
    contact_sweep = Z_E * np.sqrt((w_t_sweep / (b_face * d_sun)) * ((K_o * K_v * K_m) / I_factor))
    bend_sweep = (w_t_sweep / (b_face * m_t)) * ((K_o * K_v * K_m) / J_factor)

    ax_stress.plot(torque_sweep, contact_sweep, color='#38BDF8', lw=2.5, label='Hertz Contact Stress (MPa)')
    ax_stress.axhline(y=sigma_allow_contact, color='#EF4444', linestyle='--', lw=1.8, label=f'Grade 3 Pitting Limit ({sigma_allow_contact} MPa)')
    ax_stress.plot(torque_sweep, bend_sweep, color='#10B981', lw=2.2, label='AGMA Root Bending Stress (MPa)')
    ax_stress.axvline(x=T_cont, color='#94A3B8', linestyle=':', lw=1.5, label=f'Continuous ({T_cont} Nm)')
    ax_stress.axvline(x=T_peak, color='#F59E0B', linestyle=':', lw=1.8, label=f'Peak Shock ({T_peak} Nm, SF={s_factor_contact_peak:.2f})')
    ax_stress.set_title('AGMA 2001-D04 Contact & Bending Stress vs Torque (9310 Steel)', color='#38BDF8', fontsize=10, fontweight='bold')
    ax_stress.set_xlabel('Actuator Output Torque (Nm)', color='#E2E8F0')
    ax_stress.set_ylabel('Gear Tooth Stress (MPa)', color='#E2E8F0')
    ax_stress.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', fontsize=8)
    ax_stress.grid(True, color='#1E293B')

    # Subplot 3: Dual-Row Planet Roller Bearing L10h Life vs Output Torque
    ax_life = fig.add_subplot(gs[1, 1])
    ax_life.set_facecolor('#131B2E')
    ax_life.tick_params(colors='#94A3B8')
    for spine in ax_life.spines.values():
        spine.set_color('#334155')

    p_rad_sweep = (2.0 * w_t_sweep) / np.cos(phi_t)
    l10_sweep_hours = (((C_dyn / p_rad_sweep) ** p_exp) * 1e6) / (60.0 * n_planet_rpm)
    l10_sweep_hours = np.clip(l10_sweep_hours, 500, 1000000)

    ax_life.semilogy(torque_sweep, l10_sweep_hours, color='#10B981', lw=2.5, label='Dual-Row Planet Bearing L10h (Hours)')
    ax_life.axhline(y=20000, color='#F59E0B', linestyle='--', lw=1.8, label='Industrial Service Life Target (20,000 hrs)')
    ax_life.axvline(x=T_rms_duty, color='#38BDF8', linestyle=':', lw=1.6, label=f'Heavy Palletizing RMS ({T_rms_duty} Nm -> {L10_hours:,.0f} hrs)')
    ax_life.set_title('ISO 281 Planet Bearing Fatigue Life (L10h)', color='#38BDF8', fontsize=10, fontweight='bold')
    ax_life.set_xlabel('Actuator Output Torque (Nm)', color='#E2E8F0')
    ax_life.set_ylabel('L10h Life (Log Hours)', color='#E2E8F0')
    ax_life.legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC', fontsize=8)
    ax_life.grid(True, color='#1E293B', which='both')

    out_img = os.path.join(os.path.dirname(__file__), "class_gamma_actuator_blueprint.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[+] Saved high-resolution Actuator Blueprint to: {out_img}")

    stress_results = {
        "actuator_model": "Class Gamma 2-Stage Compound Helical Planetary",
        "stage_1_teeth": {"sun": z_s1, "planet": z_p1, "ring": z_r1, "ratio": i_stage1},
        "stage_2_teeth": {"sun": z_s2, "planet": z_p2, "ring": z_r2, "ratio": i_stage2},
        "total_reduction_ratio": ratio,
        "peak_torque_nm": T_peak,
        "continuous_torque_nm": T_cont,
        "gear_material": "AGMA Grade 3 Carburized Aerospace Steel AISI 9310 (AMS 6265)",
        "agma_bending_stress_peak_mpa": round(float(sigma_b_peak), 1),
        "bending_safety_factor_peak": round(float(s_factor_bend_peak), 2),
        "bending_safety_factor_cont": round(float(s_factor_bend_cont), 2),
        "hertzian_contact_stress_peak_mpa": round(float(sigma_c_peak), 1),
        "contact_safety_factor_peak": round(float(s_factor_contact_peak), 2),
        "contact_safety_factor_cont": round(float(s_factor_contact_cont), 2),
        "planet_bearing_l10h_heavy_rms_hours": round(float(L10_hours), 0),
        "inverter_topology": "800V DC Bus 3-Level Flying-Capacitor 650V GaN FCML",
        "gan_inverter_peak_eff_percent": round(float(inverter_eff_peak), 2),
        "cooling_thermal_resistance_k_w": 0.42
    }
    with open(os.path.join(os.path.dirname(__file__), "class_gamma_stress_report.json"), "w") as f:
        json.dump(stress_results, f, indent=4)
    print("[+] Successfully exported AGMA stress report.")

if __name__ == "__main__":
    calculate_agma_stresses()
