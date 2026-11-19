"""
Figure 05 - Apex-Dex 22-DoF Causal Tactile Grasp & Slip Reflex Simulator
========================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Group
Target: Kyle Edelberg & Brett Adcock, Figure AI

Simulates and benchmarks (strictly causal — NO ground-truth oracle access):
1. Dynamic grasping of an unmodeled slippery 4.5 kg machined steel shaft:
   - Oil contamination drops static friction mu_s: 0.55 -> 0.22 at t = 0.20 s.
   - Sudden 1.5g vertical upward arm acceleration jerk over t = 0.25 .. 0.45 s.
2. Stribeck Coulomb friction physics + GelSkin localized CMOS sub-ROI FPGA reflex
   (1.75 ms hardware pipeline delay, observing ONLY delayed elastomer shear
   load F_s and micro-slip velocity v_slip — ZERO knowledge of true mu(t) or jerk).
3. Physical 1st-order tendon actuator lag (tau = 2.5 ms) with slew-rate limit
   (18,000 N/s) and 320 N ceiling, compared against an Open-Loop 45 N gripper.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def run_tactile_slip_simulation():
    print("=================================================================")
    print("FIGURE 05: APEX-DEX 22-DOF CAUSAL TACTILE SLIP REFLEX ENGINE")
    print("=================================================================")

    rng = np.random.default_rng(42)

    # Simulation parameters (2 kHz physics integration = 0.5 ms step)
    dt = 0.0005
    duration = 1.0
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)

    # Grasped payload (machined steel shaft held by 2 opposing palmar/finger patches)
    m_obj = 4.5  # kg
    g = 9.80665
    weight = m_obj * g  # 44.13 N

    # Unmodeled external disturbance: upward arm jerk at t = 0.25..0.45 s (1.5g)
    accel_arm = np.zeros(n_steps)
    t_dist_start, t_dist_end = 0.25, 0.45
    accel_arm[(time >= t_dist_start) & (time < t_dist_end)] = 1.5 * g

    # True external shear load per finger patch (split across 2 patches)
    f_shear_demand = (weight + m_obj * accel_arm) / 2.0

    # True hidden surface friction profile (oil drop at t = 0.20 s, + Stribeck weakening)
    mu_static_base = np.where(time < 0.20, 0.55, 0.22)

    def effective_mu(mu_s: float, v_slip: float) -> float:
        # Stribeck velocity weakening from static mu_s down to 0.85*mu_s during gross slip
        return mu_s * (0.85 + 0.15 * np.exp(-abs(v_slip) / 0.015))

    # --- Scenario 1: Conventional Open-Loop Gripper (Constant 45 N) ---
    f_normal_open = np.full(n_steps, 45.0)
    slip_rate_open = np.zeros(n_steps)
    slip_disp_open = np.zeros(n_steps)

    for i in range(1, n_steps):
        mu_curr = effective_mu(mu_static_base[i], slip_rate_open[i - 1])
        f_max_friction = mu_curr * f_normal_open[i]
        excess_shear = f_shear_demand[i] - f_max_friction
        if excess_shear > 0.0:
            slip_accel = (2.0 * excess_shear) / m_obj
            slip_rate_open[i] = slip_rate_open[i - 1] + slip_accel * dt
        else:
            slip_rate_open[i] = max(0.0, slip_rate_open[i - 1] + (2.0 * excess_shear) / m_obj * dt)
        slip_disp_open[i] = slip_disp_open[i - 1] + slip_rate_open[i] * dt

    # --- Scenario 2: Figure 05 Causal GelSkin Reflex (NO oracle access to mu or accel) ---
    f_normal_closed = np.zeros(n_steps)
    f_normal_closed[0] = 45.0
    f_n_cmd_log = np.zeros(n_steps)
    f_n_cmd_log[0] = 45.0
    slip_rate_closed = np.zeros(n_steps)
    slip_disp_closed = np.zeros(n_steps)
    f_shear_transmitted = np.zeros(n_steps)
    mu_hat_log = np.zeros(n_steps)

    fpga_reflex_delay = 0.00175  # 1.75 ms FPGA optical-flow & shear pipeline
    delay_steps = max(1, int(round(fpga_reflex_delay / dt)))  # 4 steps (~2.0 ms discrete)
    tau_tendon = 0.0025  # 2.5 ms physical tendon + elastomer compliance time constant
    max_slew_rate = 18000.0  # N/s micro-tendon peak actuation slew rate

    slip_integral = 0.0
    mu_hat = 0.50  # Blind prior before oil arrives
    mu_hat_log[0] = mu_hat
    f_n_cmd = 45.0

    for i in range(1, n_steps):
        # 1. Plant physics evolution using PREVIOUS clamping force
        mu_true = effective_mu(mu_static_base[i], slip_rate_closed[i - 1])
        f_max_friction = mu_true * f_normal_closed[i - 1]
        excess_shear = f_shear_demand[i] - f_max_friction

        if excess_shear > 0.0 or slip_rate_closed[i - 1] > 0.0:
            slip_accel = (2.0 * (f_shear_demand[i] - f_max_friction)) / m_obj
            slip_rate_closed[i] = max(0.0, slip_rate_closed[i - 1] + slip_accel * dt)
        else:
            slip_rate_closed[i] = 0.0

        # Coulomb law: kinetic sliding transmits f_max_friction = mu*Fn; sticking transmits f_shear_demand
        f_shear_transmitted[i] = f_max_friction if slip_rate_closed[i] > 1e-5 else min(f_shear_demand[i], f_max_friction)
        slip_disp_closed[i] = slip_disp_closed[i - 1] + slip_rate_closed[i] * dt

        # 2. Causal Sensor Measurements (delayed by 1.75 ms + physical measurement noise)
        eval_idx = max(0, i - delay_steps)
        v_slip_meas = max(0.0, slip_rate_closed[eval_idx] + float(rng.normal(0.0, 1.5e-4)))
        f_shear_meas = max(0.0, f_shear_transmitted[eval_idx] + float(rng.normal(0.0, 0.25)))
        f_n_meas = max(1.0, f_normal_closed[eval_idx] + float(rng.normal(0.0, 0.3)))

        # 3. Causal Online Friction Observer (only active during kinetic slip when F_s == mu*F_n)
        if v_slip_meas > 0.0025:
            mu_obs = float(np.clip(f_shear_meas / f_n_meas, 0.14, 0.65))
            mu_hat = 0.70 * mu_hat + 0.30 * mu_obs
        mu_hat_log[i] = mu_hat

        # 4. Causal Tactile PI + Observer Margin Controller (ZERO access to true mu or jerk!)
        v_thresh = 0.0015  # 1.5 mm/s incipient elastomer shear threshold
        if v_slip_meas > v_thresh:
            slip_err = v_slip_meas - v_thresh
            slip_integral += slip_err * dt
            # Demand traction safety margin (1 / 0.68 ~ 1.47x) over observed shear via estimated mu_hat
            obs_shear_floor = (f_shear_meas / max(mu_hat * 0.68, 0.10)) + 15.0
            pi_boost = 4800.0 * slip_err + 185000.0 * slip_integral
            f_n_cmd = max(f_n_cmd, obs_shear_floor) + 0.35 * pi_boost
        else:
            # Hold safe traction level during high shear load; gently relax only when calm
            shear_margin_hold = (f_shear_meas / max(mu_hat * 0.72, 0.10)) + 12.0
            if f_n_cmd > shear_margin_hold:
                f_n_cmd = max(shear_margin_hold, f_n_cmd - 12.0 * dt)
            f_n_cmd = max(45.0, f_n_cmd)

        f_n_cmd = float(np.clip(f_n_cmd, 45.0, 320.0))
        f_n_cmd_log[i] = f_n_cmd

        # 5. Physical 1st-order Tendon Actuator Lag & Slew-Rate Saturation
        df_dt = (f_n_cmd - f_normal_closed[i - 1]) / tau_tendon
        df_dt = float(np.clip(df_dt, -max_slew_rate, max_slew_rate))
        f_normal_closed[i] = float(np.clip(f_normal_closed[i - 1] + df_dt * dt, 45.0, 320.0))

    final_slip_open_mm = float(slip_disp_open[-1] * 1000.0)
    final_slip_closed_mm = float(slip_disp_closed[-1] * 1000.0)
    peak_reflex_fn = float(np.max(f_normal_closed))

    print(f"[*] Open-Loop Gripper Final Slip: {final_slip_open_mm:.1f} mm (OBJECT DROPPED / FAILED)")
    print(f"[*] Figure 05 Causal GelSkin Slip Arrested at: {final_slip_closed_mm:.2f} mm (OBJECT RETAINED)")
    print(f"[*] Causal Estimated Friction Floor mu_hat: {mu_hat_log[-1]:.3f} (True hidden oil mu: 0.220)")
    print(f"[*] Tendon Reflex Clamping Force Peak: {peak_reflex_fn:.1f} N (Cap: 320.0 N)")

    assert final_slip_closed_mm < 1.5, f"Causal reflex slip too large: {final_slip_closed_mm:.2f} mm"
    assert peak_reflex_fn <= 320.0, f"Peak clamping force exceeded 320 N hardware cap: {peak_reflex_fn:.1f} N"
    assert abs(mu_hat_log[-1] - 0.22) < 0.06, f"Online friction observer did not converge: {mu_hat_log[-1]:.3f}"

    # Generate Publication-Quality Visualizations
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor("#07090E")

    for ax in axs.flat:
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#E2E8F0")
        ax.yaxis.label.set_color("#E2E8F0")
        ax.title.set_color("#38BDF8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Subplot 1: Clamping Normal Force Response
    ax_force = axs[0, 0]
    ax_force.plot(time * 1000, f_normal_closed, color="#38BDF8", lw=2.5, label="Figure 05 Causal GelSkin Reflex")
    ax_force.plot(time * 1000, f_normal_open, color="#94A3B8", linestyle="--", lw=2.0, label="Open-Loop Gripper (Constant 45 N)")
    ax_force.axvspan(t_dist_start * 1000, t_dist_end * 1000, color="#F59E0B", alpha=0.15, label="1.5g Acceleration Jerk")
    ax_force.axvline(200.0, color="#EF4444", linestyle=":", lw=1.4, label="Oil Drop (mu: 0.55 -> 0.22)")
    ax_force.set_title("Causal Contact Force Modulation Under Oil & 1.5g Jerk", fontsize=11, fontweight="bold")
    ax_force.set_xlabel("Time (ms)")
    ax_force.set_ylabel("Clamping Force Fn (N)")
    ax_force.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8.5)
    ax_force.grid(True, color="#1E293B")

    # Subplot 2: Relative Object Slip Displacement
    ax_slip = axs[0, 1]
    ax_slip.plot(time * 1000, slip_disp_closed * 1000, color="#10B981", lw=2.8, label=f"Causal GelSkin Reflex ({final_slip_closed_mm:.2f} mm)")
    ax_slip.plot(time * 1000, slip_disp_open * 1000, color="#EF4444", lw=2.5, linestyle="--", label=f"Open-Loop Drop ({final_slip_open_mm:.0f} mm)")
    ax_slip.axhline(5.0, color="#F59E0B", linestyle=":", lw=1.5, label="5 mm Loss-of-Grasp Threshold")
    ax_slip.set_title("Object Micro-Slip Displacement (mm)", fontsize=11, fontweight="bold")
    ax_slip.set_xlabel("Time (ms)")
    ax_slip.set_ylabel("Slip Displacement (mm)")
    ax_slip.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8.5)
    ax_slip.grid(True, color="#1E293B")

    # Subplot 3: Friction Limit & Shear Force Transition
    ax_shear = axs[1, 0]
    ax_shear.plot(time * 1000, f_shear_demand, color="#EC4899", lw=2.2, label="True External Shear Load Demand")
    ax_shear.plot(time * 1000, mu_static_base * f_normal_closed, color="#38BDF8", lw=2.0, label="Figure 05 Usable Traction (mu * Fn)")
    ax_shear.plot(time * 1000, mu_static_base * f_normal_open, color="#94A3B8", linestyle=":", lw=1.8, label="Open-Loop Traction Limit")
    ax_shear.set_title("Coulomb Stribeck Friction Cone Traction Margin", fontsize=11, fontweight="bold")
    ax_shear.set_xlabel("Time (ms)")
    ax_shear.set_ylabel("Force (N)")
    ax_shear.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8.5)
    ax_shear.grid(True, color="#1E293B")

    # Subplot 4: Causal Friction Observer Convergence vs True Hidden mu(t)
    ax_obs = axs[1, 1]
    ax_obs.plot(time * 1000, mu_static_base, color="#F59E0B", linestyle="--", lw=2.0, label="True Hidden Friction mu_s(t)")
    ax_obs.plot(time * 1000, mu_hat_log, color="#A855F7", lw=2.5, label=f"Causal FPGA Observer mu_hat ({mu_hat_log[-1]:.3f})")
    ax_obs.set_title("Online Tactile Friction Observer Convergence (Blind to Oil)", fontsize=11, fontweight="bold")
    ax_obs.set_xlabel("Time (ms)")
    ax_obs.set_ylabel("Coefficient of Friction (mu)")
    ax_obs.set_ylim(0.10, 0.65)
    ax_obs.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8.5)
    ax_obs.grid(True, color="#1E293B")

    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "figure05_tactile_slip_benchmark.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved high-resolution Tactile Slip Benchmark to: {chart_path}")

    tactile_metrics = {
        "grasped_object_mass_kg": m_obj,
        "dynamic_jerk_accel_g": 1.5,
        "causal_controller_oracle_free": True,
        "open_loop_slip_mm": round(final_slip_open_mm, 1),
        "open_loop_result": "FAILED_DROPPED",
        "gelskin_reflex_slip_mm": round(final_slip_closed_mm, 2),
        "gelskin_result": "RETAINED_STABILIZED",
        "fpga_reflex_delay_ms": round(fpga_reflex_delay * 1000.0, 2),
        "observed_mu_hat_final": round(float(mu_hat_log[-1]), 3),
        "peak_clamping_force_n": round(peak_reflex_fn, 1),
    }
    with open(os.path.join(os.path.dirname(__file__), "figure05_tactile_metrics.json"), "w") as f:
        json.dump(tactile_metrics, f, indent=4)
    print("[+] Successfully exported tactile metrics JSON.")


if __name__ == "__main__":
    run_tactile_slip_simulation()
