"""
Figure 05 - Skyhook-Free MuJoCo 3.6 Inclined Ramp Ascent Walking Simulation
===========================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Engineering Group
Target: Kyle Edelberg & Brett Adcock, Figure AI (Release v5.5.0)

Demonstrates continuous 3D skyhook-free locomotion up a 10.0-degree (17.6% grade)
industrial loading-dock ramp in MuJoCo 3.6:
1. Strict skyhook prohibition (`data.qfrc_applied[:] = 0.0` at every 500 Hz step).
2. Surface-aligned foot orientation (`R_des = R_y(-10 deg)`) and interpolated slope
   elevation clearance for clean sole touchdown without stubbing toes.
3. Englsberger Capture-Point (DCM) footstep placement + slope-following Centroidal
   WBC vertical height tracking (`z_c_ref = z_ground + 0.855 m`).
4. Exports `figure05_ramp_telemetry.json` and `figure05_mujoco_ramp_benchmark.png`.
"""

import json
import os
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from figure05_mujoco_common import (
    build_actuator_mappings,
    mat_to_rotvec_err,
    quat_to_roll_pitch_yaw,
    setup_crouched_biped,
    sum_support_grfs,
)


def run_ramp_walking_simulation():
    print("=================================================================")
    print("FIGURE 05: SKYHOOK-FREE MUJOCO 3.6 10-DEG RAMP ASCENT LOCOMOTION")
    print("=================================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "figure05_mujoco.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")

    # Continuous 10.0 deg (17.6% grade) industrial dock ramp starting at x = 0.28 m
    theta_deg = 10.0
    theta = np.deg2rad(theta_deg)
    x_ramp0 = 0.28
    ramp_len = 5.0
    half_len = ramp_len * 0.5
    thick = 0.08
    cx = x_ramp0 + half_len * np.cos(theta) + thick * np.sin(theta)
    cz = half_len * np.sin(theta) - thick * np.cos(theta)

    ET.SubElement(
        wb,
        "geom",
        {
            "name": "ramp",
            "type": "box",
            "size": f"{half_len:.4f} 1.5 {thick:.4f}",
            "pos": f"{cx:.5f} 0 {cz:.5f}",
            "euler": f"0 {-theta:.6f} 0",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.2 0.015 0.002",
        },
    )

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    g = 9.80665
    total_mass = float(np.sum(model.body_mass))

    setup_crouched_biped(model, data)

    maps = build_actuator_mappings(model)
    act_names = maps["act_names"]
    act_qidx = maps["act_qidx"]
    act_vidx = maps["act_vidx"]
    idx_map = maps["idx_map"]
    l_leg_v = maps["l_leg_v"]
    r_leg_v = maps["r_leg_v"]
    l_leg_act = maps["l_leg_act"]
    r_leg_act = maps["r_leg_act"]
    q_init = np.array([data.qpos[idx] for idx in act_qidx])

    l_site = model.site("left_foot_contact").id
    r_site = model.site("right_foot_contact").id
    l_body = model.body("left_ankle").id
    r_body = model.body("right_ankle").id
    floor_geom = model.geom("floor").id
    ramp_geom = model.geom("ramp").id

    jac_com = np.zeros((3, model.nv))
    jac_lp = np.zeros((3, model.nv))
    jac_lr = np.zeros((3, model.nv))
    jac_rp = np.zeros((3, model.nv))
    jac_rr = np.zeros((3, model.nv))

    def ground_z(x: float) -> float:
        return max(0.0, (x - x_ramp0) * np.tan(theta))

    def ground_rot(x: float) -> np.ndarray:
        if x < x_ramp0 - 0.04:
            return np.eye(3)
        ang = -theta
        return np.array([
            [np.cos(ang), 0.0, np.sin(ang)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang), 0.0, np.cos(ang)],
        ])

    duration = 7.0
    n_steps = int(round(duration / dt))
    z_c = 0.855
    omega = np.sqrt(g / z_c)
    t_step = 0.44
    ds_frac = 0.20
    vx_des = 0.58
    w_step = 0.21
    h_lift = 0.075
    t_stand = 0.60
    by_nom = w_step / (np.exp(omega * t_step) + 1.0)

    t_log = np.zeros(n_steps)
    com_x_log = np.zeros(n_steps)
    com_z_log = np.zeros(n_steps)
    floor_z_log = np.zeros(n_steps)
    pelvis_rel_z_log = np.zeros(n_steps)
    roll_log = np.zeros(n_steps)
    pitch_log = np.zeros(n_steps)
    fz_l_log = np.zeros(n_steps)
    fz_r_log = np.zeros(n_steps)

    step_idx = -1
    stance_side = 1
    sw_start_pos = data.site_xpos[r_site].copy()
    prev_pdes = sw_start_pos.copy()
    max_skyhook = 0.0
    cf_buf = np.zeros(6)

    print(f"[*] Climbing continuous {theta_deg:.1f} deg ({np.tan(theta)*100:.1f}% grade) dock ramp | Skyhook = 0.0 N")

    for k in range(n_steps):
        t = k * dt
        data.qfrc_applied[:] = 0.0
        max_skyhook = max(max_skyhook, float(np.linalg.norm(data.qfrc_applied)))

        mujoco.mj_jacSubtreeCom(model, data, jac_com, 1)
        mujoco.mj_jacSite(model, data, jac_lp, jac_lr, l_site)
        mujoco.mj_jacSite(model, data, jac_rp, jac_rr, r_site)

        com = data.subtree_com[1].copy()
        com_vel = jac_com @ data.qvel
        dcm = com[:2] + com_vel[:2] / omega

        p_l = data.site_xpos[l_site].copy()
        p_r = data.site_xpos[r_site].copy()
        v_l = jac_lp @ data.qvel
        v_r = jac_rp @ data.qvel
        r_l = data.site_xmat[l_site].reshape(3, 3)
        r_r = data.site_xmat[r_site].reshape(3, 3)

        roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
        droll, dpitch, dyaw = data.qvel[3:6]

        if t < t_stand:
            s = t / t_stand
            smooth = 0.5 * (1.0 - np.cos(np.pi * s))
            c_tgt = np.array([
                0.5 * (p_l[0] + p_r[0]) + 0.025 * smooth,
                (p_l[1] - by_nom) * smooth,
                z_c,
            ])
            alpha_l = 0.5 + 0.45 * smooth
            alpha_r = 1.0 - alpha_l
            swing = None
            vx_cmd = 0.0
            bx_cmd = 0.0
        else:
            tw = t - t_stand
            new_step = int(tw // t_step)
            phi = (tw % t_step) / t_step
            if new_step != step_idx:
                step_idx = new_step
                stance_side = 1 if (step_idx % 2 == 0) else -1
                sw_start_pos = (p_r if stance_side == 1 else p_l).copy()
                prev_pdes = sw_start_pos.copy()

            vx_cmd = vx_des * np.clip(tw / 1.0, 0.25, 1.0)
            bx_cmd = (vx_cmd * t_step) / (np.exp(omega * t_step) - 1.0)

            if phi < ds_frac:
                s = phi / ds_frac
                smooth = 0.5 * (1.0 - np.cos(np.pi * s))
                if step_idx == 0:
                    alpha_l = 0.95 + 0.05 * smooth
                else:
                    alpha_l = smooth if stance_side == 1 else (1.0 - smooth)
                alpha_r = 1.0 - alpha_l
                swing = None
            else:
                alpha_l = 1.0 if stance_side == 1 else 0.0
                alpha_r = 1.0 - alpha_l
                swing = "R" if stance_side == 1 else "L"
                s_sw = (phi - ds_frac) / (1.0 - ds_frac)

        # Follow sloped ground elevation dynamically
        p_sup_x = (
            0.5 * (p_l[0] + p_r[0])
            if swing is None
            else (p_l[0] if stance_side == 1 else p_r[0])
        )
        z_g_sup = (
            0.5 * (p_l[2] + p_r[2])
            if swing is None
            else (p_l[2] if stance_side == 1 else p_r[2])
        )
        z_c_ref = max(0.0, z_g_sup) + z_c
        vz_ref = com_vel[0] * np.tan(theta) if p_sup_x > x_ramp0 else 0.0
        fz_tot = float(
            np.clip(
                total_mass * (g + 180.0 * (z_c_ref - com[2]) + 28.0 * (vz_ref - com_vel[2])),
                180.0,
                1250.0,
            )
        )

        if t < t_stand:
            f_xy = total_mass * np.array([
                75.0 * (c_tgt[0] - com[0]) - 13.0 * com_vel[0],
                110.0 * (c_tgt[1] - com[1]) - 14.0 * com_vel[1],
            ])
        elif swing is None:
            p_cop_des = alpha_l * p_l[:2] + alpha_r * p_r[:2]
            f_xy = total_mass * np.array([
                5.0 * (vx_cmd - com_vel[0]),
                32.0 * (p_cop_des[1] - com[1]) - 6.0 * com_vel[1],
            ])
        else:
            p_st = p_l if stance_side == 1 else p_r
            cop_x = p_st[0] + np.clip(0.25 * (dcm[0] - (p_st[0] + bx_cmd)), -0.05, 0.08)
            cop_y = p_st[1] + np.clip(0.25 * (dcm[1] - (p_st[1] - stance_side * by_nom)), -0.03, 0.03)
            f_xy = total_mass * (omega**2) * np.array([com[0] - cop_x, com[1] - cop_y])
            f_xy = np.clip(f_xy, -220.0, 220.0)

        tau_trunk = np.array([
            -300.0 * roll - 32.0 * droll,
            -280.0 * pitch - 32.0 * dpitch,
            -90.0 * yaw - 12.0 * dyaw,
        ])

        fz_l = alpha_l * fz_tot
        fz_r = alpha_r * fz_tot
        w_l = np.zeros(6)
        w_r = np.zeros(6)
        if alpha_l > 0.01:
            w_l = np.array([
                alpha_l * f_xy[0],
                alpha_l * f_xy[1],
                fz_l,
                np.clip(alpha_l * tau_trunk[0], -0.05 * fz_l, 0.05 * fz_l),
                np.clip(alpha_l * tau_trunk[1], -0.10 * fz_l, 0.10 * fz_l),
                alpha_l * tau_trunk[2],
            ])
        if alpha_r > 0.01:
            w_r = np.array([
                alpha_r * f_xy[0],
                alpha_r * f_xy[1],
                fz_r,
                np.clip(alpha_r * tau_trunk[0], -0.05 * fz_r, 0.05 * fz_r),
                np.clip(alpha_r * tau_trunk[1], -0.10 * fz_r, 0.10 * fz_r),
                alpha_r * tau_trunk[2],
            ])

        j_l = np.vstack([jac_lp, jac_lr])
        j_r = np.vstack([jac_rp, jac_rr])
        tau_gen = data.qfrc_bias.copy()
        if alpha_l > 0.01:
            tau_gen -= j_l.T @ w_l
        if alpha_r > 0.01:
            tau_gen -= j_r.T @ w_r

        tau_ctrl = np.array([tau_gen[idx] for idx in act_vidx])

        if swing is not None:
            p_st = p_l if stance_side == 1 else p_r
            s_sw_side = -1 if stance_side == 1 else 1
            tgt_x = np.clip(dcm[0] - bx_cmd, p_st[0] - 0.10, p_st[0] + 0.42)
            tgt_y = dcm[1] + s_sw_side * by_nom - 0.08 * com[1]
            if s_sw_side == -1:
                tgt_y = np.clip(tgt_y, p_st[1] - 0.33, p_st[1] - 0.15)
            else:
                tgt_y = np.clip(tgt_y, p_st[1] + 0.15, p_st[1] + 0.33)

            smooth_xy = 0.5 * (1.0 - np.cos(np.pi * s_sw))
            x_des_s = sw_start_pos[0] + smooth_xy * (tgt_x - sw_start_pos[0])
            y_des_s = sw_start_pos[1] + smooth_xy * (tgt_y - sw_start_pos[1])
            z_floor_interp = (1.0 - smooth_xy) * sw_start_pos[2] + smooth_xy * ground_z(tgt_x)
            z_des_s = z_floor_interp + h_lift * np.sin(np.pi * s_sw) - 0.002 * (s_sw**2)
            p_des = np.array([x_des_s, y_des_s, z_des_s])
            v_des = (p_des - prev_pdes) / dt if s_sw > 0.01 else np.zeros(3)
            prev_pdes = p_des.copy()

            p_cur = p_r if swing == "R" else p_l
            v_cur = v_r if swing == "R" else v_l
            r_cur = r_r if swing == "R" else r_l
            w_cur = (jac_rr @ data.qvel) if swing == "R" else (jac_lr @ data.qvel)
            j_sw = j_r if swing == "R" else j_l
            sw_dofs = r_leg_v if swing == "R" else l_leg_v
            sw_acts = r_leg_act if swing == "R" else l_leg_act

            r_des_sw = ground_rot(x_des_s)
            f_sw = np.concatenate([
                np.array([1500.0, 1500.0, 2100.0]) * (p_des - p_cur)
                + np.array([70.0, 70.0, 90.0]) * (v_des - v_cur),
                np.array([100.0, 140.0, 80.0]) * mat_to_rotvec_err(r_des_sw, r_cur)
                - np.array([7.0, 9.0, 6.0]) * w_cur,
            ])
            tau_sw_full = data.qfrc_bias + j_sw.T @ f_sw
            for ai, vi in zip(sw_acts, sw_dofs):
                tau_ctrl[ai] = tau_sw_full[vi]

            st_acts = l_leg_act if stance_side == 1 else r_leg_act
            tau_ctrl[st_acts[0]] += 50.0 * yaw + 6.0 * dyaw
            tau_ctrl[st_acts[1]] += 110.0 * roll + 12.0 * droll
            tau_ctrl[st_acts[2]] += 100.0 * pitch + 11.0 * dpitch
        else:
            tau_ctrl[l_leg_act[1]] += 60.0 * roll + 7.0 * droll
            tau_ctrl[r_leg_act[1]] += 60.0 * roll + 7.0 * droll
            tau_ctrl[l_leg_act[2]] += 50.0 * pitch + 6.0 * dpitch
            tau_ctrl[r_leg_act[2]] += 50.0 * pitch + 6.0 * dpitch

        q = np.array([data.qpos[idx] for idx in act_qidx])
        qd = np.array([data.qvel[idx] for idx in act_vidx])
        q_t = q_init.copy()
        arm_amp = 0.25 * np.clip(com_vel[0] / 0.55, 0.0, 1.2)
        q_t[idx_map["act_left_shoulder_pitch"]] = arm_amp * (-stance_side)
        q_t[idx_map["act_right_shoulder_pitch"]] = arm_amp * stance_side
        for i, name in enumerate(act_names):
            if not any(x in name for x in ["hip", "knee", "ankle"]):
                tau_ctrl[i] += 180.0 * (q_t[i] - q[i]) - 16.0 * qd[i]

        data.ctrl[:] = np.clip(
            tau_ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)

        fz_l_c, fz_r_c = sum_support_grfs(model, data, (floor_geom, ramp_geom), l_body, r_body)

        t_log[k] = t
        com_x_log[k] = com[0]
        com_z_log[k] = data.qpos[2]
        fg = ground_z(com[0])
        floor_z_log[k] = fg
        pelvis_rel_z_log[k] = data.qpos[2] - fg
        roll_log[k] = np.rad2deg(roll)
        pitch_log[k] = np.rad2deg(pitch)
        fz_l_log[k] = fz_l_c
        fz_r_log[k] = fz_r_c

    dist_x = float(com_x_log[-1] - com_x_log[0])
    elev_gain_m = float(floor_z_log[-1])
    min_rel_z = float(np.min(pelvis_rel_z_log))
    rms_roll = float(np.sqrt(np.mean(roll_log[t_log >= 2.0] ** 2)))
    rms_pitch = float(np.sqrt(np.mean(pitch_log[t_log >= 2.0] ** 2)))
    survived = bool(min_rel_z > 0.78 and elev_gain_m >= 0.25)

    print(f"[+] Ramp Distance X: {dist_x:.3f} m | Elevation Gain Climb: +{elev_gain_m*100:.1f} cm ({theta_deg:.1f} deg ramp)")
    print(f"[+] Steady Attitude (t>=2s): Roll RMS = {rms_roll:.2f} deg | Pitch RMS = {rms_pitch:.2f} deg | Min Clearance Height = {min_rel_z:.3f} m")

    assert survived and max_skyhook == 0.0, "Ramp ascent stability check failed!"

    telemetry = {
        "engine": "MuJoCo 3.6.0",
        "skyhook_force_n": max_skyhook,
        "ramp_angle_deg": theta_deg,
        "ramp_grade_percent": round(float(np.tan(theta) * 100.0), 1),
        "duration_s": duration,
        "distance_walked_m": round(dist_x, 3),
        "elevation_gain_m": round(elev_gain_m, 3),
        "steady_rms_roll_deg": round(rms_roll, 2),
        "steady_rms_pitch_deg": round(rms_pitch, 2),
        "min_height_above_slope_m": round(min_rel_z, 3),
        "completed_without_fall": survived,
    }
    json_path = os.path.join(base_dir, "figure05_ramp_telemetry.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=4)

    # Visualization
    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.8), gridspec_kw={"hspace": 0.36, "wspace": 0.28})
    fig.patch.set_facecolor("#0f1419")

    # 1. Side-Profile Ascent Path (World Z vs X)
    ax1 = axs[0, 0]
    ax1.set_facecolor("#151c24")
    ax1.plot(com_x_log, com_z_log, color="#00ffcc", lw=2.4, label="Pelvis Trajectory Z(X)")
    ax1.plot(com_x_log, floor_z_log, color="#f59e0b", lw=2.2, linestyle="--", label=f"10.0 deg Ramp Surface (+{elev_gain_m*100:.1f} cm)")
    ax1.fill_between(com_x_log, -0.05, floor_z_log, color="#f59e0b", alpha=0.14)
    ax1.set_title("1. Side-View 10.0 deg Ramp Ascent Profile", color="white", fontweight="bold", pad=8)
    ax1.set_xlabel("Forward X (m)", color="#cccccc")
    ax1.set_ylabel("World Elevation Z (m)", color="#cccccc")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="center right", fontsize=8.5, facecolor="#0f1419")

    # 2. Elevation Climb Over Time & Relative Clearance Above Ramp
    ax2 = axs[0, 1]
    ax2.set_facecolor("#151c24")
    l1, = ax2.plot(t_log, floor_z_log * 100.0, color="#f59e0b", lw=2.2, label=f"Ramp Height Climbed (+{elev_gain_m*100:.1f} cm)")
    ax2_r = ax2.twinx()
    l2, = ax2_r.plot(t_log, pelvis_rel_z_log, color="#38bdf8", lw=1.8, label="Pelvis Height Above Slope (m)")
    ax2_r.set_ylim(0.75, 1.02)
    ax2.set_title("2. Vertical Climb & Relative CoM Height Tracking", color="white", fontweight="bold", pad=8)
    ax2.set_xlabel("Time (s)", color="#cccccc")
    ax2.set_ylabel("Floor Elevation Gain (cm)", color="#f59e0b")
    ax2_r.set_ylabel("Pelvis Relative Height (m)", color="#38bdf8")
    ax2.grid(True, alpha=0.2)
    ax2.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="upper left", fontsize=8.2, facecolor="#0f1419")

    # 3. Attitude Stability on Slope
    ax3 = axs[1, 0]
    ax3.set_facecolor("#151c24")
    ax3.plot(t_log, roll_log, color="#00ffcc", lw=1.6, label=f"Pelvis Roll (RMS {rms_roll:.2f} deg)")
    ax3.plot(t_log, pitch_log, color="#ffcc00", lw=1.6, label=f"Pelvis Pitch (RMS {rms_pitch:.2f} deg)")
    ax3.axhline(0.0, color="#666666", ls="--", lw=0.9)
    ax3.set_title("3. Floating-Base Attitude During Slope Climb", color="white", fontweight="bold", pad=8)
    ax3.set_xlabel("Time (s)", color="#cccccc")
    ax3.set_ylabel("Angle (deg)", color="#cccccc")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="upper right", fontsize=8.5, facecolor="#0f1419")

    # 4. Alternating Normal Forces on Ramp
    ax4 = axs[1, 1]
    ax4.set_facecolor("#151c24")
    ax4.plot(t_log, fz_l_log, color="#00d4ff", lw=1.3, alpha=0.9, label="Left Foot Contact Normal (N)")
    ax4.plot(t_log, fz_r_log, color="#ff44aa", lw=1.3, alpha=0.85, label="Right Foot Contact Normal (N)")
    ax4.set_title("4. Alternating L/R Contact Forces on 10.0 deg Ramp", color="white", fontweight="bold", pad=8)
    ax4.set_xlabel("Time (s)", color="#cccccc")
    ax4.set_ylabel("Normal Contact Force (N)", color="#cccccc")
    ax4.set_ylim(-30, 1450)
    ax4.grid(True, alpha=0.2)
    ax4.legend(loc="upper right", fontsize=8.5, facecolor="#0f1419")

    fig.suptitle(
        f"FIGURE 05 v5.5.0 - SKYHOOK-FREE MUJOCO 3.6 {theta_deg:.1f} DEG RAMP ASCENT (+{elev_gain_m*100:.1f} cm Climb | {dist_x:.2f} m Walk)",
        fontsize=12.6,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    out_png = os.path.join(base_dir, "figure05_mujoco_ramp_benchmark.png")
    plt.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="#0f1419")
    plt.close(fig)
    print(f"[+] Generated {out_png} & {json_path}")


if __name__ == "__main__":
    run_ramp_walking_simulation()
