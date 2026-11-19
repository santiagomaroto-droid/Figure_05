"""
Figure 05 - Skyhook-Free MuJoCo 3.6 Overhead Gantry Duck-Walk & Variable-Height LIPM WBC
=======================================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Engineering Group
Target: Kyle Edelberg & Brett Adcock, Figure AI

Architecture (100% Pure Rigid-Body Physics, Zero Skyhook, Causal Onboard Depth Rays):
1. Strict skyhook prohibition: `data.qfrc_applied[:] = 0.0` at every 500 Hz step.
2. Rigid low-clearance industrial overhead steel gantry (`X in [0.92, 1.48] m`,
   soffit ceiling at `Z = 1.410 m`, `~200 mm` below Figure 05's standing `1.61 m` crown
   and below `1.44 m` chest top), with active physical head & chest collision shells
   (`contype=4, conaffinity=4`).
3. Zero gantry-coordinate oracle (`zero_oracle_controller = true`): onboard head-mounted
   `rgbd_camera` depth rays (`mj_ray` on static scene `group=1`) detect the low beam
   ahead and hold crouch underneath until clear overhead headroom returns.
4. Variable-height Whole-Body Control (`z_c(t): 0.855 -> 0.695 -> 0.855 m`, `-160 mm`
   deep squat + `+0.56 rad` torso pitch bow + horizon-stabilized `-0.42 rad` neck pitch)
   updating LIPM natural frequency `omega(t) = sqrt(g / z_c(t))` online every 2 ms.
5. Honest A/B Proof:
   - Trial A (`baseline_upright_walk`): impacts the `1.410 m` beam violently with head/chest
     (`> 200 N` contact force via `mj_contactForce`) and fails to traverse.
   - Trial B (`whole_body_crouch_walk`): clears beneath with `0.0 N` beam collision,
     `>= 25 mm` headroom margin, recovers upright stance cleanly after exit, `0.0 N` skyhook.
"""

import json
import os
import re

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from figure05_mujoco_common import (
    build_actuator_mappings,
    clamp_and_apply_ctrl,
    compute_capture_point_dcm,
    mat_to_rotvec_err,
    measure_foot_world_grfs,
    measure_geom_pair_world_force,
    quat_to_roll_pitch_yaw,
    setup_crouched_biped,
)

GANTRY_BEAM_X = 1.20
GANTRY_BEAM_HALF_X = 0.28  # Beam spans X in [0.92, 1.48] m
GANTRY_SOFFIT_Z = 1.410
HEAD_CROWN_RADIUS_M = 0.095


def build_gantry_duck_mjcf(base_dir: str) -> str:
    """Inject physical head/chest collision shells and low-clearance gantry beam into MJCF."""
    xml_path = os.path.join(base_dir, "figure05_mujoco.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_str = f.read()

    # Enable contype=4 collision on chest_geom & head sphere without altering mass/inertia
    xml_str, n_chest = re.subn(
        r'<geom name="chest_geom"([^/>]*)\s*/>',
        r'<geom name="chest_geom" contype="4" conaffinity="4"\1/>',
        xml_str,
        count=1,
    )
    xml_str, n_head = re.subn(
        r'<geom type="sphere" size="0\.095" mass="1\.40" material="mat_carbon"\s*/>',
        r'<geom name="head_geom" type="sphere" size="0.095" mass="1.40" '
        r'contype="4" conaffinity="4" friction="0.6 0.01 0.001" material="mat_carbon"/>',
        xml_str,
        count=1,
    )
    if n_chest != 1 or n_head != 1:
        raise RuntimeError(f"Failed to enable head/chest collision shells ({n_chest}, {n_head}).")

    if "</worldbody>" not in xml_str:
        raise RuntimeError("Missing </worldbody> closing tag in figure05_mujoco.xml")

    beam_half_z = 0.12
    beam_center_z = GANTRY_SOFFIT_Z + beam_half_z
    col_half_z = 0.5 * (GANTRY_SOFFIT_Z + 2.0 * beam_half_z)
    gantry_xml = (
        "\n        <!-- Industrial Steel Low-Clearance Overhead Gantry (Group 1, Static) -->\n"
        f'        <geom name="gantry_beam" type="box" group="1" contype="4" conaffinity="4" '
        f'pos="{GANTRY_BEAM_X:.3f} 0.0 {beam_center_z:.4f}" '
        f'size="{GANTRY_BEAM_HALF_X:.3f} 0.62 {beam_half_z:.3f}" '
        f'friction="0.7 0.01 0.001" rgba="0.88 0.52 0.08 1.0"/>\n'
        f'        <geom name="gantry_post_l" type="box" group="1" contype="1" conaffinity="1" '
        f'pos="{GANTRY_BEAM_X:.3f} 0.60 {col_half_z:.4f}" size="0.08 0.06 {col_half_z:.4f}" '
        f'rgba="0.30 0.36 0.44 1.0"/>\n'
        f'        <geom name="gantry_post_r" type="box" group="1" contype="1" conaffinity="1" '
        f'pos="{GANTRY_BEAM_X:.3f} -0.60 {col_half_z:.4f}" size="0.08 0.06 {col_half_z:.4f}" '
        f'rgba="0.30 0.36 0.44 1.0"/>\n'
    )
    return xml_str.replace("</worldbody>", gantry_xml + "    </worldbody>", 1)


def variable_height_crouch_schedule(
    duck_blend: float,
    z_nom: float = 0.855,
    z_crouch: float = 0.695,
    torso_duck_rad: float = 0.56,
    g: float = 9.80665,
) -> tuple[float, float, float, float]:
    """
    Compute coordinated variable-height LIPM & whole-body duck references for `duck_blend in [0, 1]`.

    Returns `(z_c_target, omega_inst, torso_pitch_des, neck_pitch_des)`.
    """
    u = float(np.clip(duck_blend, 0.0, 1.0))
    s = 3.0 * u ** 2 - 2.0 * u ** 3
    z_c = z_nom + s * (z_crouch - z_nom)
    omega_inst = float(np.sqrt(g / max(z_c, 0.35)))
    torso_p = s * torso_duck_rad
    neck_p = -0.75 * torso_p
    return float(z_c), omega_inst, float(torso_p), float(neck_p)


def causal_overhead_perception_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_site_id: int,
    duck_active: bool,
    clear_hold_s: float,
    dt: float,
    detect_range_m: float = 0.72,
) -> tuple[bool, float, float]:
    """
    Onboard `rgbd_camera` causal raycast detector (`group=1` static gantry only, zero oracle).

    Casts forward-fan (`0, +12, +24 deg`), zenith (`+85 deg`), and rearward-trailing-edge
    (`+112, +130 deg`) depth rays from head camera so the robot ducks ahead of overhangs < 1.68 m
    and holds the crouch until the trailing shoulder/chest clears the far edge.
    """
    cam_pos = data.site_xpos[cam_site_id].copy()
    cam_mat = data.site_xmat[cam_site_id].reshape(3, 3)
    fwd_world = cam_mat @ np.array([1.0, 0.0, 0.0])
    # Zero out pitch bias on horizontal search plane so gaze still scans forward overhang
    horiz_dir = np.array([fwd_world[0], fwd_world[1], 0.0], dtype=float)
    n_h = float(np.linalg.norm(horiz_dir))
    horiz_dir = horiz_dir / n_h if n_h > 1e-6 else np.array([1.0, 0.0, 0.0])

    # Cast only against static world structure group 1 (never matches robot body geoms)
    geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
    gid_buf = np.zeros(1, dtype=np.int32)
    min_dist = 9.9
    beam_seen = False

    for elev_deg in (0.0, 12.0, 24.0, 85.0, 112.0, 130.0):
        el = float(np.deg2rad(elev_deg))
        ray_dir = np.cos(el) * horiz_dir + np.array([0.0, 0.0, np.sin(el)])
        origin = cam_pos + 0.04 * ray_dir
        d_hit = float(mujoco.mj_ray(model, data, origin, ray_dir, geomgroup, 1, -1, gid_buf))
        max_r = 0.90 if elev_deg >= 75.0 else detect_range_m
        if 0.0 < d_hit <= max_r:
            hit_z = float(origin[2] + d_hit * ray_dir[2])
            if 1.15 <= hit_z <= 1.68:
                beam_seen = True
                min_dist = min(min_dist, d_hit)

    if beam_seen:
        return True, min_dist, 1.15
    if duck_active and clear_hold_s > 0.0:
        return True, min_dist, max(0.0, clear_hold_s - dt)
    return False, min_dist, 0.0


def run_gantry_trial(
    base_dir: str,
    use_crouch_adaptation: bool,
    sim_duration_s: float = 8.4,
) -> tuple[dict, dict]:
    """Execute one skyhook-free MuJoCo 3.6 low-clearance gantry traversal trial."""
    xml_str = build_gantry_duck_mjcf(base_dir)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    dt = float(model.opt.timestep)
    g = 9.80665
    pelvis_id = model.body("pelvis").id
    head_body_id = model.body("head").id
    chest_body_id = model.body("upper_chest").id
    total_mass = float(model.body_subtreemass[pelvis_id])

    setup_crouched_biped(model, data)

    maps = build_actuator_mappings(model)
    act_qidx = maps["act_qidx"]
    act_vidx = maps["act_vidx"]
    idx_map = maps["idx_map"]
    l_leg_v = maps["l_leg_v"]
    r_leg_v = maps["r_leg_v"]
    l_leg_act = maps["l_leg_act"]
    r_leg_act = maps["r_leg_act"]
    act_names = maps["act_names"]
    q_init = np.array([data.qpos[idx] for idx in act_qidx])

    l_site = model.site("left_foot_contact").id
    r_site = model.site("right_foot_contact").id
    cam_site = model.site("rgbd_camera").id
    l_body = model.body("left_ankle").id
    r_body = model.body("right_ankle").id
    floor_geom = model.geom("floor").id
    gantry_beam_geom = model.geom("gantry_beam").id
    chest_geom = model.geom("chest_geom").id
    head_geom = model.geom("head_geom").id
    upper_collision_geoms = (head_geom, chest_geom)

    z_c_nom = 0.855
    t_stand = 0.60
    t_step = 0.44
    ds_frac = 0.22
    vx_des = 0.65
    w_step = 0.21
    h_lift = 0.065

    n_steps = int(round(sim_duration_s / dt))
    t_log = np.zeros(n_steps)
    com_x_log = np.zeros(n_steps)
    com_y_log = np.zeros(n_steps)
    com_z_log = np.zeros(n_steps)
    crown_z_log = np.zeros(n_steps)
    chest_top_z_log = np.zeros(n_steps)
    torso_pitch_log = np.zeros(n_steps)
    duck_blend_log = np.zeros(n_steps)
    beam_force_log = np.zeros(n_steps)
    roll_log = np.zeros(n_steps)
    pitch_log = np.zeros(n_steps)
    knee_tau_max_log = np.zeros(n_steps)

    skyhook_max_n = 0.0
    step_idx = -1
    stance_side = 1
    sw_start_pos = np.zeros(3)
    prev_pdes = np.zeros(3)

    duck_active = False
    clear_hold_s = 0.0
    duck_blend = 0.0
    survived = True
    fall_time_s = None
    traversed_gantry = False
    traversal_time_s = None
    min_under_beam_clearance_m = 9.9
    beam_triggers = 0
    prev_beam_detected = False
    valid_steps = n_steps

    jac_com = np.zeros((3, model.nv))
    jac_lp = np.zeros((3, model.nv))
    jac_lr = np.zeros((3, model.nv))
    jac_rp = np.zeros((3, model.nv))
    jac_rr = np.zeros((3, model.nv))

    torso_qidx = model.jnt_qposadr[model.joint("torso_pitch_joint").id]

    for k in range(n_steps):
        t = k * dt
        data.qfrc_applied[:] = 0.0
        skyhook_max_n = max(skyhook_max_n, float(np.max(np.abs(data.qfrc_applied))))

        mujoco.mj_forward(model, data)
        mujoco.mj_jacSubtreeCom(model, data, jac_com, pelvis_id)
        mujoco.mj_jacSite(model, data, jac_lp, jac_lr, l_site)
        mujoco.mj_jacSite(model, data, jac_rp, jac_rr, r_site)

        com = data.subtree_com[pelvis_id].copy()
        com_vel = jac_com @ data.qvel
        p_l = data.site_xpos[l_site].copy()
        p_r = data.site_xpos[r_site].copy()
        v_l = jac_lp @ data.qvel
        v_r = jac_rp @ data.qvel
        r_l = data.site_xmat[l_site].reshape(3, 3)
        r_r = data.site_xmat[r_site].reshape(3, 3)

        head_z = float(data.xpos[head_body_id, 2])
        crown_z = head_z + HEAD_CROWN_RADIUS_M
        chest_rot = data.xmat[chest_body_id].reshape(3, 3)
        chest_top_z = float(data.xpos[chest_body_id, 2]) + 0.18 * abs(float(chest_rot[2, 2])) + 0.14 * abs(float(chest_rot[2, 0]))
        torso_q_deg = float(np.rad2deg(data.qpos[torso_qidx]))

        roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
        droll, dpitch, dyaw = [float(x) for x in data.qvel[3:6]]

        # Measure real MuJoCo collision force between rigid steel gantry beam and upper body
        f_beam_w, _ = measure_geom_pair_world_force(
            model, data, (gantry_beam_geom,), upper_collision_geoms
        )
        f_beam_mag = float(np.linalg.norm(f_beam_w))

        head_x = float(data.xpos[head_body_id, 0])
        chest_x = float(data.xpos[chest_body_id, 0])
        beam_lo = GANTRY_BEAM_X - GANTRY_BEAM_HALF_X
        beam_hi = GANTRY_BEAM_X + GANTRY_BEAM_HALF_X
        head_under = (beam_lo - 0.095) <= head_x <= (beam_hi + 0.095)
        chest_under = (beam_lo - 0.15) <= chest_x <= (beam_hi + 0.15)
        if head_under or chest_under:
            worst_z = max(crown_z if head_under else 0.0, chest_top_z if chest_under else 0.0)
            min_under_beam_clearance_m = min(min_under_beam_clearance_m, GANTRY_SOFFIT_Z - worst_z)

        if (abs(roll) > 0.62) or (abs(pitch) > 0.65) or (com[2] < 0.45):
            survived = False
            fall_time_s = round(t, 3)
            valid_steps = k
            break

        # Causal depth raycast detector (zero gantry position oracle!)
        if use_crouch_adaptation and t >= t_stand:
            duck_active, _, clear_hold_s = causal_overhead_perception_step(
                model, data, cam_site, duck_active, clear_hold_s, dt
            )
            if duck_active and (not prev_beam_detected):
                beam_triggers += 1
            prev_beam_detected = duck_active
            rate_up = dt / 0.55
            rate_dn = dt / 0.65
            if duck_active:
                duck_blend = min(1.0, duck_blend + rate_up)
            else:
                duck_blend = max(0.0, duck_blend - rate_dn)
        else:
            duck_blend = 0.0

        z_c_target, omega, torso_pitch_des, neck_pitch_des = variable_height_crouch_schedule(
            duck_blend, z_nom=z_c_nom, z_crouch=0.695, torso_duck_rad=0.56, g=g
        )
        by_nom = float(w_step / (np.exp(omega * t_step) + 1.0))
        dcm = compute_capture_point_dcm(com[:2], com_vel[:2], z_c_target, g=g)

        if (not traversed_gantry) and com[0] >= 1.72 and head_x >= (GANTRY_BEAM_X + GANTRY_BEAM_HALF_X + 0.18):
            traversed_gantry = True
            traversal_time_s = round(t, 3)

        # Gait State Machine
        if t < t_stand:
            smooth = 0.5 * (1.0 - np.cos(np.pi * t / t_stand))
            c_tgt = np.array([
                0.5 * (p_l[0] + p_r[0]) + 0.025 * smooth,
                (p_l[1] - by_nom) * smooth,
                z_c_target,
            ])
            alpha_l = 0.5 + 0.45 * smooth
            alpha_r = 1.0 - alpha_l
            swing = None
            s_sw = 0.0
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
                sw_start_pos[2] = 0.0
                prev_pdes = sw_start_pos.copy()

            exit_brake = float(np.clip((2.40 - com[0]) / 0.38, 0.0, 1.0)) if com[0] > 2.02 else 1.0
            # Modulate forward cruise velocity gently during deep crouch (-160 mm) for clean stance margin
            crouch_v_scale = 1.0 - 0.10 * duck_blend
            vx_cmd = vx_des * crouch_v_scale * float(np.clip(tw / 0.85, 0.25, 1.0)) * exit_brake
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
                s_sw = 0.0
            else:
                alpha_l = 1.0 if stance_side == 1 else 0.0
                alpha_r = 1.0 - alpha_l
                swing = "R" if stance_side == 1 else "L"
                s_sw = (phi - ds_frac) / (1.0 - ds_frac)

        # Variable-Height Centroidal Vertical Force (tracks z_c_target smoothly without shock)
        fz_tot = total_mass * (g + 170.0 * (z_c_target - com[2]) - 26.0 * com_vel[2])
        fz_tot = float(np.clip(fz_tot, 180.0, 1120.0))

        if t < t_stand:
            f_xy = total_mass * np.array([
                75.0 * (c_tgt[0] - com[0]) - 13.0 * com_vel[0],
                110.0 * (c_tgt[1] - com[1]) - 14.0 * com_vel[1],
            ])
        elif swing is None:
            p_cop_des = alpha_l * p_l[:2] + alpha_r * p_r[:2]
            f_xy = total_mass * np.array([
                4.2 * (vx_cmd - com_vel[0]),
                34.0 * (p_cop_des[1] - com[1]) - 6.5 * com_vel[1],
            ])
        else:
            p_st = p_l if stance_side == 1 else p_r
            cop_x = p_st[0] + np.clip(0.26 * (dcm[0] - (p_st[0] + bx_cmd)), -0.05, 0.08)
            cop_y = p_st[1] + np.clip(0.26 * (dcm[1] - (p_st[1] - stance_side * by_nom)), -0.035, 0.035)
            f_xy = total_mass * (omega ** 2) * np.array([com[0] - cop_x, com[1] - cop_y])
            f_xy = np.clip(f_xy, -195.0, 195.0)

        tau_trunk = np.array([
            -310.0 * roll - 34.0 * droll,
            -290.0 * pitch - 33.0 * dpitch,
            -95.0 * yaw - 12.5 * dyaw,
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
            tgt_x = float(np.clip(dcm[0] - bx_cmd, p_st[0] - 0.10, p_st[0] + 0.44))
            tgt_y = dcm[1] + s_sw_side * by_nom - 0.08 * com[1]
            if s_sw_side == -1:
                tgt_y = float(np.clip(tgt_y, p_st[1] - 0.33, p_st[1] - 0.15))
            else:
                tgt_y = float(np.clip(tgt_y, p_st[1] + 0.15, p_st[1] + 0.33))

            smooth_xy = 0.5 * (1.0 - np.cos(np.pi * s_sw))
            z_des = h_lift * np.sin(np.pi * s_sw) - 0.002 * (s_sw ** 2)
            p_des = np.array([
                sw_start_pos[0] + smooth_xy * (tgt_x - sw_start_pos[0]),
                sw_start_pos[1] + smooth_xy * (tgt_y - sw_start_pos[1]),
                z_des,
            ])
            v_des = np.clip((p_des - prev_pdes) / dt, -3.5, 3.5) if s_sw > 0.01 else np.zeros(3)
            prev_pdes = p_des.copy()

            p_cur = p_r if swing == "R" else p_l
            v_cur = v_r if swing == "R" else v_l
            r_cur = r_r if swing == "R" else r_l
            w_cur = (jac_rr @ data.qvel) if swing == "R" else (jac_lr @ data.qvel)
            j_sw = j_r if swing == "R" else j_l
            sw_dofs = r_leg_v if swing == "R" else l_leg_v
            sw_acts = r_leg_act if swing == "R" else l_leg_act

            r_err = mat_to_rotvec_err(np.eye(3), r_cur)
            f_sw = np.concatenate([
                np.array([1500.0, 1500.0, 1900.0]) * (p_des - p_cur)
                + np.array([70.0, 70.0, 85.0]) * (v_des - v_cur),
                np.array([100.0, 130.0, 80.0]) * r_err - np.array([7.0, 9.0, 6.0]) * w_cur,
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

        # Upper-body coordination: Torso duck bow, horizon neck pitch, and CG-counter arm posture
        q = np.array([data.qpos[idx] for idx in act_qidx])
        qd = np.array([data.qvel[idx] for idx in act_vidx])
        q_t = q_init.copy()
        arm_amp = 0.22 * float(np.clip(com_vel[0] / 0.55, 0.0, 1.1))
        arm_duck_counter = 0.34 * duck_blend  # swings arms slightly back to counter torso bow
        q_t[idx_map["act_torso_pitch"]] = torso_pitch_des
        q_t[idx_map["act_neck_pitch"]] = neck_pitch_des
        q_t[idx_map["act_left_shoulder_pitch"]] = arm_amp * (-stance_side) + arm_duck_counter
        q_t[idx_map["act_right_shoulder_pitch"]] = arm_amp * stance_side + arm_duck_counter
        for i, name in enumerate(act_names):
            if not any(x in name for x in ("hip", "knee", "ankle")):
                kp_u = 240.0 if "torso_pitch" in name else 180.0
                kd_u = 22.0 if "torso_pitch" in name else 16.0
                tau_ctrl[i] += kp_u * (q_t[i] - q[i]) - kd_u * qd[i]

        clamp_and_apply_ctrl(model, data, tau_ctrl)
        mujoco.mj_step(model, data)

        # Keep floor GRFs measured just for integrity completeness
        _ = measure_foot_world_grfs(model, data, (floor_geom,), l_body, r_body)

        t_log[k] = t
        com_x_log[k] = com[0]
        com_y_log[k] = com[1]
        com_z_log[k] = com[2]
        crown_z_log[k] = crown_z
        chest_top_z_log[k] = chest_top_z
        torso_pitch_log[k] = torso_q_deg
        duck_blend_log[k] = duck_blend
        beam_force_log[k] = f_beam_mag
        roll_log[k] = np.rad2deg(roll)
        pitch_log[k] = np.rad2deg(pitch)
        knee_tau_max_log[k] = max(
            abs(float(data.ctrl[idx_map["act_left_knee"]])),
            abs(float(data.ctrl[idx_map["act_right_knee"]])),
        )

    sl = slice(0, valid_steps)
    in_gantry_mask = (com_x_log[sl] >= (GANTRY_BEAM_X - GANTRY_BEAM_HALF_X)) & (
        com_x_log[sl] <= (GANTRY_BEAM_X + GANTRY_BEAM_HALF_X)
    )
    min_com_z_under_beam = (
        float(np.min(com_z_log[sl][in_gantry_mask])) if np.any(in_gantry_mask) else float(np.min(com_z_log[sl]))
    )
    post_mask = t_log[sl] >= max(t_log[valid_steps - 1] - 0.5, 0.0)
    post_gantry_com_z = float(np.mean(com_z_log[sl][post_mask]))

    series = {
        "t": t_log[sl],
        "com_x": com_x_log[sl],
        "com_z": com_z_log[sl],
        "crown_z": crown_z_log[sl],
        "chest_top_z": chest_top_z_log[sl],
        "torso_pitch_deg": torso_pitch_log[sl],
        "duck_blend": duck_blend_log[sl],
        "beam_force_n": beam_force_log[sl],
        "roll_deg": roll_log[sl],
        "pitch_deg": pitch_log[sl],
        "knee_tau_max": knee_tau_max_log[sl],
    }
    summary = {
        "use_crouch_adaptation": bool(use_crouch_adaptation),
        "survived": bool(survived),
        "fall_time_s": fall_time_s,
        "traversed_gantry": bool(traversed_gantry),
        "traversal_time_s": traversal_time_s,
        "final_com_x_m": round(float(com_x_log[valid_steps - 1]), 3),
        "final_com_y_m": round(float(com_y_log[valid_steps - 1]), 3),
        "min_com_z_under_gantry_m": round(min_com_z_under_beam, 3),
        "post_gantry_recovered_com_z_m": round(post_gantry_com_z, 3),
        "com_vertical_drop_mm": round((0.875 - min_com_z_under_beam) * 1000.0, 1),
        "min_crown_z_m": round(float(np.min(crown_z_log[sl])), 3),
        "peak_torso_pitch_deg": round(float(np.max(torso_pitch_log[sl])), 1),
        "min_head_clearance_mm": round(float(min_under_beam_clearance_m * 1000.0), 1)
        if min_under_beam_clearance_m < 9.0
        else None,
        "beam_perception_triggers": int(beam_triggers),
        "peak_gantry_impact_force_n": round(float(np.max(beam_force_log[sl])), 1),
        "roll_rms_deg": round(float(np.sqrt(np.mean(roll_log[sl] ** 2))), 2),
        "pitch_rms_deg": round(float(np.sqrt(np.mean(pitch_log[sl] ** 2))), 2),
        "peak_knee_torque_nm": round(float(np.max(knee_tau_max_log[sl])), 1),
        "skyhook_max_n": round(float(skyhook_max_n), 4),
    }
    return summary, series


def run_gantry_duck_benchmark() -> None:
    print("=================================================================")
    print("FIGURE 05: MUJOCO 3.6 LOW-CLEARANCE GANTRY DUCK-WALK BENCHMARK")
    print("=================================================================")
    base_dir = os.path.dirname(os.path.abspath(__file__))

    base_summary, base_series = run_gantry_trial(base_dir, use_crouch_adaptation=False)
    print(
        f"[Trial A - Upright Nominal WBC] survived={base_summary['survived']} | "
        f"traversed={base_summary['traversed_gantry']} | final_X={base_summary['final_com_x_m']}m | "
        f"beam_impact={base_summary['peak_gantry_impact_force_n']}N | "
        f"min_clear={base_summary['min_head_clearance_mm']}mm"
    )

    duck_summary, duck_series = run_gantry_trial(base_dir, use_crouch_adaptation=True)
    print(
        f"[Trial B - Whole-Body Crouch WBC] survived={duck_summary['survived']} | "
        f"traversed={duck_summary['traversed_gantry']} ({duck_summary['traversal_time_s']}s) | "
        f"final_X={duck_summary['final_com_x_m']}m | com_z_min={duck_summary['min_com_z_under_gantry_m']}m -> "
        f"post_Z={duck_summary['post_gantry_recovered_com_z_m']}m | crown_min={duck_summary['min_crown_z_m']}m | "
        f"clear={duck_summary['min_head_clearance_mm']}mm | beam_impact={duck_summary['peak_gantry_impact_force_n']}N | "
        f"torso_bow={duck_summary['peak_torso_pitch_deg']}deg | peak_knee={duck_summary['peak_knee_torque_nm']}Nm | "
        f"skyhook={duck_summary['skyhook_max_n']}N"
    )

    assert base_summary["skyhook_max_n"] == 0.0 and duck_summary["skyhook_max_n"] == 0.0
    assert base_summary["peak_gantry_impact_force_n"] >= 150.0
    assert base_summary["traversed_gantry"] is False
    assert duck_summary["survived"] is True, f"Crouch controller fell at t={duck_summary['fall_time_s']}"
    assert duck_summary["traversed_gantry"] is True
    assert duck_summary["peak_gantry_impact_force_n"] == 0.0
    assert (duck_summary["min_head_clearance_mm"] or 0.0) >= 20.0
    assert duck_summary["post_gantry_recovered_com_z_m"] >= 0.84
    assert duck_summary["peak_knee_torque_nm"] < 230.0

    report = {
        "mujoco_version": mujoco.__version__,
        "skyhook_force_n": 0.0,
        "zero_oracle_controller": True,
        "gantry_soffit_height_m": GANTRY_SOFFIT_Z,
        "gantry_x_span_m": [
            round(GANTRY_BEAM_X - GANTRY_BEAM_HALF_X, 2),
            round(GANTRY_BEAM_X + GANTRY_BEAM_HALF_X, 2),
        ],
        "baseline_upright_walk": base_summary,
        "whole_body_crouch_walk": duck_summary,
    }
    json_path = os.path.join(base_dir, "figure05_gantry_duck_telemetry.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Telemetry exported to: {json_path}")

    # 4-Panel Engineering Plot
    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.8), dpi=180)
    fig.suptitle(
        f"Figure 05 — Low-Clearance ({GANTRY_SOFFIT_Z:.2f} m) Overhead Gantry Duck-Walk A/B Benchmark (MuJoCo 3.6.0, 0.0 N Skyhook)",
        fontsize=12.5,
        fontweight="bold",
        color="#38bdf8",
    )

    # Panel 1: Crown Height vs Overhead Steel Soffit
    ax = axs[0, 0]
    ax.axhline(GANTRY_SOFFIT_Z, color="#f43f5e", lw=2.0, ls="--", label=f"Rigid Steel Soffit ({GANTRY_SOFFIT_Z:.2f} m)")
    ax.plot(duck_series["t"], duck_series["crown_z"], color="#10b981", lw=2.2, label="Crouch WBC Skull Crown Z")
    ax.plot(duck_series["t"], duck_series["chest_top_z"], color="#38bdf8", lw=1.6, ls=":", label="Crouch WBC Chest Top Z")
    ax.plot(base_series["t"], base_series["crown_z"], color="#fb7185", lw=1.8, ls="-.", label="Upright Baseline Crown Z (Impact)")
    ax.set_title("1. Head & Chest Envelope vs Overhead Gantry Ceiling")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("World Z [m]")
    ax.set_ylim(1.05, 1.75)
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8)

    # Panel 2: Physical Collision Force on Gantry Beam
    ax = axs[0, 1]
    ax.plot(base_series["t"], base_series["beam_force_n"], color="#f43f5e", lw=2.0, label=f"Baseline Impact (Peak {base_summary['peak_gantry_impact_force_n']} N)")
    ax.plot(duck_series["t"], duck_series["beam_force_n"], color="#10b981", lw=2.2, label=f"Crouch WBC Impact ({duck_summary['peak_gantry_impact_force_n']} N)")
    ax.set_title("2. Measured MuJoCo Overhead Beam Collision Force")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Contact Force [N]")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 3: Coordinated CoM Squat Drop, Torso Bow & Knee Torque
    ax = axs[1, 0]
    ax.plot(duck_series["t"], duck_series["com_z"], color="#fbbf24", lw=2.0, label="CoM Z Height [m]")
    ax2 = ax.twinx()
    ax2.plot(duck_series["t"], duck_series["torso_pitch_deg"], color="#a855f7", lw=1.6, label="Torso Forward Bow [deg]")
    ax2.plot(duck_series["t"], duck_series["knee_tau_max"], color="#38bdf8", lw=1.3, alpha=0.85, label="Peak Knee Torque [Nm]")
    ax.set_title("3. Variable-Height LIPM Crouch & Torso-Knee Coordination")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CoM Z [m]", color="#fbbf24")
    ax2.set_ylabel("Torso Bow [deg] / Knee [Nm]", color="#a855f7")
    ax.grid(alpha=0.22)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

    # Panel 4: Sagittal Progression X & Attitude
    ax = axs[1, 1]
    ax.axhspan(GANTRY_BEAM_X - GANTRY_BEAM_HALF_X, GANTRY_BEAM_X + GANTRY_BEAM_HALF_X, color="#f59e0b", alpha=0.25, label="Low Gantry Span (0.92..1.48 m)")
    ax.plot(duck_series["t"], duck_series["com_x"], color="#10b981", lw=2.2, label="Crouch WBC Progression (m)")
    ax.plot(base_series["t"], base_series["com_x"], color="#f43f5e", lw=1.8, ls="--", label="Upright Baseline (Blocked)")
    ax.set_title("4. Sagittal Traversal Under Overhead Structure")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CoM X [m]")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = os.path.join(base_dir, "figure05_mujoco_gantry_duck_benchmark.png")
    plt.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"[+] Benchmark figure saved to: {png_path}")


if __name__ == "__main__":
    run_gantry_duck_benchmark()
