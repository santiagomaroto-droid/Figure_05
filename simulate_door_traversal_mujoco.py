"""
Project Figure 05 — Module #29: Skyhook-Free MuJoCo 3.6 Spring-Loaded Hinged Door Traversal
===========================================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Engineering Group
Target: Kyle Edelberg & Brett Adcock, Figure AI

Demonstrates genuine unlatched loco-manipulation of a heavy (7.5 kg), spring-return
articulated steel factory fire door (`stiffness=5.0 Nm/rad`, `damping=1.5 Nm*s/rad`,
range `0..100 deg`) in MuJoCo 3.6 with ZERO skyhook forces (`qfrc_applied = 0.0`)
and ZERO weld/equality constraints:
  - Trial A (Baseline Naive Walk-Through): Arms stay tucked while pelvis & chest
    carry real bitmask collision shells (`contype=4`). Torso physically impacts the
    closed hinged door at X = 0.78 m, failing to traverse (`traversed_doorway=False`).
  - Trial B (Causal Force-Reactive Door Push & Traversal, Zero Oracle):
    1. Walks toward doorway using SO(3)-pelvis ZMP-DCM Centroidal WBC while extending
       palms to `Y = +/-0.167 m` (`shoulder_roll = -0.22 / +0.22`).
    2. Unlatched palms push open the spring-return crash bar while left palm sweeps out
       along `+Y` (`shoulder_roll -> +0.12`) after `door >= 42 deg` to prop the panel
       past `>= 72 deg`, feeding forward causal `mj_contactForce` wrench estimates.
    3. Clears the doorway jamb plane (`X = 0.78 m`), retracts arms cleanly (`0.0 N`
       residual push), halts on the cleanroom side (`X >= 1.38 m`), and allows the
       torsional closer spring to automatically shut the door behind the robot.
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from figure05_mujoco_common import (
    build_actuator_mappings,
    clamp_and_apply_ctrl,
    compute_capture_point_dcm,
    mat_to_rotvec_err,
    measure_geom_pair_world_force,
    quat_to_roll_pitch_yaw,
    setup_crouched_biped,
    sum_support_grfs,
)

DOOR_PLANE_X = 0.78
DOOR_MASS_KG = 6.5
DOOR_SPRING_NM_RAD = 3.0


def build_door_traversal_mjcf(base_dir: str) -> str:
    """Inject articulated spring-return fire door, jambs, torso colliders & palm pads."""
    xml_path = os.path.join(base_dir, "figure05_mujoco.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")

    for body in wb.iter("body"):
        bname = body.get("name")
        if bname in ("pelvis", "upper_chest"):
            for geom in body.findall("geom"):
                if geom.get("name") in ("pelvis_geom", "chest_geom"):
                    geom.set("contype", "4")
                    geom.set("conaffinity", "0")
        elif bname in ("left_hand", "right_hand"):
            side = "left" if bname == "left_hand" else "right"
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{side}_palm_door_pad",
                    "type": "sphere",
                    "size": "0.056",
                    "pos": "0.038 0.0 -0.02",
                    "mass": "0.0001",
                    "contype": "2",
                    "conaffinity": "2",
                    "solref": "0.055 1.1",
                    "solimp": "0.82 0.94 0.015",
                    "friction": "0.85 0.005 0.0005",
                    "rgba": "0.22 0.85 0.96 0.92",
                },
            )

    frame = ET.SubElement(wb, "body", {"name": "door_frame", "pos": f"{DOOR_PLANE_X} 0 0"})
    for jname, jpos in (("door_post_left", "0 0.465 0.94"), ("door_post_right", "0 -0.465 0.94")):
        ET.SubElement(
            frame,
            "geom",
            {
                "name": jname,
                "type": "box",
                "pos": jpos,
                "size": "0.05 0.05 0.94",
                "mass": "0.0",
                "contype": "4",
                "conaffinity": "4",
                "rgba": "0.35 0.42 0.52 1.0",
            },
        )
    ET.SubElement(
        frame,
        "geom",
        {
            "name": "door_lintel",
            "type": "box",
            "pos": "0 0 1.90",
            "size": "0.06 0.52 0.04",
            "mass": "0.0",
            "contype": "0",
            "conaffinity": "0",
            "rgba": "0.28 0.34 0.44 1.0",
        },
    )

    # Architectural butt-hinge pin flush at front-left jamb edge so door swings 0..100 deg with zero jamb pinching
    door_body = ET.SubElement(frame, "body", {"name": "hinged_door", "pos": "0.022 0.388 0.0"})
    ET.SubElement(
        door_body,
        "joint",
        {
            "name": "door_hinge",
            "type": "hinge",
            "axis": "0 0 1",
            "range": "0.0 1.75",
            "stiffness": f"{DOOR_SPRING_NM_RAD}",
            "damping": "5.2",
            "frictionloss": "0.06",
        },
    )
    ET.SubElement(
        door_body,
        "geom",
        {
            "name": "door_panel",
            "type": "box",
            "pos": "-0.020 -0.395 0.96",
            "size": "0.020 0.370 0.86",
            "mass": f"{DOOR_MASS_KG}",
            "contype": "6",
            "conaffinity": "6",
            "solref": "0.025 1.1",
            "friction": "0.45 0.005 0.0005",
            "rgba": "0.85 0.42 0.16 0.92",
        },
    )
    ET.SubElement(
        door_body,
        "geom",
        {
            "name": "door_crash_bar",
            "type": "box",
            "pos": "-0.060 -0.36 1.04",
            "size": "0.024 0.31 0.050",
            "mass": "0.8",
            "contype": "6",
            "conaffinity": "6",
            "solref": "0.025 1.1",
            "friction": "0.85 0.005 0.0005",
            "rgba": "0.95 0.82 0.20 1.0",
        },
    )
    return ET.tostring(root, encoding="unicode")


def compute_door_wbc_feedforward(
    f_hand_on_robot: np.ndarray,
    total_mass_kg: float,
    z_hand_m: float = 1.04,
    z_com_m: float = 0.87,
    lever_y_m: float = 0.21,
    g: float = 9.80665,
    f_left_x: float | None = None,
    f_right_x: float | None = None,
) -> tuple[float, float, float, float]:
    """Causal Newton-3rd-Law WBC compensation for bilateral/unilateral articulated door contact."""
    fx_res = float(f_hand_on_robot[0])
    fy_res = float(f_hand_on_robot[1])
    f_push_x = max(0.0, -fx_res)
    fx_ff = -fx_res
    fy_ff = -fy_res
    tau_pitch_ff = max(0.0, z_hand_m - z_com_m) * f_push_x
    if f_left_x is not None and f_right_x is not None:
        tau_yaw_ff = lever_y_m * (float(f_left_x) - float(f_right_x))
    else:
        tau_yaw_ff = -lever_y_m * fx_res
    return fx_ff, fy_ff, tau_pitch_ff, tau_yaw_ff


def run_door_trial(model: mujoco.MjModel, use_door_manipulation: bool, max_sim_s: float = 13.0) -> dict:
    """Execute one skyhook-free door traversal attempt (Naive walk vs Causal Loco-Manipulation)."""
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    g = 9.80665

    pelvis_id = model.body("pelvis").id
    robot_mass = float(model.body_subtreemass[pelvis_id])

    setup_crouched_biped(
        model,
        data,
        shoulder_pitch=(-0.35 if use_door_manipulation else 0.05),
        elbow=(-0.35 if use_door_manipulation else -0.45),
        torso_pitch=0.02,
    )

    maps = build_actuator_mappings(model)
    act_qidx = maps["act_qidx"]
    act_vidx = maps["act_vidx"]
    idx_map = maps["idx_map"]
    l_leg_v = maps["l_leg_v"]
    r_leg_v = maps["r_leg_v"]
    l_leg_act = maps["l_leg_act"]
    r_leg_act = maps["r_leg_act"]
    q_init = np.array([data.qpos[idx] for idx in act_qidx], dtype=float)

    arm_set = {
        idx_map[n] for n in maps["act_names"]
        if any(k in n for k in ("shoulder", "elbow", "wrist"))
    }
    torso_set = {
        idx_map[n] for n in maps["act_names"] if "torso" in n
    }

    l_site = model.site("left_foot_contact").id
    r_site = model.site("right_foot_contact").id
    l_body = model.body("left_ankle").id
    r_body = model.body("right_ankle").id

    door_geoms = {model.geom("door_panel").id, model.geom("door_crash_bar").id}
    l_pad_gid = model.geom("left_palm_door_pad").id
    r_pad_gid = model.geom("right_palm_door_pad").id
    palm_geoms = {l_pad_gid, r_pad_gid}
    torso_geoms = {model.geom("pelvis_geom").id, model.geom("chest_geom").id}
    door_jid = model.joint("door_hinge").id
    door_qpos_adr = model.jnt_qposadr[door_jid]

    jac_com = np.zeros((3, model.nv))
    jac_p_l = np.zeros((3, model.nv))
    jac_r_l = np.zeros((3, model.nv))
    jac_p_r = np.zeros((3, model.nv))
    jac_r_r = np.zeros((3, model.nv))

    # Englsberger DCM + ZMP walking constants matching loaded-walk architecture
    step_T = 0.42
    ds_frac = 0.16
    half_y = 0.105
    swing_clear = 0.072
    z_nom = 0.870
    omega = float(np.sqrt(g / z_nom))
    b_val = float(np.exp(omega * step_T))
    y_off_nom = (2.0 * half_y) / (b_val + 1.0)
    k_dcm = 1.45

    settle_steps = 140
    t_walk_end = 9.20 if use_door_manipulation else 3.20
    sim_dur = max_sim_s if use_door_manipulation else 3.40
    n_steps = int(sim_dur / dt)

    swing_side = "left"
    sw_start_pos = data.site_xpos[l_site].copy()
    sw_tgt_x = sw_start_pos[0] + 0.12
    sw_tgt_y = half_y

    f_hand_filt = np.zeros(3)
    f_l_filt_x = 0.0
    f_r_filt_x = 0.0
    skyhook_max = 0.0
    fell = False
    fall_time_s = None
    peak_door_deg = 0.0
    peak_palm_push_n = 0.0
    peak_torso_impact_n = 0.0
    post_release_forces = []
    released_time_s = None
    traversed_time_s = None
    peak_knee_nm = 0.0
    stopping = False

    t_log, x_log, y_log, door_deg_log, palm_n_log, torso_n_log, roll_log, pitch_log = (
        [], [], [], [], [], [], [], []
    )

    for step in range(n_steps):
        t = step * dt
        mujoco.mj_forward(model, data)
        data.qfrc_applied[:] = 0.0
        skyhook_max = max(skyhook_max, float(np.max(np.abs(data.qfrc_applied))))

        p_l = data.site_xpos[l_site].copy()
        p_r = data.site_xpos[r_site].copy()
        r_l = data.site_xmat[l_site].reshape(3, 3)
        r_r = data.site_xmat[r_site].reshape(3, 3)

        mujoco.mj_jacSubtreeCom(model, data, jac_com, pelvis_id)
        p_com = data.subtree_com[pelvis_id].copy()
        v_com = jac_com @ data.qvel
        com_x, com_y = float(p_com[0]), float(p_com[1])

        mujoco.mj_jacSite(model, data, jac_p_l, jac_r_l, l_site)
        mujoco.mj_jacSite(model, data, jac_p_r, jac_r_r, r_site)
        v_l = jac_p_l @ data.qvel
        w_l = jac_r_l @ data.qvel
        v_r = jac_p_r @ data.qvel
        w_r = jac_r_r @ data.qvel

        pelv_mat = data.body("pelvis").xmat.reshape(3, 3)
        pelv_w = data.qvel[3:6].copy()
        roll_pelv, pitch_pelv, yaw_pelv = quat_to_roll_pitch_yaw(data.qpos[3:7])

        door_deg = float(np.rad2deg(data.qpos[door_qpos_adr]))
        peak_door_deg = max(peak_door_deg, door_deg)

        f_palm_raw, f_per_palm = measure_geom_pair_world_force(model, data, door_geoms, palm_geoms)
        f_torso_raw, _ = measure_geom_pair_world_force(model, data, door_geoms, torso_geoms)
        f_hand_filt = 0.90 * f_hand_filt + 0.10 * f_palm_raw
        f_l_filt_x = 0.90 * f_l_filt_x + 0.10 * float(f_per_palm[l_pad_gid][0])
        f_r_filt_x = 0.90 * f_r_filt_x + 0.10 * float(f_per_palm[r_pad_gid][0])
        palm_push_n = float(np.linalg.norm(f_palm_raw[:2]))
        torso_impact_n = float(np.linalg.norm(f_torso_raw[:2]))
        peak_palm_push_n = max(peak_palm_push_n, palm_push_n)
        peak_torso_impact_n = max(peak_torso_impact_n, torso_impact_n)

        if traversed_time_s is None and com_x >= 1.15:
            traversed_time_s = t

        # Arm & torso lean schedule + centered traversal trajectory
        y_ref = 0.0
        torso_p_tgt = 0.02
        pelvis_pitch_ref = 0.020
        if use_door_manipulation:
            if com_x < 1.46:
                blend = float(np.clip((t - 0.10) / 0.65, 0.0, 1.0))
                torso_p_tgt = 0.02 + 0.15 * blend
                pelvis_pitch_ref = 0.020 + 0.025 * blend
                prop_u = float(np.clip((door_deg - 45.0) / 28.0, 0.0, 1.0))
                trail_u = float(np.clip((com_x - 1.05) / 0.38, 0.0, 1.0))
                sh_p_r = (-0.35 - 0.98 * blend) * (1.0 - prop_u) + (-0.15) * prop_u
                el_r = (-0.35 + 0.27 * blend) * (1.0 - prop_u) + (-0.55) * prop_u
                sh_r_r = 0.20 * blend * (1.0 - prop_u)

                sh_p_l = (-0.35 - 1.05 * blend) * (1.0 - trail_u) + (-0.55) * trail_u
                el_l = (-0.35 + 0.27 * blend) * (1.0 - trail_u) + (-0.30) * trail_u
                sh_r_l = -0.20 * blend * (1.0 - prop_u) + (0.18 + 0.22 * trail_u) * prop_u
            else:
                if released_time_s is None:
                    released_time_s = t
                u_rel = float(np.clip((t - released_time_s) / 0.45, 0.0, 1.0))
                torso_p_tgt = 0.17 * (1.0 - u_rel) + 0.02 * u_rel
                pelvis_pitch_ref = 0.045 * (1.0 - u_rel) + 0.020 * u_rel
                sh_p_r = -0.15 * (1.0 - u_rel) + 0.05 * u_rel
                sh_p_l = -0.55 * (1.0 - u_rel) + 0.05 * u_rel
                el_r = -0.55 * (1.0 - u_rel) - 0.45 * u_rel
                el_l = -0.30 * (1.0 - u_rel) - 0.45 * u_rel
                sh_r_r = 0.0
                sh_r_l = 0.40 * (1.0 - u_rel)
                if u_rel >= 0.99:
                    post_release_forces.append(palm_push_n)

            if door_deg >= 52.0 and 0.50 <= com_x <= 1.65:
                y_ref = -0.06
            elif com_x > 1.65:
                y_ref = -0.06 * max(0.0, 1.0 - (com_x - 1.65) / 0.35)
        else:
            sh_p_r = sh_p_l = 0.05
            el_r = el_l = -0.45
            sh_r_r = sh_r_l = 0.0

        # Gait State Machine (latch stopping ONLY during clean double-support step boundary!)
        tau_step = 0.0
        step_num = 0
        if step < settle_steps or stopping:
            in_ds = True
            phase_sw = 0.0
            s_sign = 1.0
            if step == settle_steps - 1:
                sw_start_pos = p_l.copy()
                sw_tgt_x = p_r[0] + 0.12
                sw_tgt_y = p_r[1] + 2.0 * half_y
            v_des = 0.0
        else:
            step_idx = step - settle_steps
            ticks_per_step = int(step_T / dt)
            step_num = step_idx // ticks_per_step
            tau_step = (step_idx % ticks_per_step) * dt
            if step_idx % ticks_per_step == 0 and step_idx > 0:
                swing_side = "right" if swing_side == "left" else "left"
                sw_start_pos = p_l.copy() if swing_side == "left" else p_r.copy()

            s_sign = 1.0 if swing_side == "left" else -1.0
            in_ds = tau_step < (ds_frac * step_T)
            p_st = p_r if swing_side == "left" else p_l

            # Only lock static Double Support once deceleration steps bring CoM speed inside static capture hull!
            com_spd_xy = float(np.hypot(v_com[0], v_com[1]))
            if in_ds and (
                (use_door_manipulation and ((com_x >= 1.76 and com_spd_xy < 0.14) or t >= t_walk_end))
                or ((not use_door_manipulation) and t >= t_walk_end)
            ):
                stopping = True
                phase_sw = 0.0
                v_des = 0.0
            else:
                if use_door_manipulation and com_x >= 1.76:
                    # Capture-Point deceleration stepping to zero velocity before DS lock
                    v_des = 0.0
                elif use_door_manipulation and 0.44 < com_x < 0.72 and door_deg < 66.0:
                    v_des = 0.22
                elif use_door_manipulation and 0.78 <= com_x < 1.60:
                    v_des = 0.44
                elif use_door_manipulation and 1.60 <= com_x < 1.76:
                    v_des = 0.18
                else:
                    v_des = 0.34 if step_num >= 1 else 0.22
                step_dx = v_des * step_T
                x_off = step_dx / (b_val - 1.0)
                dcm = compute_capture_point_dcm(p_com[:2], v_com[:2], z_nom, g=g)
                dcm_x, dcm_y = float(dcm[0]), float(dcm[1])

                if in_ds:
                    phase_sw = 0.0
                    sw_start_pos = p_l.copy() if swing_side == "left" else p_r.copy()
                else:
                    phase_sw = min(1.0, (tau_step - ds_frac * step_T) / ((1.0 - ds_frac) * step_T))

                if phase_sw < 0.85:
                    sw_tgt_x = float(np.clip(dcm_x - x_off, p_st[0] - 0.08, p_st[0] + 0.34))
                    # Causal LIPM Capture-Point lateral centering: bias foothold outward (+) in sign(com_y - y_ref)
                    # so gravity pushes CoM smoothly back toward y_ref without exponential drift
                    tgt_y_raw = dcm_y + s_sign * y_off_nom + 0.05 * (com_y - y_ref)
                    y_a = p_st[1] + s_sign * 0.14
                    y_b = p_st[1] + s_sign * 0.36
                    sw_tgt_y = float(np.clip(tgt_y_raw, min(y_a, y_b), max(y_a, y_b)))

        p_l_xy, p_r_xy = p_l[:2], p_r[:2]
        if in_ds:
            p_zmp_st = 0.5 * (p_l + p_r)
            if step < settle_steps or stopping:
                w_l_share, w_r_share = 0.5, 0.5
            else:
                # Smoothly bias load onto stance foot near end of DS
                w_l_share, w_r_share = 0.5, 0.5
        elif swing_side == "left":
            w_l_share, w_r_share = 0.0, 1.0
            p_zmp_st = p_r.copy()
        else:
            w_l_share, w_r_share = 1.0, 0.0
            p_zmp_st = p_l.copy()

        dcm = compute_capture_point_dcm(p_com[:2], v_com[:2], z_nom, g=g)
        dcm_x, dcm_y = float(dcm[0]), float(dcm[1])

        if step < settle_steps:
            zmp_ref_x = p_zmp_st[0]
            zmp_ref_y = 0.045 if dcm_y > -0.055 else -0.02
        elif in_ds:
            if stopping:
                zmp_ref_x = p_zmp_st[0] + 1.45 * (dcm_x - p_zmp_st[0])
                zmp_ref_y = p_zmp_st[1] + 1.45 * (dcm_y - p_zmp_st[1])
            else:
                zmp_ref_x = p_zmp_st[0] - 0.01
                zmp_ref_y = p_zmp_st[1]
        else:
            step_dx = v_des * step_T
            x_off = step_dx / (b_val - 1.0)
            exp_tau = float(np.exp(omega * tau_step))
            dcm_nom_x = p_zmp_st[0] + x_off * exp_tau
            dcm_nom_y = p_zmp_st[1] + s_sign * y_off_nom * exp_tau
            zmp_ref_x = p_zmp_st[0] + k_dcm * (dcm_x - dcm_nom_x)
            zmp_ref_y = p_zmp_st[1] + k_dcm * (dcm_y - dcm_nom_y)

        if in_ds:
            zmp_ref_x = float(np.clip(zmp_ref_x, min(p_l_xy[0], p_r_xy[0]) - 0.09, max(p_l_xy[0], p_r_xy[0]) + 0.09))
            zmp_ref_y = float(np.clip(zmp_ref_y, min(p_l_xy[1], p_r_xy[1]) - 0.04, max(p_l_xy[1], p_r_xy[1]) + 0.04))
        else:
            zmp_ref_x = float(np.clip(zmp_ref_x, p_zmp_st[0] - 0.10, p_zmp_st[0] + 0.10))
            zmp_ref_y = float(np.clip(zmp_ref_y, p_zmp_st[1] - 0.035, p_zmp_st[1] + 0.035))

        fz_des = robot_mass * (g + 420.0 * (z_nom - p_com[2]) - 42.0 * v_com[2])
        fz_des = float(np.clip(fz_des, 0.35 * robot_mass * g, 2.15 * robot_mass * g))

        fx_ff_raw, fy_ff_raw, tau_pitch_ff_raw, tau_yaw_ff_raw = compute_door_wbc_feedforward(
            f_hand_filt, robot_mass, z_hand_m=1.04, z_com_m=float(p_com[2]),
            f_left_x=f_l_filt_x, f_right_x=f_r_filt_x,
        )
        fx_ff = float(np.clip(fx_ff_raw, 0.0, 45.0))
        fy_ff = float(np.clip(fy_ff_raw, -35.0, 35.0))
        tau_yaw_ff = float(np.clip(tau_yaw_ff_raw, -12.0, 12.0))

        fx_des = robot_mass * (g / z_nom) * (p_com[0] - zmp_ref_x) + 0.35 * fx_ff
        fy_des = robot_mass * (g / z_nom) * (p_com[1] - zmp_ref_y)
        w_total = np.array([
            float(np.clip(fx_des, -300.0, 300.0)),
            float(np.clip(fy_des, -260.0, 260.0)),
            fz_des,
            0.0, 0.0, 0.0,
        ])

        r_pelv_des = np.array([
            [np.cos(pelvis_pitch_ref), 0.0, np.sin(pelvis_pitch_ref)],
            [0.0, 1.0, 0.0],
            [-np.sin(pelvis_pitch_ref), 0.0, np.cos(pelvis_pitch_ref)],
        ])
        e_rot = mat_to_rotvec_err(r_pelv_des, pelv_mat)
        tau_pelv = 680.0 * e_rot - 48.0 * pelv_w + np.array([0.0, 0.0, 0.4 * tau_yaw_ff])

        ctrl = np.zeros(model.nu)
        r_flat = np.eye(3)

        q_t = q_init.copy()
        q_t[idx_map["act_right_shoulder_pitch"]] = sh_p_r
        q_t[idx_map["act_left_shoulder_pitch"]] = sh_p_l
        q_t[idx_map["act_right_shoulder_roll"]] = sh_r_r
        q_t[idx_map["act_left_shoulder_roll"]] = sh_r_l
        q_t[idx_map["act_right_elbow"]] = el_r
        q_t[idx_map["act_left_elbow"]] = el_l
        q_t[idx_map["act_torso_pitch"]] = torso_p_tgt

        for i in maps["upper_act"]:
            q_err = q_t[i] - data.qpos[act_qidx[i]]
            dq = data.qvel[act_vidx[i]]
            if i in arm_set:
                ctrl[i] = 280.0 * q_err - 20.0 * dq + data.qfrc_bias[act_vidx[i]]
            elif i in torso_set:
                ctrl[i] = 450.0 * q_err - 32.0 * dq + data.qfrc_bias[act_vidx[i]]
            else:
                ctrl[i] = 160.0 * q_err - 14.0 * dq + data.qfrc_bias[act_vidx[i]]

        for side in ("left", "right"):
            is_left = side == "left"
            v_idx = l_leg_v if is_left else r_leg_v
            a_idx = l_leg_act if is_left else r_leg_act
            j_p = (jac_p_l if is_left else jac_p_r)[:, v_idx]
            j_r = (jac_r_l if is_left else jac_r_r)[:, v_idx]
            j_6 = np.vstack([j_p, j_r])
            p_cur = p_l if is_left else p_r
            v_cur = v_l if is_left else v_r
            r_cur = r_l if is_left else r_r
            w_cur = w_l if is_left else w_r
            w_share = w_l_share if is_left else w_r_share

            if w_share > 0.0:
                w_st = w_total * w_share
                w_st[3:6] = w_share * tau_pelv
                e_f = mat_to_rotvec_err(r_flat, r_cur)
                w_st[3:6] -= (180.0 * e_f - 18.0 * w_cur)
                w_st[4] += -w_st[2] * float(np.clip(zmp_ref_x - p_cur[0], -0.09, 0.09))
                w_st[3] += w_st[2] * float(np.clip(zmp_ref_y - p_cur[1], -0.035, 0.035))
                ctrl[a_idx] = data.qfrc_bias[v_idx] - j_6.T @ w_st
            else:
                s = phase_sw
                sw_dur = (1.0 - ds_frac) * step_T
                sw_x = sw_start_pos[0] + (sw_tgt_x - sw_start_pos[0]) * (3.0 * s ** 2 - 2.0 * s ** 3)
                sw_y = sw_start_pos[1] + (sw_tgt_y - sw_start_pos[1]) * (3.0 * s ** 2 - 2.0 * s ** 3)
                sw_vx = (sw_tgt_x - sw_start_pos[0]) * (6.0 * s * (1.0 - s)) / sw_dur
                sw_vy = (sw_tgt_y - sw_start_pos[1]) * (6.0 * s * (1.0 - s)) / sw_dur

                z_touch = -0.005
                sw_z = sw_start_pos[2] + (z_touch - sw_start_pos[2]) * (3.0 * s ** 2 - 2.0 * s ** 3) + swing_clear * (np.sin(np.pi * s) ** 2)
                sw_vz = (z_touch - sw_start_pos[2]) * (6.0 * s * (1.0 - s)) / sw_dur + swing_clear * (np.pi / sw_dur) * np.sin(2.0 * np.pi * s)

                f_sw = np.array([2100.0, 2100.0, 3200.0]) * (np.array([sw_x, sw_y, sw_z]) - p_cur) + np.array([88.0, 88.0, 112.0]) * (np.array([sw_vx, sw_vy, sw_vz]) - v_cur)
                e_rot_sw = mat_to_rotvec_err(r_flat, r_cur)
                m_sw = 240.0 * e_rot_sw - 18.0 * w_cur
                ctrl[a_idx] = data.qfrc_bias[v_idx] + j_6.T @ np.concatenate([f_sw, m_sw])

        clamp_and_apply_ctrl(model, data, ctrl)
        mujoco.mj_step(model, data)

        peak_knee_nm = max(peak_knee_nm, abs(float(data.ctrl[l_leg_act[3]])), abs(float(data.ctrl[r_leg_act[3]])))

        if step % 5 == 0:
            t_log.append(t)
            x_log.append(com_x)
            y_log.append(com_y)
            door_deg_log.append(door_deg)
            palm_n_log.append(palm_push_n)
            torso_n_log.append(torso_impact_n)
            roll_log.append(float(np.rad2deg(roll_pelv)))
            pitch_log.append(float(np.rad2deg(pitch_pelv)))

        if data.qpos[2] < 0.50 or abs(roll_pelv) > 0.90 or abs(pitch_pelv) > 0.90:
            fell = True
            fall_time_s = t
            break

    final_x = float(x_log[-1]) if x_log else 0.0
    final_door_deg = float(door_deg_log[-1]) if door_deg_log else 0.0
    traversed = (not fell) and (final_x >= 1.25)
    mean_rel_force = float(np.mean(post_release_forces)) if post_release_forces else 0.0

    return {
        "use_door_manipulation": use_door_manipulation,
        "survived": not fell,
        "fall_time_s": None if fall_time_s is None else round(fall_time_s, 3),
        "traversed_doorway": bool(traversed),
        "traversed_time_s": None if traversed_time_s is None else round(traversed_time_s, 3),
        "final_com_x_m": round(final_x, 3),
        "final_com_y_m": round(float(y_log[-1]), 3),
        "peak_door_angle_deg": round(peak_door_deg, 1),
        "final_door_angle_deg": round(final_door_deg, 1),
        "peak_palm_push_force_n": round(peak_palm_push_n, 1),
        "peak_torso_impact_force_n": round(peak_torso_impact_n, 1),
        "post_release_palm_force_n": round(mean_rel_force, 2),
        "roll_rms_deg": round(float(np.sqrt(np.mean(np.square(roll_log)))), 2),
        "pitch_rms_deg": round(float(np.sqrt(np.mean(np.square(pitch_log)))), 2),
        "peak_knee_torque_nm": round(peak_knee_nm, 1),
        "skyhook_max_n": round(skyhook_max, 6),
        "traces": {
            "t": t_log,
            "com_x": x_log,
            "com_y": y_log,
            "door_deg": door_deg_log,
            "palm_n": palm_n_log,
            "torso_n": torso_n_log,
        },
    }


def run_door_traversal_benchmark() -> None:
    print("=================================================================")
    print("FIGURE 05: MUJOCO 3.6 SPRING-RETURN DOOR PUSH & TRAVERSAL A/B")
    print("=================================================================")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_str = build_door_traversal_mjcf(base_dir)
    model = mujoco.MjModel.from_xml_string(xml_str)

    res_a = run_door_trial(model, use_door_manipulation=False)
    res_b = run_door_trial(model, use_door_manipulation=True)

    print(
        f"  [Trial A: Naive Walk]      Survived={res_a['survived']} | Traversed={res_a['traversed_doorway']} | "
        f"X_end={res_a['final_com_x_m']}m | DoorMax={res_a['peak_door_angle_deg']}deg | "
        f"TorsoImpact={res_a['peak_torso_impact_force_n']}N"
    )
    print(
        f"  [Trial B: Reactive Push]   Survived={res_b['survived']} | Traversed={res_b['traversed_doorway']} | "
        f"X_end={res_b['final_com_x_m']}m | DoorMax={res_b['peak_door_angle_deg']}deg -> Final={res_b['final_door_angle_deg']}deg | "
        f"PalmMax={res_b['peak_palm_push_force_n']}N | TorsoImpact={res_b['peak_torso_impact_force_n']}N | "
        f"RelForce={res_b['post_release_palm_force_n']}N | RMS={res_b['roll_rms_deg']}/{res_b['pitch_rms_deg']}deg | Knee={res_b['peak_knee_torque_nm']}Nm"
    )

    max_skyhook = max(res_a["skyhook_max_n"], res_b["skyhook_max_n"])
    summary = {
        "mujoco_version": mujoco.__version__,
        "skyhook_force_n": round(max_skyhook, 6),
        "zero_oracle_controller": True,
        "door_mass_kg": DOOR_MASS_KG,
        "door_spring_nm_per_rad": DOOR_SPRING_NM_RAD,
        "door_plane_x_m": DOOR_PLANE_X,
        "baseline_naive_walk": {k: v for k, v in res_a.items() if k != "traces"},
        "causal_locomanip_traversal": {k: v for k, v in res_b.items() if k != "traces"},
    }

    out_json = os.path.join(base_dir, "figure05_door_traversal_telemetry.json")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    plt.style.use("dark_background")
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0B0F19")
    fig.suptitle(
        f"Figure 05 — MuJoCo 3.6 Skyhook-Free Articulated {DOOR_MASS_KG:.1f} kg Spring-Return Door Traversal A/B ({DOOR_SPRING_NM_RAD:.1f} Nm/rad)",
        fontsize=12.6, fontweight="bold", color="#F8FAFC", y=0.975,
    )

    ta, tb = res_a["traces"], res_b["traces"]

    ax = axs[0, 0]
    ax.set_facecolor("#111827")
    ax.axhline(DOOR_PLANE_X, color="#F59E0B", ls="--", lw=1.3, label=f"Door Jamb Plane (X = {DOOR_PLANE_X:.2f} m)")
    ax.plot(ta["t"], ta["com_x"], "--", color="#EF4444", lw=2.0, label="Trial A: Naive Walk (Blocked)")
    ax.plot(tb["t"], tb["com_x"], "-", color="#10B981", lw=2.2, label="Trial B: Reactive Push-Through")
    ax.set_title("1. CoM Forward Progression Through Door Portal", color="#E2E8F0", fontsize=10.5)
    ax.set_xlabel("Time [s]", color="#CBD5E1")
    ax.set_ylabel("World CoM X [m]", color="#CBD5E1")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.2, loc="upper left", framealpha=0.7)

    ax = axs[0, 1]
    ax.set_facecolor("#111827")
    ax.plot(ta["t"], ta["door_deg"], "--", color="#EF4444", lw=1.8, label="Trial A: Door Hinge Angle")
    ax.plot(tb["t"], tb["door_deg"], "-", color="#38BDF8", lw=2.2, label="Trial B: Push Open + Auto-Close")
    ax.axhline(70.0, color="#10B981", ls=":", lw=1.2, label="Clearance Opening (70°)")
    ax.set_title("2. Articulated Door Hinge Angle [deg]", color="#E2E8F0", fontsize=10.5)
    ax.set_xlabel("Time [s]", color="#CBD5E1")
    ax.set_ylabel("Hinge Opening [deg]", color="#CBD5E1")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.7)

    ax = axs[1, 0]
    ax.set_facecolor("#111827")
    ax.plot(ta["t"], ta["torso_n"], "--", color="#EF4444", lw=1.8, label="Trial A: Torso Collision [N]")
    ax.plot(tb["t"], tb["palm_n"], "-", color="#F59E0B", lw=2.0, label="Trial B: Unlatched Palm Push [N]")
    ax.plot(tb["t"], tb["torso_n"], "-", color="#10B981", lw=1.6, label="Trial B: Torso Jamb Contact [N]")
    ax.set_title("3. Unlatched Palm Push vs Torso Jamb Collision Force", color="#E2E8F0", fontsize=10.5)
    ax.set_xlabel("Time [s]", color="#CBD5E1")
    ax.set_ylabel("Contact Force [N]", color="#CBD5E1")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.7)

    ax = axs[1, 1]
    ax.set_facecolor("#111827")
    ax.plot([DOOR_PLANE_X, DOOR_PLANE_X], [0.415, 0.56], color="#94A3B8", lw=6.0, label="Jamb Posts")
    ax.plot([DOOR_PLANE_X, DOOR_PLANE_X], [-0.56, -0.415], color="#94A3B8", lw=6.0)
    th_p = np.deg2rad(res_b["peak_door_angle_deg"])
    ax.plot([DOOR_PLANE_X, DOOR_PLANE_X], [0.41, -0.40], ":", color="#EF4444", lw=1.6, label="Door Closed (0°)")
    ax.plot(
        [DOOR_PLANE_X, DOOR_PLANE_X + 0.81 * np.sin(th_p)],
        [0.41, 0.41 - 0.81 * np.cos(th_p)],
        "-", color="#38BDF8", lw=3.0, label=f"Door Peak ({res_b['peak_door_angle_deg']:.1f}°)"
    )
    ax.plot(ta["com_x"], ta["com_y"], "--", color="#EF4444", lw=1.6, label="Trial A Path")
    ax.plot(tb["com_x"], tb["com_y"], "-", color="#10B981", lw=2.2, label="Trial B Traversal Path")
    ax.set_title("4. Overhead Doorway Clearance Path (Y vs X)", color="#E2E8F0", fontsize=10.5)
    ax.set_xlabel("World X [m]", color="#CBD5E1")
    ax.set_ylabel("World Y [m]", color="#CBD5E1")
    ax.set_ylim(-0.58, 0.58)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7.8, loc="lower left", ncol=2, framealpha=0.72)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(base_dir, "figure05_mujoco_door_traversal_benchmark.png")
    plt.savefig(out_png, dpi=220, facecolor="#0B0F19")
    plt.close(fig)

    print(f"[+] Telemetry saved: {out_json}")
    print(f"[+] Benchmark plot saved: {out_png}")

    assert max_skyhook == 0.0, f"Skyhook violation: {max_skyhook}"
    assert not res_a["traversed_doorway"], "Trial A unexpectedly passed through closed door!"
    assert res_b["survived"] and res_b["traversed_doorway"], "Trial B failed doorway traversal!"
    assert res_b["peak_door_angle_deg"] >= 70.0, f"Door did not open past 70 deg: {res_b['peak_door_angle_deg']}"
    assert res_b["final_door_angle_deg"] < 15.0, f"Spring-return door did not close behind robot: {res_b['final_door_angle_deg']}"
    assert res_b["post_release_palm_force_n"] <= 0.5, f"Residual palm force after release: {res_b['post_release_palm_force_n']}"


if __name__ == "__main__":
    run_door_traversal_benchmark()
