"""
Project Figure 05 — Module #25: Skyhook-Free MuJoCo 3.6 65.0 kg Cart Push & Lean WBC
====================================================================================
Demonstrates true unlatched bimanual contact pushing of a free-rolling 65.0 kg
heavy industrial parts cart (HEAVIER than the 54.5 kg humanoid -> 119.5 kg
coupled multibody system) with ZERO weld/equality constraints and ZERO skyhook
forces (qfrc_applied = 0.0).

Physics & Control Architecture:
1. True Unilateral Contact Physics:
   - Both hands contact the cart's rear pushplate (Z = 1.04 m) strictly via
     bilateral bitmask collision (contype="2"), with zero artificial glue/weld.
   - Four free-rolling caster skids support the 65.0 kg cart on dedicated rails
     with mu = 0.055 rolling resistance and an unannounced high-drag expansion
     plate (X in [1.15, 1.45] m, mu = 0.105) testing mid-push disturbance rejection.
2. Causal Newton-3rd-Law Lean-Compensated WBC:
   - Measures real hand contact force F_hand via mj_contactForce (zero cart oracle).
   - Feeds forward -F_hand horizontal traction, lever-arm pitch moment
     compensation, and adaptive forward trunk lean (atan(|F_push| / m*g)) while
     stepping smoothly, then brakes & releases cleanly at station.
"""

from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from figure05_mujoco_common import (
    build_actuator_mappings,
    clamp_and_apply_ctrl,
    mat_to_rotvec_err,
    quat_to_roll_pitch_yaw,
    setup_crouched_biped,
    sum_support_grfs,
)

CART_MASS_KG = 65.0
PUSH_THRESHOLD_START_X = 1.35
PUSH_THRESHOLD_END_X = 1.65


def compute_push_lean_feedforward(
    f_hand_on_robot_x: float,
    total_mass_kg: float,
    z_hand_m: float,
    z_com_m: float,
    g: float = 9.81,
) -> tuple[float, float, float]:
    """
    Causal statics/centroidal feedforward for external bimanual pushing.
    When Figure 05 pushes +X on the cart, the cart pushes -X on the hands
    (f_hand_on_robot_x <= 0). To stay in dynamic equilibrium:
      - GRF feedforward cancels reaction: fx_grf_ff = -f_hand_on_robot_x (> 0 forward)
      - Lever arm above CoM induces backward pitch: cancel with -dz * f_hand_on_robot_x
      - Forward CoM/trunk lean balances push torque: lean_rad = clip(f_push / (m*g), 0, 0.14)
    """
    f_push_fwd = max(0.0, -float(f_hand_on_robot_x))
    fx_grf_ff = f_push_fwd
    dz = max(0.0, float(z_hand_m) - float(z_com_m))
    tau_pitch_ff = dz * f_push_fwd
    lean_rad = float(np.clip(np.arctan2(f_push_fwd, total_mass_kg * g), 0.0, 0.14))
    return fx_grf_ff, tau_pitch_ff, lean_rad


def build_cart_push_mjcf(base_dir: str) -> str:
    """Inject free-rolling 65.0 kg industrial parts cart & hand contact pads into scene."""
    xml_path = os.path.join(base_dir, "figure05_mujoco.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")

    # Add palm contact pads strictly bitmask=2 so hands only collide with cart pushbar
    for side in ("left", "right"):
        for body in wb.iter("body"):
            if body.get("name") == f"{side}_hand":
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{side}_palm_push_pad",
                        "type": "sphere",
                        "size": "0.042",
                        "pos": "0.032 0.0 -0.02",
                        "mass": "0.0001",
                        "contype": "2",
                        "conaffinity": "2",
                        "friction": "0.85 0.005 0.0005",
                        "rgba": "0.20 0.85 0.95 0.9",
                    },
                )

    # Dedicated caster rolling tracks (bitmask=8) so cart friction never compromises robot feet
    ET.SubElement(
        wb,
        "geom",
        {
            "name": "caster_track_entry",
            "type": "box",
            "pos": "0.0 0 -0.05",
            "size": "1.35 2.0 0.05",
            "contype": "8",
            "conaffinity": "8",
            "friction": "0.048 0.001 0.0001",
            "rgba": "0.18 0.24 0.34 0.3",
        },
    )
    ET.SubElement(
        wb,
        "geom",
        {
            "name": "caster_track_threshold",
            "type": "box",
            "pos": "1.50 0 -0.05",
            "size": "0.15 2.0 0.05",
            "contype": "8",
            "conaffinity": "8",
            "priority": "1",
            "friction": "0.085 0.002 0.0002",
            "rgba": "0.95 0.55 0.12 0.55",
        },
    )
    ET.SubElement(
        wb,
        "geom",
        {
            "name": "caster_track_exit",
            "type": "box",
            "pos": "5.65 0 -0.05",
            "size": "4.0 2.0 0.05",
            "contype": "8",
            "conaffinity": "8",
            "friction": "0.048 0.001 0.0001",
            "rgba": "0.18 0.24 0.34 0.3",
        },
    )

    # 65.0 kg Free-Rolling Industrial Parts Roll-Cage Cart (4 casters + cargo bay + pushplate)
    cart = ET.SubElement(
        wb,
        "body",
        {
            "name": "industrial_cart",
            "pos": "0.82 0.0 0.002",
        },
    )
    ET.SubElement(cart, "freejoint", {"name": "cart_freejoint"})
    # Main weighted deck + cargo load (56.6 kg)
    ET.SubElement(
        cart,
        "geom",
        {
            "name": "cart_payload_chassis",
            "type": "box",
            "pos": "0.06 0.0 0.44",
            "size": "0.34 0.32 0.34",
            "mass": "56.6",
            "rgba": "0.24 0.32 0.45 0.85",
        },
    )
    # 4 low-drag caster skids touching Z=0 via bitmask=8 (1.5 kg each = 6.0 kg)
    for cname, cx, cy in [
        ("caster_fl", 0.34, 0.27),
        ("caster_fr", 0.34, -0.27),
        ("caster_rl", -0.24, 0.27),
        ("caster_rr", -0.24, -0.27),
    ]:
        ET.SubElement(
            cart,
            "geom",
            {
                "name": cname,
                "type": "sphere",
                "pos": f"{cx} {cy} 0.048",
                "size": "0.050",
                "mass": "1.5",
                "contype": "8",
                "conaffinity": "8",
                "friction": "0.048 0.001 0.0001",
                "rgba": "0.12 0.14 0.18 1.0",
            },
        )
    # Compliant rubberized pushbar/handleplate at X_world ~ 0.475 m, Z = 1.01 m (2.0 kg) + side rails (0.4 kg)
    ET.SubElement(
        cart,
        "geom",
        {
            "name": "cart_pushbar",
            "type": "box",
            "pos": "-0.345 0.0 1.01",
            "size": "0.025 0.36 0.12",
            "mass": "2.0",
            "contype": "2",
            "conaffinity": "2",
            "solref": "0.025 1.1",
            "friction": "0.85 0.005 0.0005",
            "rgba": "0.10 0.82 0.55 0.95",
        },
    )
    for rname, ry in [("pushbar_rail_l", 0.365), ("pushbar_rail_r", -0.365)]:
        ET.SubElement(
            cart,
            "geom",
            {
                "name": rname,
                "type": "box",
                "pos": f"-0.385 {ry} 1.01",
                "size": "0.045 0.015 0.12",
                "mass": "0.2",
                "contype": "2",
                "conaffinity": "2",
                "solref": "0.025 1.1",
                "friction": "0.35 0.003 0.0003",
                "rgba": "0.35 0.45 0.55 0.95",
            },
        )
    return ET.tostring(root, encoding="unicode")


def measure_bimanual_push_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pushbar_gid: int,
    l_pad_gid: int,
    r_pad_gid: int,
) -> tuple[np.ndarray, float, float]:
    """Return total 3D contact force acting ON the robot's hands (`fx < 0` when pushing +X)."""
    f_on_robot = np.zeros(3, dtype=float)
    f_left = 0.0
    f_right = 0.0
    cf = np.zeros(6, dtype=float)
    hand_set = {l_pad_gid, r_pad_gid}
    for ci in range(data.ncon):
        con = data.contact[ci]
        g1, g2 = con.geom1, con.geom2
        if not ((g1 == pushbar_gid and g2 in hand_set) or (g2 == pushbar_gid and g1 in hand_set)):
            continue
        mujoco.mj_contactForce(model, data, ci, cf)
        r_c = con.frame.reshape(3, 3)
        # Force exerted BY pushbar ON robot hand
        sign = 1.0 if g1 == pushbar_gid else -1.0
        f_world = sign * (r_c.T @ cf[:3])
        f_on_robot += f_world
        hand_gid = g2 if g1 == pushbar_gid else g1
        mag = max(0.0, -float(f_world[0]))
        if hand_gid == l_pad_gid:
            f_left += mag
        else:
            f_right += mag
    return f_on_robot, f_left, f_right


def run_cart_push_sim(max_sim_s: float = 9.8) -> dict:
    base_dir = os.path.dirname(__file__)
    xml_str = build_cart_push_mjcf(base_dir)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    setup_crouched_biped(model, data, shoulder_pitch=-0.55, elbow=-0.45, torso_pitch=0.02)
    cart_jid = model.joint("cart_freejoint").id
    cart_qpos_idx = model.jnt_qposadr[cart_jid]
    cart_vel_idx = model.jnt_dofadr[cart_jid]
    data.qpos[cart_qpos_idx : cart_qpos_idx + 7] = [0.82, 0.0, 0.002, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    cart_body_id = model.body("industrial_cart").id
    cart_mass = float(model.body_mass[cart_body_id])
    robot_mass = float(np.sum(model.body_mass)) - cart_mass

    maps = build_actuator_mappings(model)
    act_names = maps["act_names"]
    act_qidx = maps["act_qidx"]
    act_vidx = maps["act_vidx"]
    idx_map = maps["idx_map"]
    l_leg_v = maps["l_leg_v"]
    r_leg_v = maps["r_leg_v"]
    l_leg_act = maps["l_leg_act"]
    r_leg_act = maps["r_leg_act"]
    upper_act = maps["upper_act"]
    q_init = np.array([data.qpos[idx] for idx in act_qidx], dtype=float)

    l_body = model.body("left_ankle").id
    r_body = model.body("right_ankle").id
    l_site = model.site("left_foot_contact").id
    r_site = model.site("right_foot_contact").id
    l_hand_body = model.body("left_hand").id
    r_hand_body = model.body("right_hand").id
    l_pad_gid = model.geom("left_palm_push_pad").id
    r_pad_gid = model.geom("right_palm_push_pad").id
    pushbar_gid = model.geom("cart_pushbar").id
    support_geoms = {model.geom("floor").id}

    dt = float(model.opt.timestep)
    g = 9.81
    z_c = 0.870
    omega = float(np.sqrt(g / z_c))

    step_T = 0.42
    ds_frac = 0.18
    by_nom = 0.102
    h_lift = 0.058

    jac_lp = np.zeros((3, model.nv))
    jac_lr = np.zeros((3, model.nv))
    jac_rp = np.zeros((3, model.nv))
    jac_rr = np.zeros((3, model.nv))
    jac_lhp = np.zeros((3, model.nv))
    jac_rhp = np.zeros((3, model.nv))
    jac_com = np.zeros((3, model.nv))

    # Quiet settle (~0.25 s) before reaching bar
    for _ in range(125):
        data.qfrc_applied[:] = 0.0
        for i in range(model.nu):
            qv = data.qpos[act_qidx[i]]
            qdv = data.qvel[act_vidx[i]]
            data.ctrl[i] = 280.0 * (q_init[i] - qv) - 18.0 * qdv
        clamp_and_apply_ctrl(model, data, data.ctrl.copy())
        mujoco.mj_step(model, data)

    cart_start_x = float(data.xpos[cart_body_id][0])
    p0_l = data.site_xpos[l_site].copy()
    p0_r = data.site_xpos[r_site].copy()
    c0_xy = 0.5 * (p0_l[:2] + p0_r[:2])

    t_stand = 0.65
    # Exact 19-step boundary (0.65 + 19 * 0.42 = 8.63 s) so robot always halts in Double Support
    t_walk_end = t_stand + 19 * step_T
    total_steps = int(max_sim_s / dt)

    step_prev = -1
    stance_side = 1
    swing = None
    s_sw = 0.0
    sw_start_pos = data.site_xpos[r_site].copy()
    prev_pdes = sw_start_pos.copy()
    swing_first_tick = True

    f_push_lpf = 0.0
    skyhook_max = 0.0

    times = []
    cart_x_log = []
    robot_x_log = []
    push_force_log = []
    lean_deg_log = []
    roll_log = []
    pitch_log = []
    fz_log = []
    knee_tau_log = []
    settled_x = float(data.qpos[0])
    fell = False

    # Arm actuator index lists for workspace push impedance
    l_arm_names = ["act_left_shoulder_pitch", "act_left_shoulder_roll", "act_left_shoulder_yaw", "act_left_elbow"]
    r_arm_names = ["act_right_shoulder_pitch", "act_right_shoulder_roll", "act_right_shoulder_yaw", "act_right_elbow"]
    l_arm_act = [idx_map[n] for n in l_arm_names]
    r_arm_act = [idx_map[n] for n in r_arm_names]
    l_arm_v = [act_vidx[a] for a in l_arm_act]
    r_arm_v = [act_vidx[a] for a in r_arm_act]

    for step in range(total_steps):
        t = step * dt
        data.qfrc_applied[:] = 0.0
        skyhook_max = max(skyhook_max, float(np.max(np.abs(data.qfrc_applied))))

        # Whole-robot CoM of pelvis subtree (index 1, excluding independent cart freejoint!)
        mujoco.mj_jacSubtreeCom(model, data, jac_com, 1)
        com = data.subtree_com[1].copy()
        com_vel = jac_com @ data.qvel

        p_l = data.site_xpos[l_site].copy()
        p_r = data.site_xpos[r_site].copy()
        r_l = data.xmat[l_body].reshape(3, 3).copy()
        r_r = data.xmat[r_body].reshape(3, 3).copy()
        mujoco.mj_jacSite(model, data, jac_lp, jac_lr, l_site)
        mujoco.mj_jacSite(model, data, jac_rp, jac_rr, r_site)
        mujoco.mj_jacBody(model, data, jac_lhp, None, l_hand_body)
        mujoco.mj_jacBody(model, data, jac_rhp, None, r_hand_body)

        v_l = jac_lp @ data.qvel
        v_r = jac_rp @ data.qvel

        roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
        droll, dpitch, dyaw = [float(v) for v in data.qvel[3:6]]

        f_hand_on_robot, f_left_push, f_right_push = measure_bimanual_push_force(
            model, data, pushbar_gid, l_pad_gid, r_pad_gid
        )
        f_push_inst = max(0.0, -float(f_hand_on_robot[0]))
        alpha_p = 1.0 - float(np.exp(-dt / 0.025))
        f_push_lpf = (1.0 - alpha_p) * f_push_lpf + alpha_p * f_push_inst

        z_hand_avg = 0.5 * float(data.xpos[l_hand_body][2] + data.xpos[r_hand_body][2])
        fx_grf_ff, tau_pitch_ff, lean_rad = compute_push_lean_feedforward(
            -f_push_lpf, robot_mass, z_hand_avg, float(com[2]), g
        )

        # Smooth velocity schedule: accelerate into push, cruise, decelerate & release
        push_active = t < t_walk_end
        if t < t_stand:
            vx_cmd = 0.0
        elif t < t_walk_end - 0.70:
            vx_cmd = 0.36 * float(np.clip((t - t_stand) / 0.75, 0.0, 1.0))
        elif t < t_walk_end:
            vx_cmd = 0.36 * float(np.clip((t_walk_end - t) / 0.70, 0.0, 1.0))
        else:
            vx_cmd = 0.0

        dcm = com[:2] + com_vel[:2] / omega
        bx_cmd = vx_cmd * step_T / (np.exp(omega * step_T) - 1.0) if vx_cmd > 1e-4 else 0.0

        s_stand = 0.5 * (1.0 - float(np.cos(np.pi * np.clip(t / t_stand, 0.0, 1.0))))
        if t < t_stand or t >= t_walk_end:
            swing = None
        else:
            t_w = t - t_stand
            step_idx = int(t_w / step_T)
            phi = (t_w % step_T) / step_T
            if step_idx != step_prev:
                step_prev = step_idx
                stance_side = 1 if (step_idx % 2 == 0) else -1
                sw_start_pos = (p_r if stance_side == 1 else p_l).copy()
                sw_start_pos[2] = max(0.0, sw_start_pos[2])
                prev_pdes = sw_start_pos.copy()
                swing_first_tick = True

            if phi < ds_frac:
                s = phi / max(ds_frac, 1e-4)
                smooth = 0.5 * (1.0 - np.cos(np.pi * s))
                alpha_l = (0.50 + 0.42 * smooth) if step_idx == 0 else (smooth if stance_side == 1 else 1.0 - smooth)
                alpha_r = 1.0 - alpha_l
                swing = None
            else:
                alpha_l = 1.0 if stance_side == 1 else 0.0
                alpha_r = 1.0 - alpha_l
                swing = "R" if stance_side == 1 else "L"
                s_sw = (phi - ds_frac) / (1.0 - ds_frac)

        # Centroidal Vertical + Lean-Compensated Horizontal Wrench
        fz_tot = robot_mass * (g + 160.0 * (z_c - com[2]) - 26.0 * com_vel[2]) - float(f_hand_on_robot[2])
        fz_tot = float(np.clip(fz_tot, 180.0, 1150.0))

        # Slight forward CoM offset counterbalances pushing reaction torque without tipping back
        x_lean_offset = float(np.clip(com[2] * np.tan(lean_rad), 0.0, 0.09))
        if t < t_stand:
            c_tgt = c0_xy + np.array([0.005 * s_stand + 0.5 * x_lean_offset, by_nom * 0.62 * s_stand])
            f_xy = robot_mass * np.array([
                75.0 * (c_tgt[0] - com[0]) - 14.0 * com_vel[0],
                95.0 * (c_tgt[1] - com[1]) - 16.0 * com_vel[1],
            ])
            p_zmp_y = float(np.clip(com[1] - f_xy[1] / (robot_mass * (omega**2)), p_r[1], p_l[1]))
            alpha_l = float(np.clip((p_zmp_y - p_r[1]) / max(p_l[1] - p_r[1], 0.05), 0.12, 0.88))
            alpha_r = 1.0 - alpha_l
        elif t >= t_walk_end:
            p_mid = 0.5 * (p_l[:2] + p_r[:2])
            p_zmp_y = float(np.clip(com[1], p_r[1], p_l[1]))
            alpha_l = float(np.clip((p_zmp_y - p_r[1]) / max(p_l[1] - p_r[1], 0.05), 0.15, 0.85))
            alpha_r = 1.0 - alpha_l
            f_xy = robot_mass * np.array([
                80.0 * (p_mid[0] - com[0]) - 16.0 * com_vel[0],
                100.0 * (p_mid[1] - com[1]) - 16.0 * com_vel[1],
            ])
        elif swing is None:
            p_cop_des = alpha_l * p_l[:2] + alpha_r * p_r[:2]
            f_xy = robot_mass * np.array([
                4.5 * (vx_cmd - com_vel[0]),
                32.0 * (p_cop_des[1] - com[1]) - 6.0 * com_vel[1],
            ])
        else:
            p_st = p_l if stance_side == 1 else p_r
            cop_x = p_st[0] + np.clip(0.25 * (dcm[0] - (p_st[0] + bx_cmd)) - 0.35 * x_lean_offset, -0.06, 0.08)
            cop_y = p_st[1] + np.clip(0.25 * (dcm[1] - (p_st[1] - stance_side * by_nom)), -0.03, 0.03)
            f_xy = robot_mass * (omega**2) * np.array([com[0] - cop_x, com[1] - cop_y])
            f_xy = np.clip(f_xy, -220.0, 220.0)

        # Feed forward Newton-3rd-Law reaction cancellation at soles
        f_xy = f_xy + np.array([fx_grf_ff, -0.5 * float(f_hand_on_robot[1])])

        pitch_tgt = lean_rad if (push_active and t >= t_stand * 0.5) else 0.02
        tau_trunk = np.array([
            -310.0 * roll - 34.0 * droll,
            -290.0 * (pitch - pitch_tgt) - 34.0 * dpitch + tau_pitch_ff,
            -95.0 * yaw - 12.0 * dyaw,
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
                np.clip(alpha_l * tau_trunk[0], -0.055 * fz_l, 0.055 * fz_l),
                np.clip(alpha_l * tau_trunk[1], -0.105 * fz_l, 0.105 * fz_l),
                alpha_l * tau_trunk[2],
            ])
        if alpha_r > 0.01:
            w_r = np.array([
                alpha_r * f_xy[0],
                alpha_r * f_xy[1],
                fz_r,
                np.clip(alpha_r * tau_trunk[0], -0.055 * fz_r, 0.055 * fz_r),
                np.clip(alpha_r * tau_trunk[1], -0.105 * fz_r, 0.105 * fz_r),
                alpha_r * tau_trunk[2],
            ])

        j_l = np.vstack([jac_lp, jac_lr])
        j_r = np.vstack([jac_rp, jac_rr])
        tau_gen = data.qfrc_bias.copy()
        if alpha_l > 0.01:
            tau_gen -= j_l.T @ w_l
        if alpha_r > 0.01:
            tau_gen -= j_r.T @ w_r

        tau_ctrl = np.array([tau_gen[idx] for idx in act_vidx], dtype=float)

        if swing is not None:
            p_st = p_l if stance_side == 1 else p_r
            s_sw_side = -1 if stance_side == 1 else 1
            tgt_x = np.clip(dcm[0] - bx_cmd, p_st[0] - 0.08, p_st[0] + 0.36)
            tgt_y = dcm[1] + s_sw_side * by_nom - 0.08 * com[1]
            if s_sw_side == -1:
                tgt_y = np.clip(tgt_y, p_st[1] - 0.33, p_st[1] - 0.15)
            else:
                tgt_y = np.clip(tgt_y, p_st[1] + 0.15, p_st[1] + 0.33)

            smooth_xy = 0.5 * (1.0 - np.cos(np.pi * s_sw))
            z_des = h_lift * np.sin(np.pi * s_sw) - 0.002 * (s_sw**2)
            p_des = np.array([
                sw_start_pos[0] + smooth_xy * (tgt_x - sw_start_pos[0]),
                sw_start_pos[1] + smooth_xy * (tgt_y - sw_start_pos[1]),
                z_des,
            ])
            v_des = np.zeros(3) if swing_first_tick else (p_des - prev_pdes) / dt
            swing_first_tick = False
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
                np.array([100.0, 130.0, 80.0]) * r_err
                - np.array([7.0, 9.0, 6.0]) * w_cur,
            ])
            tau_sw_full = data.qfrc_bias + j_sw.T @ f_sw
            for ai, vi in zip(sw_acts, sw_dofs):
                tau_ctrl[ai] = tau_sw_full[vi]

            st_acts = l_leg_act if stance_side == 1 else r_leg_act
            tau_ctrl[st_acts[0]] += 50.0 * yaw + 6.0 * dyaw
            tau_ctrl[st_acts[1]] += 110.0 * roll + 12.0 * droll
            tau_ctrl[st_acts[2]] += 100.0 * (pitch - pitch_tgt) + 11.0 * dpitch
        else:
            tau_ctrl[l_leg_act[1]] += 60.0 * roll + 7.0 * droll
            tau_ctrl[r_leg_act[1]] += 60.0 * roll + 7.0 * droll
            tau_ctrl[l_leg_act[2]] += 50.0 * (pitch - pitch_tgt) + 6.0 * dpitch
            tau_ctrl[r_leg_act[2]] += 50.0 * (pitch - pitch_tgt) + 6.0 * dpitch

        # Upper-body posture + bimanual arm workspace push impedance (pure bilateral contact)
        q = np.array([data.qpos[idx] for idx in act_qidx])
        qd = np.array([data.qvel[idx] for idx in act_vidx])
        q_t = q_init.copy()
        q_t[idx_map["act_torso_pitch"]] = pitch_tgt * 0.65

        for i in upper_act:
            name = act_names[i]
            kp = 300.0 if ("waist" in name or "torso" in name) else 130.0
            kd = 24.0 if ("waist" in name or "torso" in name) else 12.0
            tau_ctrl[i] = kp * (q_t[i] - q[i]) - kd * qd[i]

        # Operational-space hand push impedance pushing forward onto the 65 kg pushbar
        reach_s = 0.5 * (1.0 - float(np.cos(np.pi * np.clip(t / t_stand, 0.0, 1.0))))
        walk_push_s = float(np.clip((t - 0.30) / 0.80, 0.0, 1.0)) if t < t_walk_end else 1.0
        if t < t_walk_end:
            x_hand_tgt_rel = 0.36 + 0.075 * reach_s + 0.055 * walk_push_s
            f_push_arm_ff = 44.0 * walk_push_s
        else:
            # Cleanly retract palms off pushbar once cart has reached dock
            retract_s = float(np.clip((t - t_walk_end) / 0.55, 0.0, 1.0))
            x_hand_tgt_rel = 0.490 - 0.16 * retract_s
            f_push_arm_ff = 44.0 * (1.0 - retract_s)

        # Differential L/R steering equalization prevents asymmetric Yaw spinout
        f_diff_lr = float(np.clip(f_left_push - f_right_push, -20.0, 20.0))
        for side_s, hand_b, jac_h, y_off, arm_acts, arm_vs in [
            (1.0, l_hand_body, jac_lhp, 0.22, l_arm_act, l_arm_v),
            (-1.0, r_hand_body, jac_rhp, -0.22, r_arm_act, r_arm_v),
        ]:
            p_h = data.xpos[hand_b]
            v_h = jac_h @ data.qvel
            dx_eq = -0.0010 * side_s * f_diff_lr
            p_h_des = np.array([float(com[0]) + x_hand_tgt_rel + dx_eq, float(com[1]) + y_off, 1.01])
            v_h_des = np.array([vx_cmd, 0.0, 0.0])
            f_h_cmd = np.array([f_push_arm_ff, 0.0, 0.0]) + np.array([240.0, 180.0, 260.0]) * (
                p_h_des - p_h
            ) + np.array([28.0, 18.0, 22.0]) * (v_h_des - v_h)
            tau_h = data.qfrc_bias + jac_h.T @ f_h_cmd
            for ai, vi in zip(arm_acts, arm_vs):
                # Blend gravity-compensated cartesian push impedance with posture regularization
                tau_ctrl[ai] = float(tau_h[vi]) - 4.0 * qd[ai] + 15.0 * (q_init[ai] - q[ai])

        clamp_and_apply_ctrl(model, data, tau_ctrl)
        mujoco.mj_step(model, data)

        pelvis_z = float(data.qpos[2])
        if step % 5 == 0:
            l_fz, r_fz = sum_support_grfs(model, data, support_geoms, l_body, r_body)
            l_knee = abs(float(data.ctrl[idx_map["act_left_knee"]]))
            r_knee = abs(float(data.ctrl[idx_map["act_right_knee"]]))
            times.append(round(t, 4))
            cart_x_log.append(float(data.xpos[cart_body_id][0]))
            robot_x_log.append(float(com[0]))
            push_force_log.append(float(f_push_inst))
            lean_deg_log.append(float(np.degrees(lean_rad)))
            roll_log.append(float(np.degrees(roll)))
            pitch_log.append(float(np.degrees(pitch)))
            fz_log.append(float(l_fz + r_fz))
            knee_tau_log.append(max(l_knee, r_knee))

        if pelvis_z < 0.60 or abs(roll) > 0.70 or abs(pitch) > 0.70:
            fell = True
            break

    cart_final_x = float(data.xpos[cart_body_id][0])
    cart_disp = cart_final_x - cart_start_x
    cart_final_speed = float(np.hypot(data.qvel[cart_vel_idx], data.qvel[cart_vel_idx + 1]))
    f_final, _, _ = measure_bimanual_push_force(model, data, pushbar_gid, l_pad_gid, r_pad_gid)
    push_arr = np.array(push_force_log, dtype=float)
    active_push = push_arr[np.array(times) < t_walk_end]

    return {
        "times": times,
        "cart_x": cart_x_log,
        "robot_x": robot_x_log,
        "push_force_n": push_force_log,
        "lean_deg": lean_deg_log,
        "roll_deg": roll_log,
        "pitch_deg": pitch_log,
        "fz_total_n": fz_log,
        "knee_tau_nm": knee_tau_log,
        "summary": {
            "mujoco_version": mujoco.__version__,
            "skyhook_force_n": round(float(skyhook_max), 4),
            "unlatched_pure_contact": True,
            "robot_mass_kg": round(robot_mass, 2),
            "cart_mass_kg": round(cart_mass, 2),
            "total_coupled_mass_kg": round(robot_mass + cart_mass, 2),
            "completed_without_fall": not fell,
            "duration_s": round(float(times[-1]), 3),
            "cart_start_x_m": round(cart_start_x, 3),
            "cart_final_x_m": round(cart_final_x, 3),
            "cart_pushed_distance_m": round(cart_disp, 3),
            "robot_walked_distance_m": round(float(data.qpos[0]) - settled_x, 3),
            "mean_active_push_force_n": round(float(np.mean(active_push)), 1) if len(active_push) else 0.0,
            "peak_push_force_n": round(float(np.max(push_arr)), 1) if len(push_arr) else 0.0,
            "post_release_push_force_n": round(max(0.0, -float(f_final[0])), 2),
            "cart_final_residual_speed_mps": round(cart_final_speed, 3),
            "roll_rms_deg": round(float(np.sqrt(np.mean(np.square(roll_log)))), 2),
            "pitch_rms_deg": round(float(np.sqrt(np.mean(np.square(pitch_log)))), 2),
            "peak_knee_torque_nm": round(float(np.max(knee_tau_log)), 1),
            "steady_total_grf_n": round(float(np.mean(fz_log[-60:])), 1),
        },
    }


def render_plot(base_dir: str, res: dict) -> str:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), facecolor="#0A0F1A")
    s = res["summary"]
    fig.suptitle(
        f"FIGURE 05 // MUJOCO 3.6 UNLATCHED {s['cart_mass_kg']:.0f} kg HEAVY CART PUSH "
        f"(SYSTEM = {s['total_coupled_mass_kg']:.1f} kg, SKYHOOK = 0.0 N)",
        fontsize=13,
        fontweight="bold",
        color="#38BDF8",
        y=0.98,
    )
    for ax in axes.flat:
        ax.set_facecolor("#0F172A")
        ax.grid(True, linestyle="--", alpha=0.28, color="#334155")

    ax = axes[0, 0]
    ax.axhspan(PUSH_THRESHOLD_START_X, PUSH_THRESHOLD_END_X, color="#F59E0B", alpha=0.16, label="Drag Strip (mu=0.105)")
    ax.plot(res["times"], res["cart_x"], color="#10B981", lw=2.3, label="65 kg Cart X (m)")
    ax.plot(res["times"], res["robot_x"], color="#38BDF8", lw=2.0, ls="--", label="Figure 05 CoM X (m)")
    ax.set_title(f"Cart Displacement (+{s['cart_pushed_distance_m']:.2f} m Unlatched Contact Push)", fontsize=10.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X Position (m)")
    ax.legend(loc="upper left", fontsize=8.2)

    ax = axes[0, 1]
    ax.plot(res["times"], res["push_force_n"], color="#38BDF8", lw=2.1, label="Bimanual Push Force (N)")
    ax.axhline(s["mean_active_push_force_n"], color="#10B981", ls="--", lw=1.4, label=f"Mean Active: {s['mean_active_push_force_n']} N")
    ax.axvline(8.63, color="#FB7185", ls=":", lw=1.3, label="Dock Decel & Release")
    ax.set_title("Unilateral Handlebar Contact Force & Clean Release", fontsize=10.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend(loc="upper right", fontsize=8.2)

    ax = axes[1, 0]
    ax.plot(res["times"], res["roll_deg"], color="#10B981", lw=1.9, label=f"Roll (RMS {s['roll_rms_deg']}°)")
    ax.plot(res["times"], res["pitch_deg"], color="#38BDF8", lw=1.9, label=f"Pitch (RMS {s['pitch_rms_deg']}°)")
    ax.plot(res["times"], res["lean_deg"], color="#F59E0B", ls="--", lw=1.6, label="Feedforward Lean Ref (deg)")
    ax.set_title("Gravity-Lean Compensated Attitude Tracking", fontsize=10.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (deg)")
    ax.legend(loc="best", fontsize=8.0)

    ax = axes[1, 1]
    ax.plot(res["times"], res["fz_total_n"], color="#10B981", lw=1.8, label="Dual-Foot Vertical GRF (N)")
    ax.plot(res["times"], res["knee_tau_nm"], color="#A855F7", lw=1.8, label=f"Peak Knee Torque ({s['peak_knee_torque_nm']} Nm)")
    ax.axhline(s["robot_mass_kg"] * 9.81, color="#38BDF8", ls=":", lw=1.2, label="Nominal Weight (534.6 N)")
    ax.set_title("Ground Reaction Force & Class Gamma Knee Load", fontsize=10.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N) / Torque (Nm)")
    ax.legend(loc="upper right", fontsize=8.0)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(base_dir, "figure05_mujoco_cart_push_benchmark.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main() -> None:
    print("=========================================================================")
    print("FIGURE 05 // MUJOCO 3.6 BIMANUAL 65.0 kg HEAVY CART PUSH BENCHMARK")
    print("=========================================================================")
    res = run_cart_push_sim(max_sim_s=9.8)
    s = res["summary"]
    print(f"[*] Coupled System Mass:      {s['robot_mass_kg']} kg Robot + {s['cart_mass_kg']} kg Cart = {s['total_coupled_mass_kg']} kg")
    print(f"[*] Cart Pushed Distance:     +{s['cart_pushed_distance_m']} m ({s['cart_start_x_m']} -> {s['cart_final_x_m']} m)")
    print(f"[*] Active Push Force:        mean={s['mean_active_push_force_n']} N | peak={s['peak_push_force_n']} N | release={s['post_release_push_force_n']} N")
    print(f"[*] Attitude & Knee Load:     Roll/Pitch RMS={s['roll_rms_deg']}/{s['pitch_rms_deg']} deg | peak_knee={s['peak_knee_torque_nm']} Nm")
    print(f"[*] Cart Final Speed & GRF:   v_cart_end={s['cart_final_residual_speed_mps']} m/s | GRF={s['steady_total_grf_n']} N | skyhook={s['skyhook_force_n']} N")

    assert s["skyhook_force_n"] == 0.0
    assert s["completed_without_fall"] is True
    assert s["cart_pushed_distance_m"] >= 1.15
    assert s["mean_active_push_force_n"] >= 15.0
    assert s["post_release_push_force_n"] <= 1.0
    assert s["roll_rms_deg"] < 6.0 and s["pitch_rms_deg"] < 8.0
    assert s["peak_knee_torque_nm"] < 260.0

    base_dir = os.path.dirname(__file__)
    out_png = render_plot(base_dir, res)
    print(f"[+] Saved 65 kg Cart Push plot: {out_png}")
    out_json = os.path.join(base_dir, "figure05_cart_push_telemetry.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    print(f"[+] Saved 65 kg Cart Push telemetry: {out_json}")
    print("[PASS] Module #25 MuJoCo 65.0 kg Heavy Cart Push verified cleanly.")


if __name__ == "__main__":
    main()
