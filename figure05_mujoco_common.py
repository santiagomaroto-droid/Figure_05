"""
Figure 05 Shared MuJoCo 3.6 Physics & WBC Utilities
==================================================
Cohesive rigid-body kinematics, contact GRF measurement, posture initialization,
and actuator mapping helpers shared across Figure 05 MuJoCo benchmarks.
"""

from typing import Iterable

import mujoco
import numpy as np

LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")


def mat_to_rotvec_err(r_des: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
    """Small-angle SO(3) orientation error vector from current to desired rotation matrix."""
    r_err = r_des @ r_cur.T
    return 0.5 * np.array([
        r_err[2, 1] - r_err[1, 2],
        r_err[0, 2] - r_err[2, 0],
        r_err[1, 0] - r_err[0, 1],
    ])


def rot_z(yaw: float) -> np.ndarray:
    """Return 3x3 SO(3) planar yaw rotation matrix about +Z [rad]."""
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wrap_angle_diff(target: float, current: float) -> float:
    """Wrap `(target - current)` into `[-pi, +pi]` radians cleanly."""
    return float(np.arctan2(np.sin(target - current), np.cos(target - current)))


def quat_to_roll_pitch_yaw(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo `[qw, qx, qy, qz]` quaternion into `(roll, pitch, yaw)` radians."""
    qw, qx, qy, qz = [float(v) for v in quat]
    roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx ** 2 + qy ** 2)))
    pitch = float(np.arcsin(np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)))
    yaw = float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy ** 2 + qz ** 2)))
    return roll, pitch, yaw


def build_actuator_mappings(model: mujoco.MjModel) -> dict:
    """Build index tables for leg WBC control and remaining upper-body posture actuators."""
    act_names = [model.actuator(i).name for i in range(model.nu)]
    act_qidx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    act_vidx = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    idx_map = {n: i for i, n in enumerate(act_names)}

    l_leg_v = [model.jnt_dofadr[model.joint(f"left_{j}").id] for j in LEG_JOINTS]
    r_leg_v = [model.jnt_dofadr[model.joint(f"right_{j}").id] for j in LEG_JOINTS]
    l_leg_act = [idx_map[f"act_left_{j}"] for j in LEG_JOINTS]
    r_leg_act = [idx_map[f"act_right_{j}"] for j in LEG_JOINTS]
    leg_set = set(l_leg_act) | set(r_leg_act)
    upper_act = [i for i in range(model.nu) if i not in leg_set]

    return {
        "act_names": act_names,
        "act_qidx": act_qidx,
        "act_vidx": act_vidx,
        "idx_map": idx_map,
        "l_leg_v": l_leg_v,
        "r_leg_v": r_leg_v,
        "l_leg_act": l_leg_act,
        "r_leg_act": r_leg_act,
        "upper_act": upper_act,
    }


def setup_crouched_biped(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    shoulder_pitch: float = 0.0,
    elbow: float = -0.45,
    torso_pitch: float = 0.0,
) -> None:
    """Initialize Figure 05 in a crouched athletic pose placed flush on `floor` (`z = 0`)."""
    def set_j(name: str, val: float) -> None:
        jid = model.joint(name).id
        data.qpos[model.jnt_qposadr[jid]] = val

    data.qpos[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 0.925]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    set_j("torso_pitch_joint", torso_pitch)
    for side in ("left", "right"):
        set_j(f"{side}_hip_pitch", -0.32)
        set_j(f"{side}_knee", 0.64)
        set_j(f"{side}_ankle_pitch", -0.32)
        set_j(f"{side}_shoulder_pitch", shoulder_pitch)
        set_j(f"{side}_elbow", elbow)

    mujoco.mj_forward(model, data)
    z_offset = 0.5 * (
        data.site("left_foot_contact").xpos[2]
        + data.site("right_foot_contact").xpos[2]
    )
    data.qpos[2] -= z_offset
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def sum_support_grfs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    support_geoms: Iterable[int],
    l_body: int,
    r_body: int,
) -> tuple[float, float]:
    """Sum true MuJoCo contact-frame vertical GRFs for left & right ankle bodies."""
    support_set = set(support_geoms)
    l_grf, r_grf = 0.0, 0.0
    cf = np.zeros(6)
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = contact.geom1, contact.geom2
        if g1 not in support_set and g2 not in support_set:
            continue
        other = g2 if g1 in support_set else g1
        other_body = int(model.geom_bodyid[other])
        mujoco.mj_contactForce(model, data, ci, cf)
        fz = abs(float(cf[0]))
        if other_body == l_body:
            l_grf += fz
        elif other_body == r_body:
            r_grf += fz
    return l_grf, r_grf


def measure_foot_world_grfs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    support_geoms: Iterable[int],
    l_body: int,
    r_body: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact 3D world GRF vectors `[fx, fy, fz]` acting ON left & right ankle bodies."""
    support_set = set(support_geoms)
    f_l = np.zeros(3, dtype=float)
    f_r = np.zeros(3, dtype=float)
    cf = np.zeros(6, dtype=float)
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = contact.geom1, contact.geom2
        if g1 not in support_set and g2 not in support_set:
            continue
        other = g2 if g1 in support_set else g1
        other_body = int(model.geom_bodyid[other])
        if other_body not in (l_body, r_body):
            continue
        mujoco.mj_contactForce(model, data, ci, cf)
        r_c = contact.frame.reshape(3, 3)
        sign = 1.0 if g1 in support_set else -1.0
        f_world = sign * (r_c.T @ cf[:3])
        if other_body == l_body:
            f_l += f_world
        else:
            f_r += f_world
    return f_l, f_r


def clamp_and_apply_ctrl(model: mujoco.MjModel, data: mujoco.MjData, tau_ctrl: np.ndarray) -> None:
    """Clip joint torque vector strictly within MJCF actuator ctrlrange limits and assign data.ctrl."""
    data.ctrl[:] = np.clip(tau_ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def compute_capture_point_dcm(
    com_xy: np.ndarray,
    com_vel_xy: np.ndarray,
    com_z: float,
    g: float = 9.80665,
) -> np.ndarray:
    """Compute 2D LIPM Divergent Component of Motion (Instantaneous Capture Point)."""
    omega0 = float(np.sqrt(g / max(float(com_z), 0.25)))
    return np.asarray(com_xy, dtype=float) + np.asarray(com_vel_xy, dtype=float) / omega0


def support_polygon_dcm_excess(
    dcm_xy: np.ndarray,
    p_left_xy: np.ndarray,
    p_right_xy: np.ndarray,
    foot_half_len: float = 0.085,
    foot_half_wid: float = 0.045,
    safety_inset_x: float = 0.0,
    safety_inset_y: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Return `(escape_dist_m, escape_vec_xy)` of Capture Point outside dual-sole bounds (with optional safety inset)."""
    x_lo = min(float(p_left_xy[0]), float(p_right_xy[0])) - foot_half_len + safety_inset_x
    x_hi = max(float(p_left_xy[0]), float(p_right_xy[0])) + foot_half_len - safety_inset_x
    if x_lo > x_hi:
        xm = 0.5 * (x_lo + x_hi)
        x_lo = x_hi = xm
    y_lo = min(float(p_left_xy[1]), float(p_right_xy[1])) - foot_half_wid + safety_inset_y
    y_hi = max(float(p_left_xy[1]), float(p_right_xy[1])) + foot_half_wid - safety_inset_y
    if y_lo > y_hi:
        ym = 0.5 * (y_lo + y_hi)
        y_lo = y_hi = ym
    clamped = np.array([
        np.clip(float(dcm_xy[0]), x_lo, x_hi),
        np.clip(float(dcm_xy[1]), y_lo, y_hi),
    ], dtype=float)
    diff = np.asarray(dcm_xy, dtype=float) - clamped
    return float(np.linalg.norm(diff)), diff


def measure_geom_pair_world_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geoms: Iterable[int],
    robot_geoms: Iterable[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Return `(total_force_on_robot_3d, per_robot_geom_force_3d)` via native `mj_contactForce`."""
    target_set = set(target_geoms)
    robot_set = set(robot_geoms)
    total_f = np.zeros(3, dtype=float)
    per_geom = {int(gid): np.zeros(3, dtype=float) for gid in robot_set}
    cf = np.zeros(6, dtype=float)
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1, g2 = contact.geom1, contact.geom2
        if g1 in target_set and g2 in robot_set:
            sign = 1.0
            r_gid = g2
        elif g2 in target_set and g1 in robot_set:
            sign = -1.0
            r_gid = g1
        else:
            continue
        mujoco.mj_contactForce(model, data, ci, cf)
        r_c = contact.frame.reshape(3, 3)
        f_world = sign * (r_c.T @ cf[:3])
        total_f += f_world
        per_geom[int(r_gid)] += f_world
    return total_f, per_geom
