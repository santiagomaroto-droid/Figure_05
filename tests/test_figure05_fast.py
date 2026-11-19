"""
Figure 05 Fast Standalone Regression Suite (< 3 s CI Gate)
=========================================================
Verifies deterministic kinematics/inertias, AGMA planetary assembly math,
STL manifold integrity, BOM volume scaling monotonicity, 3D chamfer insertion
convergence, version/changelog sync, and skyhook-free MuJoCo 3.6 sanity.
"""

import csv
import os
import re
import sys
import unittest

import mujoco
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from figure05_automotive_cycle_benchmark import simulate_chamfer_insertion
from figure05_gait_generator import solve_3dof_sagittal_leg_ik
from figure05_mujoco_common import (
    build_actuator_mappings,
    clamp_and_apply_ctrl,
    compute_capture_point_dcm,
    measure_foot_world_grfs,
    quat_to_roll_pitch_yaw,
    rot_z,
    setup_crouched_biped,
    sum_support_grfs,
    support_polygon_dcm_excess,
    wrap_angle_diff,
)
from generate_urdf import box_inertia, cyl_inertia, sphere_inertia
from simulate_cart_push_mujoco import (
    build_cart_push_mjcf,
    compute_push_lean_feedforward,
)
from simulate_door_traversal_mujoco import (
    DOOR_MASS_KG,
    DOOR_SPRING_NM_RAD,
    build_door_traversal_mjcf,
    compute_door_wbc_feedforward,
)
from simulate_gantry_duck_walk_mujoco import (
    GANTRY_SOFFIT_Z,
    build_gantry_duck_mjcf,
    causal_overhead_perception_step,
    variable_height_crouch_schedule,
)
from simulate_loaded_walk_mujoco import build_loaded_walk_model
from simulate_lateral_sidestep_mujoco import (
    anti_scissor_lateral_foothold,
    sidestep_velocity_schedule,
)
from simulate_locomanip_mujoco import build_locomanip_model, smoothstep
from simulate_push_recovery_stepping_mujoco import (
    compute_capture_foothold,
    select_reactive_swing_foot,
)
from simulate_slippery_walk_mujoco import (
    causal_friction_cone_clamp,
    terrain_friction_at,
    update_stance_slip_observer,
)
from simulate_stair_descent_mujoco import (
    compute_electromechanical_step_power,
    descent_level_at,
    descent_stair_height,
    stair_descent_swing_profile,
)
from simulate_stairs_mujoco import (
    snap_stair_foothold,
    stair_swing_elevation,
    stair_terrain_height,
)
from simulate_stepdown_mujoco import swing_descent_profile, upper_deck_height_at
from simulate_stepup_mujoco import (
    snap_safe_foothold,
    swing_elevation_profile,
    terrain_height_at,
)
from simulate_turning_walk_mujoco import yaw_schedule
from simulate_uneven_terrain_mujoco import (
    UNEVEN_TERRAIN_PLATES,
    build_uneven_terrain_mjcf,
    swing_probe_profile,
    terrain_elevation_observer_step,
)
from simulate_trip_recovery_mujoco import (
    CLEAT_HEIGHT_M,
    causal_trip_reflex_observer,
    trip_elevating_swing_profile,
)
from simulate_walking_mujoco import mat_to_rotvec_err
from verify_digital_twin_consistency import verify_binary_stl


class TestFigure05Fast(unittest.TestCase):
    """Fast deterministic sanity & invariant checks across Figure 05 subsystems."""

    def test_version_and_changelog_sync(self) -> None:
        with open(os.path.join(ROOT_DIR, "VERSION"), "r", encoding="utf-8") as f:
            version = f.read().strip()
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

        with open(os.path.join(ROOT_DIR, "CHANGELOG.md"), "r", encoding="utf-8") as f:
            changelog = f.read()
        match = re.search(r"## \[(\d+\.\d+\.\d+)\]", changelog)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), version)

    def test_compound_planetary_assembly_ratio(self) -> None:
        z_s1, z_p1, z_r1 = 20, 16, 52
        z_s2, z_p2, z_r2 = 16, 24, 64
        planets = 4
        self.assertEqual((z_s1 + z_r1) % planets, 0)
        self.assertEqual(z_s1 + 2 * z_p1, z_r1)
        self.assertEqual((z_s2 + z_r2) % planets, 0)
        self.assertEqual(z_s2 + 2 * z_p2, z_r2)
        ratio = (1.0 + z_r1 / z_s1) * (1.0 + z_r2 / z_s2)
        self.assertAlmostEqual(ratio, 18.0, places=6)

    def test_analytic_inertia_triangle_inequality(self) -> None:
        tensors = [
            box_inertia(2.0, 0.20, 0.30, 0.40),
            cyl_inertia(3.5, 0.08, 0.45),
            sphere_inertia(1.2, 0.09),
        ]
        for ixx, iyy, izz in tensors:
            self.assertGreater(ixx, 0.0)
            self.assertGreater(iyy, 0.0)
            self.assertGreater(izz, 0.0)
            self.assertGreater(ixx + iyy, izz)
            self.assertGreater(ixx + izz, iyy)
            self.assertGreater(iyy + izz, ixx)

    def test_so3_rotvec_error_identities(self) -> None:
        np.testing.assert_allclose(mat_to_rotvec_err(np.eye(3), np.eye(3)), np.zeros(3), atol=1e-12)
        ang = 0.05
        r_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ang), -np.sin(ang)],
            [0.0, np.sin(ang), np.cos(ang)],
        ])
        err = mat_to_rotvec_err(r_x, np.eye(3))
        self.assertAlmostEqual(float(err[0]), np.sin(ang), places=6)
        self.assertAlmostEqual(float(err[1]), 0.0, places=12)
        self.assertAlmostEqual(float(err[2]), 0.0, places=12)

    def test_stepup_planner_curb_exclusion(self) -> None:
        x_step = 0.34
        self.assertEqual(terrain_height_at(0.20, x_step=x_step, step_h=0.075), 0.0)
        self.assertEqual(terrain_height_at(0.35, x_step=x_step, step_h=0.075), 0.075)

        safe_pre = snap_safe_foothold(0.28, p_st_x=0.0, x_step=x_step)
        safe_post = snap_safe_foothold(0.36, p_st_x=0.20, x_step=x_step)
        self.assertLessEqual(safe_pre, x_step - 0.135)
        self.assertGreaterEqual(safe_post, x_step + 0.135)

        z0, _ = swing_elevation_profile(0.0, 0.0, 0.075, 0.45, x_step, 0.075, 0.085, 0.35)
        z_mid, _ = swing_elevation_profile(0.5, 0.0, 0.075, 0.45, x_step, 0.075, 0.085, 0.35)
        self.assertAlmostEqual(z0, 0.0, places=5)
        self.assertGreater(z_mid, 0.15)

    def test_stl_closed_manifold_fast(self) -> None:
        for fname in ("class_gamma_sun_gear.stl", "figure05_monocoque_pelvis.stl"):
            path = os.path.join(ROOT_DIR, fname)
            tris, vol_cm3 = verify_binary_stl(path)
            self.assertGreater(tris, 2000)
            self.assertGreater(vol_cm3, 10.0)

    def test_chamfer_insertion_ode_convergence(self) -> None:
        t_far, r_far, _, fz_far, _, _, t_cap_far = simulate_chamfer_insertion(r0_mm=2.75)
        t_near, r_near, _, fz_near, _, _, t_cap_near = simulate_chamfer_insertion(r0_mm=1.80)
        self.assertGreater(len(t_far), 100)
        self.assertLessEqual(float(r_far[-1]), 0.05)
        self.assertLessEqual(float(r_near[-1]), 0.05)
        self.assertAlmostEqual(float(fz_far[-1]), 35.0, places=1)
        self.assertAlmostEqual(float(fz_near[-1]), 35.0, places=1)
        self.assertGreater(float(t_cap_far), 0.0)
        self.assertLess(float(t_cap_near), float(t_cap_far))

    def test_bom_volume_discount_monotonic(self) -> None:
        bom_path = os.path.join(ROOT_DIR, "figure05_detailed_bom.csv")
        with open(bom_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 16)
        tot_1k = sum(float(r["Total_1k"]) for r in rows)
        tot_10k = sum(float(r["Total_10k"]) for r in rows)
        tot_50k = sum(float(r["Total_50k"]) for r in rows)
        self.assertLess(tot_50k, tot_10k)
        self.assertLess(tot_10k, tot_1k)
        for row in rows:
            self.assertGreater(int(row["Qty"]), 0)

    def test_mujoco_model_mass_and_zero_skyhook(self) -> None:
        xml_path = os.path.join(ROOT_DIR, "figure05_mujoco.xml")
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        self.assertAlmostEqual(float(np.sum(model.body_mass)), 54.50, places=2)
        self.assertEqual(model.nu, 29)
        data.qpos[2] = 0.925
        data.qpos[3] = 1.0
        for _ in range(50):
            data.qfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            self.assertTrue(np.all(np.isfinite(data.qpos)))
            self.assertTrue(np.all(np.isfinite(data.qvel)))
            self.assertEqual(float(np.max(np.abs(data.qfrc_applied))), 0.0)

    def test_mujoco_common_helpers_and_loaded_mass(self) -> None:
        roll, pitch, yaw = quat_to_roll_pitch_yaw(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertAlmostEqual(roll, 0.0, places=12)
        self.assertAlmostEqual(pitch, 0.0, places=12)
        self.assertAlmostEqual(yaw, 0.0, places=12)

        xml_path = os.path.join(ROOT_DIR, "figure05_mujoco.xml")
        model, data = build_loaded_walk_model(xml_path, tote_mass_kg=15.0)
        self.assertAlmostEqual(float(np.sum(model.body_mass)), 69.50, places=2)
        maps = build_actuator_mappings(model)
        self.assertEqual(len(maps["l_leg_act"]) + len(maps["r_leg_act"]) + len(maps["upper_act"]), model.nu)
        setup_crouched_biped(model, data)
        l_grf, r_grf = sum_support_grfs(
            model,
            data,
            (model.geom("floor").id,),
            model.body("left_ankle").id,
            model.body("right_ankle").id,
        )
        self.assertGreaterEqual(l_grf, 0.0)
        self.assertGreaterEqual(r_grf, 0.0)

    def test_turning_schedule_and_se2_rotations(self) -> None:
        r90 = rot_z(0.5 * np.pi)
        np.testing.assert_allclose(r90 @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-12)
        np.testing.assert_allclose(r90.T @ r90, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(wrap_angle_diff(0.1 - np.pi, np.pi - 0.1), 0.2, places=6)

        y0, w0 = yaw_schedule(0.5)
        ym, wm = yaw_schedule(4.2)
        y1, w1 = yaw_schedule(8.0)
        self.assertEqual(y0, 0.0)
        self.assertEqual(w0, 0.0)
        self.assertGreater(ym, 0.0)
        self.assertGreater(wm, 0.0)
        self.assertAlmostEqual(float(np.rad2deg(y1)), 100.0, places=5)
        self.assertEqual(w1, 0.0)

    def test_stepdown_planner_bounds_and_profile(self) -> None:
        self.assertEqual(upper_deck_height_at(0.35, x_drop=0.40, step_h=0.075), 0.075)
        self.assertEqual(upper_deck_height_at(0.45, x_drop=0.40, step_h=0.075), 0.0)
        z0, _ = swing_descent_profile(0.0, 0.075, 0.0, 0.075, 0.080, 0.35)
        z_early, _ = swing_descent_profile(0.20, 0.075, 0.0, 0.075, 0.080, 0.35)
        z_end, _ = swing_descent_profile(1.0, 0.075, 0.0, 0.075, 0.080, 0.35)
        self.assertAlmostEqual(z0, 0.075, places=5)
        self.assertGreater(z_early, 0.075)
        self.assertLessEqual(z_end, 0.0)

    def test_stairs_height_and_riser_clearance(self) -> None:
        self.assertEqual(stair_terrain_height(0.20), 0.0)
        self.assertEqual(stair_terrain_height(0.55), 0.070)
        self.assertEqual(stair_terrain_height(0.95), 0.140)
        safe_1 = snap_stair_foothold(0.32, p_st_x=0.05, sw_start_x=0.05)
        self.assertGreaterEqual(safe_1, 0.34 + 0.135)
        safe_no_skip = snap_stair_foothold(0.85, p_st_x=0.50, sw_start_x=0.20)
        self.assertLessEqual(safe_no_skip, 0.78 - 0.135)
        z0, _ = stair_swing_elevation(0.0, 0.0, 0.070, 0.085, 0.35)
        z_early, _ = stair_swing_elevation(0.35, 0.0, 0.070, 0.085, 0.35)
        self.assertAlmostEqual(z0, 0.0, places=5)
        self.assertGreater(z_early, 0.14)

    def test_locomanip_model_and_initial_release(self) -> None:
        xml_path = os.path.join(ROOT_DIR, "figure05_mujoco.xml")
        model, data = build_locomanip_model(xml_path, tote_mass_kg=15.0)
        self.assertAlmostEqual(float(np.sum(model.body_mass)), 69.50, places=2)
        self.assertEqual(model.neq, 2)
        self.assertEqual(int(np.sum(data.eq_active)), 0)
        tote_id = model.body("warehouse_tote").id
        self.assertAlmostEqual(float(data.xpos[tote_id][0]), 0.27, places=3)
        self.assertAlmostEqual(float(data.xpos[tote_id][2]), 0.635, places=3)
        self.assertAlmostEqual(smoothstep(0.0), 0.0, places=12)
        self.assertAlmostEqual(smoothstep(0.5), 0.5, places=12)
        self.assertAlmostEqual(smoothstep(1.0), 1.0, places=12)

    def test_stair_descent_and_regen_power_math(self) -> None:
        self.assertAlmostEqual(descent_stair_height(0.20), 0.140, places=6)
        self.assertAlmostEqual(descent_stair_height(0.60), 0.070, places=6)
        self.assertAlmostEqual(descent_stair_height(0.95), 0.0, places=6)
        self.assertEqual(descent_level_at(0.20), 2)
        self.assertEqual(descent_level_at(0.60), 1)
        self.assertEqual(descent_level_at(0.95), 0)

        z0, _ = stair_descent_swing_profile(0.0, 0.140, 0.070, 0.080, 0.35)
        z_early, _ = stair_descent_swing_profile(0.20, 0.140, 0.070, 0.080, 0.35)
        z_end, _ = stair_descent_swing_profile(1.0, 0.140, 0.070, 0.080, 0.35)
        self.assertAlmostEqual(z0, 0.140, places=5)
        self.assertGreater(z_early, 0.140)
        self.assertAlmostEqual(z_end, 0.069, places=5)

        tau = np.array([-40.0, 20.0])
        qd = np.array([2.0, 1.5])
        kt_n = np.array([3.33, 3.33])
        rw = np.array([0.0553, 0.0553])
        p_pos, p_neg, p_cu, p_regen = compute_electromechanical_step_power(tau, qd, kt_n, rw, eta_inv=0.9524)
        self.assertAlmostEqual(p_pos, 30.0, places=6)
        self.assertAlmostEqual(p_neg, 80.0, places=6)
        self.assertGreater(p_cu, 0.0)
        expected_regen = 80.0 * 0.9524 - 1.5 * ((-40.0 / 3.33) ** 2) * 0.0553
        self.assertAlmostEqual(p_regen, expected_regen, places=5)

    def test_slippery_observer_and_friction_cone_clamp(self) -> None:
        self.assertAlmostEqual(terrain_friction_at(0.20), 1.20, places=6)
        self.assertAlmostEqual(terrain_friction_at(1.00), 0.18, places=6)
        self.assertAlmostEqual(terrain_friction_at(1.90), 1.20, places=6)

        clamped = causal_friction_cone_clamp(np.array([300.0, 0.0]), fz=500.0, mu_hat=0.20, safety_factor=0.65)
        self.assertAlmostEqual(float(np.hypot(clamped[0], clamped[1])), 65.0, places=6)
        unclamped = causal_friction_cone_clamp(np.array([20.0, -10.0]), fz=500.0, mu_hat=0.80, safety_factor=0.65)
        self.assertAlmostEqual(float(unclamped[0]), 20.0, places=6)

        # Causal slip drops mu_hat sharply when loaded sole slides
        mu = 1.00
        for _ in range(15):
            mu = update_stance_slip_observer(mu, v_slip_mps=0.25, f_tangential_n=85.0, fz_n=500.0, dt=0.002)
        self.assertLess(mu, 0.28)
        # Sustained zero-slip loaded stance cautiously recovers mu_hat
        mu_low = mu
        for _ in range(250):
            mu = update_stance_slip_observer(mu, v_slip_mps=0.005, f_tangential_n=90.0, fz_n=500.0, dt=0.002)
        self.assertGreater(mu, mu_low + 0.05)

    def test_cart_push_model_and_lean_feedforward(self) -> None:
        xml_str = build_cart_push_mjcf(ROOT_DIR)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        setup_crouched_biped(model, data, shoulder_pitch=-0.55, elbow=-0.45, torso_pitch=0.02)
        cart_mass = float(model.body_mass[model.body("industrial_cart").id])
        self.assertAlmostEqual(cart_mass, 65.0, places=2)
        self.assertAlmostEqual(float(np.sum(model.body_mass)) - cart_mass, 54.5, places=1)

        # Verify world GRF vector helper & actuator saturation helper
        clamp_and_apply_ctrl(model, data, np.full(model.nu, 9999.0))
        self.assertTrue(np.all(data.ctrl <= model.actuator_ctrlrange[:, 1] + 1e-9))
        fl, fr = measure_foot_world_grfs(
            model, data, {model.geom("floor").id}, model.body("left_ankle").id, model.body("right_ankle").id
        )
        self.assertEqual(fl.shape, (3,))
        self.assertEqual(fr.shape, (3,))

        fx_ff, tau_ff, lean = compute_push_lean_feedforward(-50.0, 54.5, z_hand_m=1.02, z_com_m=0.87)
        self.assertAlmostEqual(fx_ff, 50.0, places=6)
        self.assertAlmostEqual(tau_ff, 0.15 * 50.0, places=6)
        self.assertGreater(lean, 0.05)
        self.assertLess(lean, 0.14)

    def test_trip_recovery_observer_and_vault(self) -> None:
        self.assertAlmostEqual(CLEAT_HEIGHT_M, 0.045, places=6)
        self.assertFalse(causal_trip_reflex_observer(0.45, 0.44, 0.40, 0.03, 0.40, False))
        self.assertTrue(causal_trip_reflex_observer(0.55, 0.51, 0.05, 0.03, 0.40, False))
        self.assertTrue(causal_trip_reflex_observer(0.0, 0.0, 0.0, 0.0, 0.99, True))

        x_early, z_early = trip_elevating_swing_profile(0.45, 0.40, 0.30, 0.68)
        self.assertLess(x_early, 0.30 + 0.5 * (1.0 - np.cos(np.pi * 0.45)) * 0.38)
        self.assertGreater(z_early, 0.080)
        x_end, z_end = trip_elevating_swing_profile(1.0, 0.40, 0.30, 0.68)
        self.assertAlmostEqual(x_end, 0.68, places=5)
        self.assertLess(z_end, 0.001)

    def test_sidestep_anti_scissor_and_3dof_ik(self) -> None:
        # Left swing (+1) must always land >= p_st_y + min_clear_y; Right swing (-1) <= p_st_y - min_clear_y
        y_l_min = anti_scissor_lateral_foothold(-5.0, 0.0, p_st_y=0.20, swing_side_sign=1, min_clear_y=0.15)
        y_r_max = anti_scissor_lateral_foothold(+5.0, 0.0, p_st_y=0.40, swing_side_sign=-1, min_clear_y=0.15)
        self.assertAlmostEqual(y_l_min, 0.35, places=6)
        self.assertAlmostEqual(y_r_max, 0.25, places=6)

        v_left, ph_l = sidestep_velocity_schedule(2.0)
        v_right, ph_r = sidestep_velocity_schedule(8.0)
        self.assertGreater(v_left, 0.10)
        self.assertEqual(ph_l, "SHUTTLE_LEFT")
        self.assertLess(v_right, -0.10)
        self.assertEqual(ph_r, "SHUTTLE_RIGHT")

        hip_p, knee_f, ankle_p = solve_3dof_sagittal_leg_ik(0.18, 0.665, l1=0.40, l2=0.40)
        self.assertGreater(float(knee_f), 0.1)
        self.assertAlmostEqual(float(hip_p + knee_f + ankle_p), 0.0, places=6)

    def test_capture_point_dcm_and_reactive_push_selection(self) -> None:
        g = 9.80665
        z_c = 0.875
        omega0 = float(np.sqrt(g / z_c))
        dcm = compute_capture_point_dcm(np.array([0.05, -0.02]), np.array([0.66, -0.33]), z_c, g=g)
        self.assertAlmostEqual(float(dcm[0]), 0.05 + 0.66 / omega0, places=6)
        self.assertAlmostEqual(float(dcm[1]), -0.02 - 0.33 / omega0, places=6)

        p_l = np.array([0.0, 0.11])
        p_r = np.array([0.0, -0.11])
        esc_in, vec_in = support_polygon_dcm_excess(np.array([0.02, 0.01]), p_l, p_r)
        self.assertAlmostEqual(esc_in, 0.0, places=9)
        self.assertAlmostEqual(float(np.linalg.norm(vec_in)), 0.0, places=9)

        esc_out, vec_out = support_polygon_dcm_excess(np.array([0.185, 0.0]), p_l, p_r, foot_half_len=0.085)
        self.assertAlmostEqual(esc_out, 0.10, places=6)
        self.assertAlmostEqual(float(vec_out[0]), 0.10, places=6)

        # Biomechanical non-crossing lead foot selection across 4 compass directions
        self.assertEqual(select_reactive_swing_foot(np.array([0.0, 0.12]), p_l, p_r, None), "L")
        self.assertEqual(select_reactive_swing_foot(np.array([0.0, -0.12]), p_l, p_r, None), "R")
        self.assertEqual(select_reactive_swing_foot(np.array([0.12, 0.0]), p_l, p_r, "L"), "R")
        self.assertEqual(select_reactive_swing_foot(np.array([-0.12, 0.0]), p_l, p_r, "R"), "L")

        # Anti-scissor foothold clamps enforce strict inter-foot lateral separation
        tgt_l = compute_capture_foothold(
            np.array([0.30, -0.50]), np.array([0.8, -0.5]), np.array([0.0, -0.11]),
            np.array([0.0, 0.11]), "L", omega0, min_clear_y=0.185,
        )
        tgt_r = compute_capture_foothold(
            np.array([-0.30, 0.50]), np.array([-0.8, 0.5]), np.array([0.0, 0.11]),
            np.array([0.0, -0.11]), "R", omega0, min_clear_y=0.185,
        )
        self.assertGreaterEqual(float(tgt_l[1] - (-0.11)), 0.185 - 1e-9)
        self.assertLessEqual(float(tgt_r[1] - 0.11), -0.185 + 1e-9)

    def test_door_traversal_model_and_wrench_feedforward(self) -> None:
        xml_str = build_door_traversal_mjcf(ROOT_DIR)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        setup_crouched_biped(model, data)

        pelvis_id = model.body("pelvis").id
        door_id = model.body("hinged_door").id
        hinge_id = model.joint("door_hinge").id
        self.assertAlmostEqual(float(model.body_subtreemass[pelvis_id]), 54.5, places=1)
        self.assertAlmostEqual(float(model.body_mass[door_id]), DOOR_MASS_KG + 0.8, places=2)
        self.assertAlmostEqual(float(model.jnt_stiffness[hinge_id]), DOOR_SPRING_NM_RAD, places=2)

        fx_ff, fy_ff, tau_p, tau_y = compute_door_wbc_feedforward(
            np.array([-50.0, -20.0, 0.0]), 54.5, z_hand_m=1.07, z_com_m=0.87,
            lever_y_m=0.21, f_left_x=-50.0, f_right_x=0.0,
        )
        self.assertAlmostEqual(fx_ff, 50.0, places=6)
        self.assertAlmostEqual(fy_ff, 20.0, places=6)
        self.assertAlmostEqual(tau_p, 10.0, places=6)
        self.assertAlmostEqual(tau_y, -10.5, places=6)

    def test_uneven_terrain_model_and_observer(self) -> None:
        xml_str = build_uneven_terrain_mjcf(ROOT_DIR)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        setup_crouched_biped(model, data)

        pelvis_id = model.body("pelvis").id
        self.assertAlmostEqual(float(model.body_subtreemass[pelvis_id]), 54.5, places=1)
        self.assertEqual(len(UNEVEN_TERRAIN_PLATES), 12)
        for gname, *_ in UNEVEN_TERRAIN_PLATES:
            gid = model.geom(gname).id
            self.assertGreater(int(gid), 0)

        # Casual terrain observer must reject unloaded, heel-off (>13.5 deg), and bouncing contacts
        self.assertAlmostEqual(terrain_elevation_observer_step(0.02, 0.08, 30.0, 4.0, 0.0), 0.02, places=9)
        self.assertAlmostEqual(terrain_elevation_observer_step(0.02, 0.08, 400.0, 18.0, 0.0), 0.02, places=9)
        self.assertAlmostEqual(terrain_elevation_observer_step(0.02, 0.08, 400.0, 4.0, 0.45), 0.02, places=9)
        updated = terrain_elevation_observer_step(0.0, 0.05, 400.0, 7.5, 0.02, alpha=0.20)
        self.assertAlmostEqual(updated, 0.01, places=6)

        # Swing profile clears unseen steps at mid-swing, probes dips at end, and latches on GRF
        z_apex = swing_probe_profile(0.50, 0.0, 0.04, 0.05, early_touchdown=False, z_latch=0.0, clear_h=0.125)
        self.assertGreaterEqual(z_apex, 0.174)
        z_probe = swing_probe_profile(1.0, 0.0, 0.04, 0.04, early_touchdown=False, z_latch=0.0, clear_h=0.125)
        self.assertLess(z_probe, 0.04 - 0.055)
        z_latched = swing_probe_profile(0.80, 0.0, 0.04, 0.05, early_touchdown=True, z_latch=0.068)
        self.assertAlmostEqual(z_latched, 0.065, places=6)

    def test_gantry_duck_model_and_perception(self) -> None:
        xml_str = build_gantry_duck_mjcf(ROOT_DIR)
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        setup_crouched_biped(model, data)

        pelvis_id = model.body("pelvis").id
        self.assertAlmostEqual(float(model.body_subtreemass[pelvis_id]), 54.5, places=1)
        gantry_gid = model.geom("gantry_beam").id
        self.assertEqual(int(model.geom_group[gantry_gid]), 1)
        self.assertEqual(int(model.geom_contype[model.geom("head_geom").id]), 4)
        self.assertEqual(int(model.geom_contype[model.geom("chest_geom").id]), 4)
        self.assertAlmostEqual(GANTRY_SOFFIT_Z, 1.41, places=3)

        zc_up, om_up, tp_up, np_up = variable_height_crouch_schedule(0.0)
        zc_dn, om_dn, tp_dn, np_dn = variable_height_crouch_schedule(1.0)
        self.assertAlmostEqual(zc_up, 0.855, places=5)
        self.assertAlmostEqual(tp_up, 0.0, places=5)
        self.assertAlmostEqual(np_up, 0.0, places=5)
        self.assertAlmostEqual(zc_dn, 0.695, places=5)
        self.assertGreater(om_dn, om_up + 0.30)
        self.assertAlmostEqual(tp_dn, 0.56, places=5)
        self.assertLess(np_dn, -0.35)

        cam_site = model.site("rgbd_camera").id
        # Standing near beam entrance (x = 0.55) must trigger causal overhang detection
        data.qpos[0] = 0.55
        mujoco.mj_forward(model, data)
        seen_ahead, dist_ahead, hold_s = causal_overhead_perception_step(model, data, cam_site, False, 0.0, 0.002)
        self.assertTrue(seen_ahead)
        self.assertLess(dist_ahead, 0.72)
        self.assertGreater(hold_s, 1.0)

        # Far past beam (x = 2.00) with zero timer must clear crouch state
        data.qpos[0] = 2.00
        mujoco.mj_forward(model, data)
        seen_far, _, hold_far = causal_overhead_perception_step(model, data, cam_site, True, 0.0, 0.002)
        self.assertFalse(seen_far)
        self.assertEqual(hold_far, 0.0)


if __name__ == "__main__":
    unittest.main()
