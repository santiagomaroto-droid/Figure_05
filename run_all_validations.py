"""
Figure 05 - Master System Validation & Quality Assurance Test Suite
===================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Group
Target: Quality Engineering & Technical Due Diligence, Figure AI

Executes and verifies all 31 engineering subsystems:
 1. Watertight Helical Involute CAD STL Mesh Generator (generate_actuator_stl.py)
 2. 56-DoF Standard Kinematic URDF Generator (generate_urdf.py)
 3. Dynamics, LPTN Thermal & Electromechanical Energy (figure05_dynamics_sim.py)
 4. AGMA 2001-D04 Gear Stress & ISO 281 Bearing Life (actuator_class_gamma_stress_analysis.py)
 5. Skyhook-Free MuJoCo 3.6 Centroidal WBC & 120N Shock (simulate_figure05_mujoco.py)
 6. Skyhook-Free MuJoCo 3.6 360-Deg Capture-Point Push Recovery (simulate_push_recovery_stepping_mujoco.py)
 7. Skyhook-Free MuJoCo 3.6 3D Walking & 70N Shove (simulate_walking_mujoco.py)
 8. Skyhook-Free MuJoCo 3.6 10-Deg Ramp Ascent (simulate_ramp_walk_mujoco.py)
 9. Skyhook-Free MuJoCo 3.6 +75mm Curb Step-Up (simulate_stepup_mujoco.py)
10. Skyhook-Free MuJoCo 3.6 -75mm Curb Step-Down (simulate_stepdown_mujoco.py)
11. Skyhook-Free MuJoCo 3.6 2-Step +140mm Stair Ascent (simulate_stairs_mujoco.py)
12. Skyhook-Free MuJoCo 3.6 2-Step -140mm Stair Descent & Regen (simulate_stair_descent_mujoco.py)
13. Skyhook-Free MuJoCo 3.6 Oil-Patch Slip Recovery & Cone WBC (simulate_slippery_walk_mujoco.py)
14. Skyhook-Free MuJoCo 3.6 Mid-Swing Obstacle Trip Reflex (simulate_trip_recovery_mujoco.py)
15. Skyhook-Free MuJoCo 3.6 15.0kg Tote Carry Walk (simulate_loaded_walk_mujoco.py)
16. Skyhook-Free MuJoCo 3.6 End-to-End 15kg Pick-Walk-Place (simulate_locomanip_mujoco.py)
17. Skyhook-Free MuJoCo 3.6 65.0kg Heavy Cart Bimanual Push (simulate_cart_push_mujoco.py)
18. Skyhook-Free MuJoCo 3.6 Spring-Return Door Traversal (simulate_door_traversal_mujoco.py)
19. Skyhook-Free MuJoCo 3.6 Blind Canted Uneven Terrain (simulate_uneven_terrain_mujoco.py)
20. Skyhook-Free MuJoCo 3.6 Low-Clearance Gantry Duck-Walk (simulate_gantry_duck_walk_mujoco.py)
21. Skyhook-Free MuJoCo 3.6 100-Deg Aisle Turn Walk (simulate_turning_walk_mujoco.py)
22. Skyhook-Free MuJoCo 3.6 Lateral Sidestep Shuttle (simulate_lateral_sidestep_mujoco.py)
23. Skyhook-Free MuJoCo 3.6 45kg Bimanual Crate Deadlift (simulate_deadlift_mujoco.py)
24. ISO/TS 15066 Collision Safety & 1000BASE-T1 Bus HIL (simulate_safety_and_bus_hil.py)
25. Kajita DARE LQR ZMP Preview Gait & 3-DoF Leg IK (figure05_gait_generator.py)
26. Causal Apex-Dex GelSkin Tactile Grasp Slip Reflex (figure05_tactile_grasp_sim.py)
27. Roofline VLA & 10,000-Tick RTOS Jitter Validator (figure05_neural_stack.py)
28. Parallel MuJoCo 96-World Sim2Real Domain Randomization (figure05_sim2real_domain_randomization.py)
29. Automotive Takt-Time, 3D Chamfer ODE & OEE Engine (figure05_automotive_cycle_benchmark.py)
30. Factory Scaling BOM & Learning Curve Engine (figure05_bom_factory_scaling.py)
31. Cross-Artifact Digital Twin & Version Auditor (verify_digital_twin_consistency.py)

Generates: FIGURE_05_SYSTEM_VALIDATION_REPORT.json
"""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

def run_master_qa_suite():
    print("=================================================================")
    print("FIGURE 05: MASTER SYSTEM VALIDATION & CI/CD TEST SUITE")
    print("Lead Researcher: Santiago Maroto")
    print("=================================================================")

    scripts = [
        ("CAD Involute STL Generator", "generate_actuator_stl.py"),
        ("56-DoF Kinematic URDF Generator", "generate_urdf.py"),
        ("Dynamics & Thermal LPTN", "figure05_dynamics_sim.py"),
        ("AGMA Gear Stress & Bearings", "actuator_class_gamma_stress_analysis.py"),
        ("MuJoCo 3.6 Skyhook-Free WBC", "simulate_figure05_mujoco.py"),
        ("MuJoCo 3.6 360-Deg Capture-Point Push Recovery", "simulate_push_recovery_stepping_mujoco.py"),
        ("MuJoCo 3.6 Skyhook-Free Walking", "simulate_walking_mujoco.py"),
        ("MuJoCo 3.6 10-Deg Ramp Ascent", "simulate_ramp_walk_mujoco.py"),
        ("MuJoCo 3.6 75mm Curb Step-Up", "simulate_stepup_mujoco.py"),
        ("MuJoCo 3.6 -75mm Curb Step-Down", "simulate_stepdown_mujoco.py"),
        ("MuJoCo 3.6 2-Step +140mm Stair Ascent", "simulate_stairs_mujoco.py"),
        ("MuJoCo 3.6 2-Step -140mm Stair Descent & Regen", "simulate_stair_descent_mujoco.py"),
        ("MuJoCo 3.6 Blind Oil-Patch Slip Recovery", "simulate_slippery_walk_mujoco.py"),
        ("MuJoCo 3.6 Blind Obstacle Trip Reflex", "simulate_trip_recovery_mujoco.py"),
        ("MuJoCo 3.6 15kg Tote Carry Walk", "simulate_loaded_walk_mujoco.py"),
        ("MuJoCo 3.6 End-to-End 15kg Pick-Walk-Place", "simulate_locomanip_mujoco.py"),
        ("MuJoCo 3.6 65.0kg Heavy Cart Bimanual Push", "simulate_cart_push_mujoco.py"),
        ("MuJoCo 3.6 Spring-Return Door Traversal", "simulate_door_traversal_mujoco.py"),
        ("MuJoCo 3.6 Blind Canted Uneven Terrain", "simulate_uneven_terrain_mujoco.py"),
        ("MuJoCo 3.6 Low-Clearance Gantry Duck-Walk", "simulate_gantry_duck_walk_mujoco.py"),
        ("MuJoCo 3.6 100-Deg Aisle Turn Walk", "simulate_turning_walk_mujoco.py"),
        ("MuJoCo 3.6 Lateral Sidestep Shuttle", "simulate_lateral_sidestep_mujoco.py"),
        ("MuJoCo 3.6 Skyhook-Free 45kg Deadlift", "simulate_deadlift_mujoco.py"),
        ("ISO 15066 Safety & 1000BASE-T1 HIL", "simulate_safety_and_bus_hil.py"),
        ("Kajita LQR Preview Gait & IK", "figure05_gait_generator.py"),
        ("GelSkin Tactile Grasp Reflex", "figure05_tactile_grasp_sim.py"),
        ("Dual Thor Neural Latency", "figure05_neural_stack.py"),
        ("MuJoCo 96-World Sim2Real DR", "figure05_sim2real_domain_randomization.py"),
        ("Automotive Takt Time & Reliability", "figure05_automotive_cycle_benchmark.py"),
        ("Factory Scaling BOM Engine", "figure05_bom_factory_scaling.py"),
        ("Cross-Artifact Digital Twin Auditor", "verify_digital_twin_consistency.py")
    ]

    results = []
    all_passed = True
    start_total = time.time()

    for name, script in scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        print(f"\n[RUNNING QA] {name} ({script})...")
        t0 = time.time()
        
        proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        t_elapsed = time.time() - t0
        
        passed = (proc.returncode == 0)
        status_str = "PASSED [OK]" if passed else "FAILED [ERROR]"
        print(f" -> {status_str} in {t_elapsed:.2f}s")
        
        if not passed:
            all_passed = False
            print(f"[ERROR TRACE]:\n{proc.stderr}")

        results.append({
            "subsystem": name,
            "script": script,
            "passed": passed,
            "execution_time_seconds": round(t_elapsed, 2)
        })

    total_time = time.time() - start_total
    print("\n=================================================================")
    print(f"MASTER QA SUITE FINISHED IN {total_time:.2f} SECONDS")
    print(f"OVERALL SYSTEM VALIDATION STATUS: {'ALL TESTS PASSED (100%)' if all_passed else 'FAILURES DETECTED'}")
    print("=================================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    version_file = os.path.join(base_dir, "VERSION")
    release_version = "unknown"
    if os.path.isfile(version_file):
        with open(version_file, "r", encoding="utf-8") as vf:
            release_version = vf.read().strip()

    runtime_versions = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas", "matplotlib", "mujoco"):
        try:
            runtime_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            runtime_versions[pkg] = "not_installed"

    report = {
        "project": "Figure 05 Apex Humanoid",
        "release_version": release_version,
        "lead_researcher": "Santiago Maroto",
        "qa_date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_versions": runtime_versions,
        "overall_status": "CERTIFIED_VALIDATED" if all_passed else "FAILED",
        "total_subsystems_tested": len(scripts),
        "tests_passed": sum(1 for r in results if r["passed"]),
        "total_test_duration_seconds": round(total_time, 2),
        "detailed_test_matrix": results
    }

    report_path = os.path.join(base_dir, "FIGURE_05_SYSTEM_VALIDATION_REPORT.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"[+] Master System Validation Report written to: {report_path}")

if __name__ == "__main__":
    run_master_qa_suite()
