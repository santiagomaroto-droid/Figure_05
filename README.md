# PROJECT FIGURE 05: The Apex Humanoid Engineering Dossier

> **Author & Lead Researcher:** Santiago Maroto  
> **Release Version:** `v13.0.0` ([VERSION](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/VERSION) | [CHANGELOG.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/CHANGELOG.md)) — 31/31 CI/CD Verified  
> **Prepared for:** Brett Adcock (CEO) & Kyle Edelberg (VP Hardware / Mechanical Eng.), Figure AI  
> **Mission:** Solving the 5 critical bottlenecks of autonomous humanoid deployment at 100,000+ unit scale.

---

## 🚀 Key Highlights & Architectural Leaps

| Feature | Figure 02 / 03 Standard | Figure 05 Solution |
| :--- | :--- | :--- |
| **System Mass** | 62 - 70 kg | **54.5 kg** (Structural composite unibody) |
| **Shift Runtime** | 4.0 - 5.5 hours | **8.5 hours continuous** (2.2 kWh Pack, 22.9% reserve, 83.2°C 3-node LPTN) |
| **Actuator Torque Density** | 16.5 Nm/kg (Harmonic) | **48.2 Nm/kg** (Class Gamma 18:1 Compound Planetary + GaN) |
| **AGMA Gear Stress Verification**| Standard catalog specs | **111.4 MPa Bend (SF 4.62) / 908.0 MPa Contact (SF 1.89) on 9310 Steel** |
| **Physics Simulation Engine** | Kinematics only | **Skyhook-Free MuJoCo 3.6: 3D Walk+70N Shove (2.44 m), +100.5° Turn (2.46 m), 10° Ramp (+29.3 cm), ±75 mm Curb Step-Up/Down, 2-Step ±140 mm Stair Ascent/Descent (2.39 / 2.37 m, 206.5 J DC bus regen), 15 kg End-to-End Pick-Walk-Place (+1.27 m), 45kg Deadlift & 120N Shock** |
| **Bipedal Gait & Sim2Real** | Open-loop step tables | **Kajita DARE Preview (0.85 m/s, 40 mm ZMP margin) + 96-World Blind MuJoCo DR (97.9%, 1.64° RMS)** |
| **Tactile Manipulation** | Rigid fingertip force cells | **22-DoF Apex-Dex Hand + Causal GelSkin Reflex (1.75 ms FPGA -> 0.26 mm slip, mu_hat 0.191)** |
| **Neural Stack Execution** | CPU/GPU mixed latency | **Dual Drive Thor (7.35 ms VLA mean) + 10k-Tick RTOS WBC (781.9 us mean, 956.5 us p99, 0 misses)** |
| **Target BOM at 50k units/yr** | > $38,000 USD | **$17,878 USD** (-42% discrete fasteners, unified castings) |

---

## 📂 Complete Project Contents

### 1. Architectural Blueprints & Whitepapers
* **[FIGURE_05_EXECUTIVE_PITCH_DECK.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/FIGURE_05_EXECUTIVE_PITCH_DECK.md)**: 10-slide executive presentation and due diligence deck for Figure AI leadership.
* **[FIGURE_05_OFFICIAL_DATASHEET.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/FIGURE_05_OFFICIAL_DATASHEET.md)**: Industrial product datasheet detailing envelopes, payload, IP65 ingress, and connector pinouts.
* **[SAFETY_AND_COMPLIANCE_SPEC.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/SAFETY_AND_COMPLIANCE_SPEC.md)**: ISO 13482:2014 & ISO/TS 15066 Cobot Safety Specification (PFL <140N, Safe Torque Off <1.5 µs, Category 4 PLe / SIL-3).
* **[COMMUNICATION_BUS_SPEC.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/COMMUNICATION_BUS_SPEC.md)**: Figure-Sync Protocol (FSRP) over Single-Pair Ethernet (1000BASE-T1) dual counter-rotating ring (<4 µs auto-heal).
* **[FIGURE_05_MASTER_BLUEPRINT.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/FIGURE_05_MASTER_BLUEPRINT.md)**: 7-chapter comprehensive engineering whitepaper.
* **[NEURAL_ARCHITECTURE.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/NEURAL_ARCHITECTURE.md)**: End-to-end 50 Hz VLA foundation model + 1 kHz RTOS Whole-Body Control specification.
* **[KYLE_EDELBERG_PITCH_DOSSIER.md](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/KYLE_EDELBERG_PITCH_DOSSIER.md)**: Surgical outreach emails and tactical follow-ups for Figure leadership.

### 2. Physics & Simulation Engines
* **[.github/workflows/ci_validation.yml](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/.github/workflows/ci_validation.yml)**: GitHub Actions automated CI/CD workflow running the `< 3 s` unit regression gate and all 31 physics, CAD, safety & consistency checks on push (`requirements.txt`).
* **[requirements.txt](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/requirements.txt)** & **[LICENSE](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/LICENSE)**: Pinned scientific Python/MuJoCo runtime dependencies and MIT open-engineering license.
* **[run_all_validations.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/run_all_validations.py)** & **[test_figure05_fast.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/tests/test_figure05_fast.py)**: Master automated CI/CD test suite verifying all 31 engineering modules in sequence (100% pass) plus `< 3 s` fast unit regression gate (`23` tests).
* **[figure05_mujoco_common.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_mujoco_common.py)**: Shared rigid-body WBC kinematics, Capture-Point (DCM) support-margin analytics, multi-geom contact wrench integration, quaternion conversions, posture initialization, and contact-frame GRF summation utilities.
* **[verify_digital_twin_consistency.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/verify_digital_twin_consistency.py)**: Cross-artifact auditor verifying MuJoCo == URDF 54.50 kg mass, watertight binary STL meshes, 0 N skyhook across all telemetry JSONs, ISO/bus HIL invariants, and $17,878 BOM CSV integrity.
* **[simulate_safety_and_bus_hil.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_safety_and_bus_hil.py)**: MuJoCo 3.6 ISO/TS 15066 sternum collision (`56.9 N`, `0.0052 J` vs `987.6 N` rigid) + 56-node 1000BASE-T1 cable-cut failover (`3.80 us`, `29.7 ns` gPTP jitter, `0` dropped frames).
* **[figure05_sim2real_domain_randomization.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_sim2real_domain_randomization.py)**: Blind zero-oracle parallel MuJoCo 3.6 domain randomization across 96 perturbed physics rollouts (97.9% survival, 1.64° mean RMS).
* **[figure05_automotive_cycle_benchmark.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_automotive_cycle_benchmark.py)**: BMW 60s Takt-Time workcell + 3D chamfer insertion ODE & 1,000-cycle Monte Carlo (16.35s nominal, p95 17.01s, 98.06% OEE, 0.000 mm final err).
* **[figure05_mujoco.xml](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_mujoco.xml)**: Native MuJoCo 3.6+ model balanced to exact 54.50 kg mass with contact soles and actuator limits.
* **[simulate_figure05_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_figure05_mujoco.py)**: Skyhook-free Centroidal WBC & Jacobian wrench projection (rejects 120 N chest shock purely via measured MuJoCo foot contact GRFs).
* **[simulate_push_recovery_stepping_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_push_recovery_stepping_mujoco.py)**: Skyhook-free MuJoCo 3.6 360° omnidirectional high-impulse (`165 N × 220 ms = 36.3 N·s`) Capture-Point reactive stepping A/B benchmark across all 4 cardinal directions (`0° +X`, `180° -X`, `90° +Y`, `270° -Y`). Baseline fixed-stance falls on **4/4** directions (`0/4`); causal zero-oracle DCM stepper survives **4/4** (`67.0 ms` mean trigger, `0.131 m` min clearance, `0.002–0.039 m/s` final CoM speed, `212.3 Nm` peak knee).
* **[simulate_walking_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_walking_mujoco.py)**: Skyhook-free MuJoCo 3.6 Englsberger DCM dynamic walking + 70 N mid-stride push recovery (2.438 m over 8.0 s, 0.349 m/s steady, 4.19° roll / 2.36° pitch RMS, 5.80 cm clearance).
* **[simulate_ramp_walk_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_ramp_walk_mujoco.py)**: Skyhook-free MuJoCo 3.6 continuous 10.0° (17.6% grade) dock ramp ascent (1.917 m distance, +29.3 cm vertical elevation gain, 2.03° roll / 2.02° pitch RMS).
* **[simulate_stepup_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_stepup_mujoco.py)**: Skyhook-free MuJoCo 3.6 discrete `+75.0 mm` raised pallet curb step-up & upper-deck cruise (2.126 m distance, 1.18° roll / 4.05° pitch RMS, `173.9 Nm` peak knee).
* **[simulate_stepdown_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_stepdown_mujoco.py)**: Skyhook-free MuJoCo 3.6 discrete `-75.0 mm` pallet curb descent onto lower floor (`1.893 m` over `5.998 s`, `2.94°` roll / `3.98°` pitch RMS, `126.5 Nm` peak knee).
* **[simulate_stairs_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_stairs_mujoco.py)**: Skyhook-free MuJoCo 3.6 multi-riser 2-step `+140.0 mm` staircase ascent (`0 -> +70 -> +140 mm` across `2.394 m` in `6.998 s`, `+13.08 cm` pelvis climb, `1.24°` roll / `4.47°` pitch RMS, `154.0 Nm` peak knee).
* **[simulate_stair_descent_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_stair_descent_mujoco.py)**: Skyhook-free MuJoCo 3.6 multi-riser 2-step `-140.0 mm` staircase descent & 29-actuator electromechanical regen audit (`2.365 m` in `7.056 s`, `-14.95 cm` pelvis drop, **`206.5 J` net 800V DC bus regen**, `2.00°` roll / `3.16°` pitch RMS, `130.1 Nm` peak knee).
* **[simulate_slippery_walk_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_slippery_walk_mujoco.py)**: Skyhook-free MuJoCo 3.6 blind oil-patch (`X in [0.55, 1.55] m`, `mu = 1.20 -> 0.18 -> 1.20`) A/B slip recovery. Baseline skids (`2.883 m/s`) and crashes at `3.422 s`; zero-oracle causal slip observer drops `mu_hat -> 0.160`, clamps Coulomb cone + spin moment, clears all **`2.081 m`** (`3.61°` roll / `2.11°` pitch RMS), and recovers `mu_hat -> 0.450` on dry exit.
* **[simulate_trip_recovery_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_trip_recovery_mujoco.py)**: Skyhook-free MuJoCo 3.6 blind mid-swing `45.0 mm` rigid steel cleat A/B trip reflex (`X in [0.555, 0.605] m`). Baseline trips & crashes at `2.906 s`; zero-oracle lag reflex (`2.434 / 2.756 s`) vaults both legs up to `61.5 mm` and walks cleanly to **`1.995 m`** (`5.08°` roll / `3.12°` pitch RMS, `119.4 Nm` peak knee).
* **[simulate_loaded_walk_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_loaded_walk_mujoco.py)**: Skyhook-free MuJoCo 3.6 dynamic bimanual carry of a free-floating `15.0 kg` steel warehouse tote (`2.008 m` over `6.00 s`, `681.5 N` combined floor GRF vs `681.6 N` theoretical, `1.014 m` min tote height, `1.20°` roll / `2.42°` pitch RMS).
* **[simulate_locomanip_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_locomanip_mujoco.py)**: Skyhook-free MuJoCo 3.6 end-to-end 6-phase bimanual Pick -> Lift -> Walk -> Align -> Place -> Release (`15.0 kg` tote transported `+1.266 m` onto receiving workbench at `Z = 0.640 m` in `10.45 s`, unlatched at `t = 8.95 s`, `527.3 N` unloaded post-release GRF, `1.23°` roll / `2.38°` pitch RMS, `117.6 Nm` peak knee).
* **[simulate_cart_push_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_cart_push_mujoco.py)**: Skyhook-free MuJoCo 3.6 unlatched bimanual pushing of a **`65.0 kg`** 4-caster parts cart (**`119.5 kg`** coupled system, zero weld constraints) using causal palm contact force + Newton-3rd-law lean feedforward across a mid-aisle drag threshold (`+1.308 m` push, `30.1 N` mean active push, **`0.0 N`** clean release, `4.10°` roll / `4.37°` pitch RMS, `131.0 Nm` peak knee).
* **[simulate_door_traversal_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_door_traversal_mujoco.py)**: Skyhook-free MuJoCo 3.6 articulated spring-return fire-door push-open & walk-through A/B benchmark (`6.5 kg` door + `0.8 kg` crash bar, `3.0 Nm/rad` spring + `5.2 Nm·s/rad` hydraulic closer, `contype=4` torso collision shell). Trial A naive walk is physically blocked at `X = 0.793 m` (`440.1 N` impact); Trial B causal reactive push opens door to **`92.0°` (`133.2 N` peak palm)**, traverses to **`X = 1.867 m`** with Capture-Point deceleration braking (`1.41°` roll / `2.16°` pitch RMS, `128.1 Nm` peak knee), releases (`0.0 N`), and allows automatic door closing to **`1.6°`**.
* **[simulate_uneven_terrain_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_uneven_terrain_mujoco.py)**: Skyhook-free MuJoCo 3.6 blind multi-axis canted uneven terrain A/B benchmark across 12 staggered plates (`+18 mm .. +64 mm` center top elevations, `36 mm` L/R delta, up to `10.4°` compound roll/pitch slope, zero oracle). Trial A flat-world WBC falls at `2.956 s` (`0.74 m`); Trial B causal proprioceptive observer (`97.5 mm` detected relief, `15.9°` sole conformance) + compliant landing admittance traverses cleanly in `7.014 s` to **`2.93 m`** (`5.44°` roll / `4.28°` pitch RMS, **`139.9 Nm` peak knee**, `0.0 N` skyhook).
* **[simulate_gantry_duck_walk_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_gantry_duck_walk_mujoco.py)**: Skyhook-free MuJoCo 3.6 low-clearance (`1.410 m` steel soffit) overhead gantry duck-walk A/B benchmark using causal `rgbd_camera` depth rays (`0°..130°`), dynamic variable-height LIPM crouch (`-181.1 mm` CoM drop `0.875 -> 0.694 m`, `+34.2°` torso forward bow), and active `contype=4` head/chest collision shells. Trial A upright walk impacts the beam (`546.0 N`, `-43.9 mm` clearance, blocked at `0.844 m`); Trial B ducks cleanly under with **`0.0 N` collision (`50.2 mm` margin)**, traverses past the gantry (`6.696 s`, **`2.25 m`**, `171.8 Nm` peak knee, `1.63°` pitch RMS), and stands back upright (`0.843 m`).
* **[simulate_turning_walk_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_turning_walk_mujoco.py)**: Skyhook-free MuJoCo 3.6 SE(2) curved factory aisle corner walk (`+100.5°` turn across `2.459 m` in `8.00 s`, `1.21°` RMS yaw error, `3.44°` roll / `3.53°` pitch RMS, `133.6 Nm` peak knee).
* **[simulate_lateral_sidestep_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_lateral_sidestep_mujoco.py)**: Skyhook-free MuJoCo 3.6 frontal-plane assembly-line lateral sidestep shuttle (`+0.430 m` left -> **65.0 N** fore-aft chest shove -> clean return `-0.009 m`, **`1.413 m`** total `±Y` travel, `0.153 m` anti-scissor gap, `0.53°` workcell yaw RMS).
* **[simulate_deadlift_mujoco.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/simulate_deadlift_mujoco.py)**: Skyhook-free MuJoCo 3.6 bimanual 45.0 kg steel crate deadlift (+38.1 cm lift, 975.7 N dual-foot GRF equilibrium, 66.8 mm sole margin, 146.9 Nm peak knee).
* **[figure05_gait_generator.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_gait_generator.py)**: Kajita DARE LQR ZMP Preview Control & 3D Leg IK (0.85 m/s C² smooth walk, 40.0 mm ZMP margin, 7.02 rad/s max knee speed).
* **[figure05_tactile_grasp_sim.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_tactile_grasp_sim.py)**: Apex-Dex 22-DoF causal tactile slip reflex simulator (no oracle leak: arrests micro-slip at 0.26 mm under oil drop + 1.5g jerk, mu_hat 0.191).
* **[figure05_kinematics.urdf](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_kinematics.urdf)**: 56-DoF standard URDF compatible with Isaac Sim / Lab and Pinocchio.
* **[figure05_dynamics_sim.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_dynamics_sim.py)**: 45 kg deadlift torque profiles, coupled 3-node LPTN with temperature-dependent `alpha_Cu` (`83.2°C / 69.1°C / 56.1°C`), and 8.5h SoC shift autonomy (`22.9%` reserve, `13.9%` regen).

### 3. CAD Meshes & Engineering Drawings
* **[class_gamma_sun_gear.stl](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/class_gamma_sun_gear.stl)**: 3D Watertight Binary STL of the 16-tooth helical involute sun pinion with Ø12mm through-bore.
* **[class_gamma_housing.stl](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/class_gamma_housing.stl)**: 3D Watertight Binary STL of the 64-tooth internal involute ring housing with cooling fins & ISO flange.
* **[class_gamma_planetary_assembly.stl](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/class_gamma_planetary_assembly.stl)**: 3D Watertight Binary STL of the 4x 24-tooth helical involute planet carrier assembly.
* **[figure05_monocoque_pelvis.stl](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_monocoque_pelvis.stl)**: 3D Watertight Binary STL of the structural battery monocoque pelvis with 4680 pockets.
* **[class_gamma_actuator_blueprint.png](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/class_gamma_actuator_blueprint.png)**: High-resolution cross-section blueprint with GD&T callouts.

### 4. Neural Benchmarks & Factory BOM Scaling
* **[figure05_neural_stack.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_neural_stack.py)**: Dual Thor FP8 VLA inference & 1 kHz WBC cycle latency benchmark.
* **[figure05_bom_factory_scaling.py](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_bom_factory_scaling.py)**: Full 50,000 units/year volume learning curve calculator.
* **[figure05_detailed_bom.csv](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/figure05_detailed_bom.csv)**: Itemized component-by-component costing dataset.

### 5. Interactive 3D Web Visualizer
* **[index.html](file:///c:/Users/717/.gemini/antigravity/scratch/Figure_05/index.html)**: Interactive WebGL visualizer featuring:
  - Thermal Hotspot Heatmap view.
  - 18:1 Compound Planetary Actuator cutaway with rotating gearset & bimanual crate payload animation.
  - Interactive joint angle sliders (live inverse kinematics).
  - Multi-tab engineering matrix (MuJoCo telemetry, interactive BOM volume scaler, and neural latency pyramid).

---

## Quickstart: Run Fast Unit Gate & Full 31-Module Suite Locally

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_all_validations.py
```

---

## Verified Benchmark Gallery (`v13.0.0`)

### Class Gamma Actuator Cross-Section & AGMA Grade 3 Stress Blueprint
![Class Gamma Actuator Blueprint](class_gamma_actuator_blueprint.png)

### Skyhook-Free MuJoCo 3.6 360° Push-Recovery Stepping, Spring-Door Traversal, Blind Canted Uneven Terrain, 1.41 m Gantry Duck-Walk, Walk, +100.5° Turn, 10.0° Ramp, ±75 mm Step-Up/Down, ±140 mm Stairs, 15 kg Tote Walk & Pick-Walk-Place
![MuJoCo 3D 360-Deg Capture-Point Push-Recovery A/B](figure05_mujoco_push_recovery_benchmark.png)
![MuJoCo 3D Articulated Spring-Return Door Traversal A/B](figure05_mujoco_door_traversal_benchmark.png)
![MuJoCo 3D Blind Canted Uneven Terrain A/B](figure05_mujoco_uneven_terrain_benchmark.png)
![MuJoCo 3D 1.41 m Overhead Gantry Duck-Walk A/B](figure05_mujoco_gantry_duck_benchmark.png)
![MuJoCo 3D Dynamic Walking + Push Recovery](figure05_mujoco_walking_benchmark.png)
![MuJoCo 3D Curved +100.5-Deg Aisle Turn](figure05_mujoco_turning_benchmark.png)
![MuJoCo 3D Bidirectional Lateral Sidestep Shuttle](figure05_mujoco_sidestep_benchmark.png)
![MuJoCo 3D 10-Deg Ramp Ascent](figure05_mujoco_ramp_benchmark.png)
![MuJoCo 3D +75 mm Pallet Curb Step-Up](figure05_mujoco_stepup_benchmark.png)
![MuJoCo 3D -75 mm Pallet Curb Step-Down](figure05_mujoco_stepdown_benchmark.png)
![MuJoCo 3D 2-Step +140 mm Staircase Ascent](figure05_mujoco_stairs_benchmark.png)
![MuJoCo 3D 2-Step -140 mm Staircase Descent & Bus Regen](figure05_mujoco_stair_descent_benchmark.png)
![MuJoCo 3D Blind Oil-Patch Slip Recovery & Friction-Cone WBC](figure05_mujoco_slippery_benchmark.png)
![MuJoCo 3D Blind Mid-Swing 45 mm Cleat Trip Reflex](figure05_mujoco_trip_benchmark.png)
![MuJoCo 3D 15 kg Tote Carry Walking](figure05_mujoco_loaded_walk_benchmark.png)
![MuJoCo 3D End-to-End 15 kg Pick-Walk-Place](figure05_mujoco_locomanip_benchmark.png)
![MuJoCo 3D Unlatched 65 kg Heavy Cart Push](figure05_mujoco_cart_push_benchmark.png)

### Skyhook-Free 45.0 kg Crate Deadlift & ISO/TS 15066 Collision + 1000BASE-T1 HIL
![MuJoCo 3D 45 kg Industrial Deadlift](figure05_mujoco_deadlift_benchmark.png)
![ISO/TS 15066 Collision & 1000BASE-T1 Ring Failover](figure05_safety_bus_benchmark.png)
