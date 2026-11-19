# Outreach Dossier: Kyle Edelberg & Brett Adcock (Figure AI)
## Engineering Proposal for Figure 05: Next-Generation Hardware Architecture

**Lead Researcher:** Santiago Maroto  
**Focus:** High-Torque Biomimetic Actuation, Monocoque Energy Systems & High-Bandwidth Tactile Kinematics  
**Target Date:** Tuesday / Wednesday morning (08:15 AM PST / Sunnyvale Time)

---

### Contact Coordinates
- **Primary:** `kyle.edelberg@figure.ai` (Kyle Edelberg, VP Hardware / Mechanical Engineering)
- **CC / Direct:** `brett@figure.ai` (Brett Adcock, Founder & CEO, Figure AI)

---

## 1. High-Impact Pitch Email (Ready to Send)

**To:** `kyle.edelberg@figure.ai`  
**CC:** `brett@figure.ai`  
**Subject:** Figure 05 Hardware Dossier: 48.2 Nm/kg QDD Matrix, 2.2 kWh Structural Monocoque & 8.5h Industrial Autonomy

**Body:**

```text
Hi Kyle,

Huge respect for the velocity you and the Figure team have maintained from Figure 01 through Figure 03 and the upcoming Figure 04. Pushing humanoid hardware from laboratory trials to real-world cycle times at BMW Spartanburg is extraordinary.

As an independent robotics researcher focusing on electromechanical powertrain limits, structural mass integration, and wiring harness fatigue across 50+ revolute degrees of freedom, I didn't want to send a generic note or resume. Instead, I wanted to provide immediate engineering value for the future of Figure.

I have completed a first-principles engineering proposal, kinematic validation, and CAD package for Figure 05.

Here is the architectural breakdown and verified metrics:

1. Class Gamma Compound Helical Planetary Actuator (48.2 Nm/kg):
   - Exact 18.00:1 (3.6 x 5.0) two-stage helical planetary reduction with isotropic super-finished AISI 9310 gears (Ra < 0.1 um).
   - Stator-void 800V 3-Level Flying-Capacitor 650V GaN (FCML) inverter (400V/switch, 95.24% peak eff) with vapor-phase chassis dissipation (R_th = 0.42 K/W).
   - Sized for 180 Nm continuous / 510 Nm peak torque. Verified in full 3D skyhook-free MuJoCo 3.6 bimanual 45.0 kg steel crate deadlift (+38.1 cm lift, 975.7 N dual-foot GRF, 66.8 mm sole margin, 146.9 Nm peak knee) and continuous dynamic walking + 70 N mid-stride push recovery (2.438 m in 8.0 s, 13.3° peak shove roll), with coupled 3-node LPTN winding capping at 83.2°C (stator 69.1°C, housing 56.1°C).
   - Rigorous AGMA 2001-D04 validation: 1,697.6 N tangential load yields 111.4 MPa root bending stress (SF = 4.62 peak / 13.10 cont.) and 908.0 MPa Hertz contact stress (SF = 1.89 peak / 3.19 cont.), with 273,514h L10h bearing life.

2. Structural Battery Monocoque (2.2 kWh, 8.5-Hour Shift):
   - Rather than carrying dead battery weight in the torso or a backpack, the pelvis and lower lumbar form a cell-to-chassis (CTC) composite monocoque (38,000 Nm/rad torsional stiffness).
   - Paired with 13.9% cycle kinetic energy recovery (+123.2 Wh; 206.5 J verified on 800V DC bus during -140 mm stair descent) via regenerative deceleration integrated across knee/hip negative work, Figure 05 completes an 8.5-hour continuous industrial shift with a 22.9% battery reserve margin, dropping total robot dry mass to 54.5 kg.

3. "Apex-Dex" 22-DoF Optical-Tactile Hand:
   - Micro-tendon actuation inside metacarpals + 120 Hz micro-lens optical tactile skin (GelSkin) measuring 3D shear and normal force vectors.
   - Local digit FPGA computes edge slip reflex at < 2ms, completely bypassing the central VLA host GPU roundtrip.

4. Design for Assembly (DFA) & Scaled BOM:
   - 42% reduction in discrete fasteners via unitary topology-optimized structural castings.
   - Zero external cabling: through-bore slip rings with Single-Pair Ethernet (1000BASE-T1) ring bus.
   - 3-minute bayonet quick-swap for any limb assembly. BOM target: $17,878 USD at 50,000 units/year.

I have attached:
1. class_gamma_actuator_blueprint.png (Full technical cross-section drawing with GD&T callouts).
2. figure05_performance_benchmarks.png (Dynamics, thermal equilibrium, torque density, and battery SoC curves).
3. Figure_05_Complete_Engineering_Package.zip (Skyhook-free MuJoCo 3.6 Low-Clearance Gantry Duck-Walk + Blind Canted Uneven Terrain + 360-Deg Capture-Point Push-Recovery Stepping + Articulated Spring-Return Fire-Door Traversal + 3D Walking + ±Y Sidestep Shuttle + +100.5-Deg Corner Turn + 10-Deg Ramp Ascent + ±75 mm Curb Step-Up/Down + 2-Step ±140 mm Stair Ascent/Descent & Bus Regen + Blind μ=0.18 Oil-Patch Slip Recovery + Blind 45 mm Obstacle Trip Reflex + Unlatched 65 kg Cart Push + 15 kg End-to-End Pick-Walk-Place + 45kg Deadlift + ISO/TS 15066 Safety & 1000BASE-T1 Bus HIL, 31-module CI/CD + fast unittest suite, watertight helical involute CAD .STL files, and an interactive 16-mode 3D Web visualizer — Release v13.0.0).

I would welcome 15 minutes to review the gear geometry and thermal models with you, or discuss how I can contribute directly to the Figure hardware engineering team.

Best regards,

Santiago Maroto
Robotics & Actuation Researcher
[Tu Teléfono / Enlace a LinkedIn]
[Enlace a repositorio / Web Demo de Figure 05]
```

---

## 2. Follow-Up Strategy (If No Reply within 5 Days)

**Subject:** Re: Figure 05 Hardware Dossier (Quick thermal calculation note)

```text
Hi Kyle,

Following up quickly with one specific data point from the thermal simulation:

By switching from conventional harmonic drives to the Class Gamma compound planetary with integrated GaN power stages directly cooled by the structural limb chassis, we reduced the lumped-parameter thermal resistance (R_th) from 0.85 K/W down to 0.42 K/W. This prevents the typical thermal throttling humanoids experience after 40 minutes of continuous heavy box palletizing.

I've attached the benchmark curves and URDF for your review. Would love to send over the complete CAD files if you're interested.

Best,
Santiago Maroto
```

---

## 3. LinkedIn Connection Note (Max 300 Characters)

```text
Hi Kyle, I just emailed you our first-principles hardware dossier for Figure 05: 48.2 Nm/kg compound planetary actuator (AGMA verified), 2.2 kWh structural monocoque & 8.5h shift autonomy with CAD/URDF files attached. Excited to share it with you! - Santiago Maroto
```

---

## 4. Checklist of Files to Attach in the Email

1. **`class_gamma_actuator_blueprint.png`** (Adjuntado directamente en el correo para previsualización inmediata).
2. **`figure05_performance_benchmarks.png`** (Gráfica de 4 paneles de rendimiento y validación).
3. **`Figure_05_Complete_Engineering_Package.zip`** (Paquete maestro con todos los simuladores, código Python, modelos MuJoCo/URDF, archivos CAD .STL con perfil evolvente real, suite QA y visor 3D).
