# FIGURE 05 // OFFICIAL ENGINEERING DATASHEET
## High-Performance Anthropomorphic General-Purpose Humanoid

**Document Revision:** v13.0.0  
**Author / Lead Researcher:** Santiago Maroto  
**Target Enterprise:** Figure AI Industrial Operations & Hardware Engineering  
**Classification:** Open Architecture Technical Specification  

---

```
                               FIGURE 05 WORKSPACE ENVELOPE
                            ==================================

                                         ┌─────┐
                                         │(o o)│  Head: 2-DoF Neck (±90° Yaw, ±45° Pitch)
                                     ┌───┴─────┴───┐
                                     │   TORSO     │  Payload: 35 kg cont. / 45 kg deadlift
               7-DoF Left Arm        │  [2.2 kWh]  │  7-DoF Right Arm
         ┌───────────────────────────┤  MONOCOQUE  ├───────────────────────────┐
         │ (Reach: 860 mm)           │   PELVIS    │ (Reach: 860 mm)           │
      ┌──▼──┐                        └──────┬──────┘                        ┌──▼──┐
      │ 22  │ 22-DoF Optical-Tactile        │ 6-DoF Hip/Knee (510 Nm)       │ 22  │
      │ DoF │ Hand (GelSkin 120 Hz)     ┌───┴───┐                           │ DoF │
      └─────┘                           │       │                           └─────┘
                                    ┌───▼──┐ ┌──▼───┐
                                    │ LEG  │ │ LEG  │ Stride: 0.65 m (1.8 m/s sprint)
                                    │  L   │ │  R   │ Shift Autonomy: 8.5 Hours
                                    └──┬───┘ └──┬───┘
                                     [SOLES]  [SOLES] Ground Clearance: 65 mm
```

---

## 1. Physical & Mechanical Envelope

| Metric | Specification | Verification Standard |
| :--- | :--- | :--- |
| **Total Standing Height** | 1,720 mm (5 ft 8 in) | 50th percentile human male stature |
| **Total Robot Mass (Dry)** | **54.5 kg** (120.1 lbs) | Integrated Cell-to-Chassis (CTC) Monocoque |
| **Degrees of Freedom (Total)** | **56 Active Actuators** | Driving 75 anatomical DoF (incl. 22-DoF hands) |
| • *Upper Body (Neck + Torso)* | 5 DoF (2 Neck, 3 Lumbar) | ±90° Yaw, -35°/+70° Pitch, ±28° Roll (Parallel Spine) |
| • *Arms (per limb)* | 7 DoF (3 Shoulder, 1 Elbow, 3 Wrist) | Reach Radius: 860 mm; Spherical Envelope: $2.1\text{ m}^3$ |
| • *Hands (per end-effector)* | **22 DoF (10 Active Tendon + 12 Coupled)** | 5-finger anthropomorphic, 10 optical tactile pads |
| • *Legs (per limb)* | 6 DoF (3 Hip, 1 Knee, 2 Ankle) | Class Gamma Compound Helical Planetary (18.00:1) |
| **Continuous Lift / Push Capacity** | **35.0 kg** lift / **65.0 kg** rolling push | Two-handed lift (`+1.27 m` pick-place) & unlatched `65.0 kg` cart push (`+1.308 m`, `119.5 kg` coupled, 0 N skyhook) |
| **Peak Dynamic Deadlift** | **45.0 kg** (99.2 lbs) | Verified in 3D MuJoCo (+38.1 cm lift, 975.7 N GRF, 66.8 mm sole margin, 0 N skyhook) |
| **Maximum Walking Speed** | 1.80 m/s (6.5 km/h) | Kajita LQR stride: 0.85 m/s (40.0 mm ZMP); MuJoCo 3D DCM walk verified: 2.438 m/8s + 70 N mid-stride push OK (4.19° Roll RMS, 0 N skyhook) |
| **Ground Incline / Stair Step** | 10°–15° ramp / ±140 mm stairs / μ=0.18 oil / 45 mm cleat | Verified 10.0° ramp, ±140 mm stairs (`206.5 J` regen), μ=0.18 oil (`2.081 m`), & 45 mm trip reflex (`1.995 m`, `61.5 mm` vault) |
| **Ingress Protection Rating** | **IP65** (Dust-tight, water jet resistant) | Fully sealed through-bore joint cavities |

---

## 2. Powertrain & Actuation Specifications

| Parameter | Class Alpha (Small) | Class Beta (Medium) | Class Gamma (Heavy) |
| :--- | :--- | :--- | :--- |
| **Anatomical Locations** | Neck, Thumb Base | Shoulder, Elbow, Forearm, Wrist, Ankle Roll | Hip Yaw/Roll/Pitch, Knee, Ankle Pitch, Parallel Spine |
| **Gearing Architecture** | Planetary Single-Stage (10:1) | Compound Planetary (14:1) | **2-Stage Compound Helical Planetary (3.6 × 5.0 = 18.00:1)** |
| **Gear Material / Treatment** | 4340 Ni-Cr-Mo Nitrided | 8620 Carburized Steel | **AGMA Grade 3 Carburized AISI 9310 (AMS 6265, 60-62 HRC)** |
| **Surface Finish ($Ra$)** | $0.2\ \mu\text{m}$ | $0.15\ \mu\text{m}$ | **$< 0.1\ \mu\text{m}$ (Isotropic Superfinish)** |
| **Peak Torque (3s shock)** | 55.0 Nm | 220.0 Nm | **510.0 Nm** |
| **Continuous Torque** | 25.0 Nm | 90.0 Nm | **180.0 Nm** |
| **Specific Torque Density** | 24.1 Nm/kg | 38.6 Nm/kg | **48.2 Nm/kg** |
| **AGMA Tooth Stress (Peak)** | 210 MPa Bend (SF 2.3) | 340 MPa Bend (SF 1.4) | **111.4 MPa Bend (SF 4.62) / 908 MPa Contact (SF 1.89)** |
| **Bearing $L_{10h}$ Fatigue Life** | > 45,000 Hours | > 28,000 Hours | **273,514 Hours (Dual-Row Roller at 230 Nm Heavy RMS)** |
| **Inverter Power Stage** | 800V GaN / PoDL | 800V 3-Level GaN | **800V 3-Level Flying-Capacitor 650V GaN (95.24% Peak Eff)** |
| **Thermal Resistance ($R_{th}$)** | $1.10\text{ K/W}$ | $0.65\text{ K/W}$ | **$0.42\text{ K/W}$ (Direct-Chassis Vapor-Phase)** |
| **Position Sensing** | 19-bit BiSS-C Magnetic | 20-bit Dual Optical/Mag | **20-bit Dual BiSS-C (0.0003° resolution)** |

---

## 3. Power & Energy Storage Architecture

- **Pack Topology:** Cell-to-Chassis (CTC) structural monocoque pelvis and lower lumbar spine.
- **Battery Chemistry:** High-Nickel 4680 cylindrical cells in syntactic epoxy structural matrix.
- **Pack Nominal Energy:** **2.20 kWh** ($7.92\text{ MJ}$).
- **Nominal Bus Voltage:** **800 V DC** (Reduced $I^2R$ copper resistive losses across internal bus).
- **Shift Operating Autonomy:** **8.5 Hours** continuous mixed-cycle industrial shift (palletizing, walking, insertion).
- **Regenerative Deceleration:** Bidirectional active GaN buck-boost converters on knees and hips recovering **13.9% of gross actuation draw** (+123.2 Wh recovered per 8.5h shift -> 22.9% SoC reserve; `206.5 J` net 800V DC bus recovery verified across `-140 mm` stair descent).
- **Fast-Charge Protocol:** Automated autonomous docking via rear lumbar contact pads: **10% to 80% SoC in 18 minutes**.

---

## 4. End-Effector: "Apex-Dex" 22-DoF Optical-Tactile Hand

- **Kinematics:** 5 anthropomorphic digits. Thumb (4 active DoF), Index (4 DoF), Middle (4 DoF), Ring (4 DoF), Little (4 DoF), Palm Arch (2 DoF).
- **Actuation:** High-tensile Tungsten-Dyneema micro-tendons encapsulated inside metacarpal bones.
- **Tactile Sensor Skin:** **GelSkin 120 Hz** micro-lens CMOS optical tactile arrays on all 10 distal and medial phalanges.
- **Spatial Resolution:** Sub-0.05 mm 3D shear and normal deformation vector field.
- **Edge Slip Reflex Latency:** **1.75 ms** causal FPGA reflex trigger (arrests micro-slip within 0.26 mm with blind observer $\hat{\mu}=0.191$).
- **Pinch Force / Power Grasp:** Continuous pinch: $85\text{ N}$; Cylindrical power wrap: $320\text{ N}$.

---

## 5. Compute Engine & Deterministic Bus Architecture

```
Layer                   Hardware                  Cycle Rate        Latency Budget
──────────────────────────────────────────────────────────────────────────────────
1. VLA Foundation Model Dual Drive Thor SoCs (FP8)   50 Hz (20 ms)    7.35 ms (p99: 7.75 ms)
2. Whole-Body Controller Lockstep ARM Cortex-R82    1000 Hz (1.0 ms)   781.9 µs (p99: 956.5 µs)
3. Local Digit Reflex   Xilinx Artix UltraScale+   500 Hz (2.0 ms)    1.75 ms
4. Motor Inverter FOC   Integrated GaN Gate Nodes 20,000 Hz (50 µs)   12.0 µs
```

- **Internal Data Bus:** Single-Pair Ethernet (**1000BASE-T1**) in redundant dual counter-rotating ring topology.
- **Through-Bore Slip Rings:** Continuous 360° rotation on waist, neck, and shoulders with zero wire twist fatigue.
- **BOM Manufacturing Target:** **$17,878 USD** at 50,000 units/year volume.
