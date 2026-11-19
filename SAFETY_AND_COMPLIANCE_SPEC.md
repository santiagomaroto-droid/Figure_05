# Figure 05: Functional Safety & Cobot Compliance Architecture
## ISO 13482:2014 & ISO/TS 15066 Industrial Collaborative Robotics Specification

**Author / Lead Researcher:** Santiago Maroto  
**Specification Revision:** v13.0.0  
**Classification:** Standardized Humanoid Collaborative Safety Dossier  
**Target Compliance:** BMW Spartanburg / Automotive Factory Floor Deployment  

---

```
                       FIGURE 05 SAFETY PERIMETER ZONES
                    ======================================

                             [Figure 05 Robot]
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │ ZONE 1: Collaborative Contact Zone (< 0.6 m)      │
           │ • Power & Force Limiting (PFL): F < 140 N         │
           │ • Compliance Skin: Dynamic Energy < 0.49 J        │
           └─────────────────────────┬─────────────────────────┘
                                     │
           ┌─────────────────────────▼─────────────────────────┐
           │ ZONE 2: Speed & Separation Monitoring (0.6 - 2.0 m)│
           │ • Speed de-rated dynamically (v < 0.45 m/s)       │
           │ • 3D RealSense + Wide-FOV LiDAR Envelope Tracking │
           └─────────────────────────┬─────────────────────────┘
                                     │
           ┌─────────────────────────▼─────────────────────────┐
           │ ZONE 3: Unrestricted Transit Zone (> 2.0 m)       │
           │ • Full nominal velocity (1.80 m/s sprint)         │
           └───────────────────────────────────────────────────┘
```

---

## 1. Regulatory Framework & Standards Alignment

Figure 05 is architected to eliminate safety cages and operate alongside humans on active automotive assembly lines:
1. **ISO 13482:2014:** Robots and robotic devices — Safety requirements for personal care robots (humanoids).
2. **ISO/TS 15066:2016:** Robots and robotic devices — Collaborative robots (PFL and SSM methodologies).
3. **ISO 13849-1:2023:** Safety of machinery — Safety-related parts of control systems (**Category 4, Performance Level e [PLe]**).
4. **IEC 61508:** Functional safety of electrical/electronic/programmable electronic safety-related systems (**SIL-3**).

---

## 2. Power and Force Limiting (PFL): Biomechanical Thresholds

In collaborative contact scenarios (accidental touch or hand-over), Figure 05 limits contact forces strictly below human pain threshold tables:

| Human Body Region | Maximum Permissible Dynamic Force | Maximum Energy Transfer | Figure 05 Hard Limit |
| :--- | :--- | :--- | :--- |
| **Chest / Sternum** | $140\text{ N}$ | $0.49\text{ J}$ | **$85\text{ N}$ (Impedance Cutoff)** |
| **Upper Arm / Shoulder** | $150\text{ N}$ | $0.52\text{ J}$ | **$95\text{ N}$** |
| **Forearm / Wrist** | $160\text{ N}$ | $0.56\text{ J}$ | **$105\text{ N}$** |
| **Hands / Fingers** | $140\text{ N}$ | $0.49\text{ J}$ | **$65\text{ N}$ (GelSkin Clamp Limit)** |

- **Compliant Skin (Passive):** 8mm dual-durometer elastomeric fairing covers structural aluminum limbs, distributing contact impact pressure to $< 35\text{ N/cm}^2$.
- **Virtual Torque Impedance (Active):** Real-time joint torque observer detects unmodeled external contact torques within **$1.2\text{ ms}$**, immediately transitioning all actuators into zero-gravity float mode.

---

## 3. Hardware-Level Dual-Channel Safety Functions

Figure 05 implements independent safety supervisor hardware isolated from the primary Linux/SoC compute boards:
- **Safe Torque Off (STO):** Redundant dual-channel solid-state disconnect pins on every distributed GaN inverter stage. Disconnects gate drive voltage in **$< 1.5\ \mu\text{s}$**, eliminating motor torque production regardless of software bugs.
- **Safe Operating Stop (SOS):** Actively holds position with zero velocity drift while maintaining motor torque under high external loads without tripping main contactors.
- **Safety Laser Time-of-Flight (ToF):** 360° base lidar sensors mounted in the pelvic chassis detect human intrusion at $30\text{ Hz}$ independent of visual occlusion.
