# Figure 05: Hierarchical Neural & Real-Time Visuomotor Architecture
## End-to-End Vision-Language-Action (VLA) & 1 kHz Whole-Body Control Stack

**Target Audience:** Figure AI AI/Robotics Software Team, Kyle Edelberg & Brett Adcock  
**Lead Researcher:** Santiago Maroto  
**System Designation:** *Figure OS / Apex-Neural Stack v13.0.0*

---

```
                        FIGURE 05 NEURAL ARCHITECTURE
                  ========================================

          [Stereo RGB-D (60fps)]    [120Hz GelSkin Optical Tactile]   [Speech/Text Token]
                    │                              │                           │
                    └──────────────┬───────────────┴───────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Multimodal Spatial Encoder │  (Dual Drive Thor SoCs)
                    │  • Vision Transformer (ViT) │  2000 TFLOPS FP8
                    │  • Tactile Patch Tokenizer  │
                    └──────────────┬──────────────┘
                                   │ (50 Hz Policy Frequency)
                    ┌──────────────▼──────────────┐
                    │  VLA Action Head Transformer│
                    │  • Predicts 6D EEF Traj.    │
                    │  • Contact Mode Affordance  │
                    └──────────────┬──────────────┘
                                   │ (Deterministic Shared Memory IPC < 0.2ms)
                    ┌──────────────▼──────────────┐
                    │ 1 kHz Real-Time WBC Kernel  │  (Lockstep ARM Cortex-R82 RTOS)
                    │ • Contact-Implicit MPC      │
                    │ • Residual Physics RL       │
                    └──────────────┬──────────────┘
                                   │ (1000BASE-T1 SPE Ring Bus)
                    ┌──────────────▼──────────────┐
                    │  Distributed GaN Inverters  │  (20 kHz FOC Current Loop)
                    │  • Class Alpha / Beta /     │
                    │    Gamma Actuator Joints    │
                    └─────────────────────────────┘
```

---

## 1. The Dual-Horizon Control Dilemma in Humanoids

Humanoid commercial deployment demands two conflicting computational regimes:
1. **Semantic Generalization (Slow, Rich):** Understanding human intent, segmenting unmodeled clutter, identifying novel affordances, and planning complex multi-stage tasks ($10 - 50\text{ Hz}$).
2. **Contact Stability & Slip Reflex (Fast, Deterministic):** Preventing slippage of fragile objects ($< 2\text{ ms}$), responding to unexpected ground terrain compliant impacts ($1\text{ kHz}$), and executing field-oriented motor commutations ($20\text{ kHz}$).

Figure 05 solves this by cleanly decoupling **semantic affordance prediction** from **whole-body predictive mechanics** via a zero-copy lockstep interface.

---

## 2. Layer 1: The 50 Hz VLA Foundation Model (Apex-Helix)

- **Backbone:** 7-Billion parameter decoder-only transformer with cross-attention over fused visual and optical-tactile tokens.
- **Vision Modality:** Dual stereo 1280x720 RGB-D streams tokenized via a lightweight Patch14 vision encoder into 256 visual tokens per frame.
- **Tactile Modality (GelSkin 120 Hz):** 10 fingertip optical arrays (5 per hand) compress 3D displacement vectors into 40 compact tactile spatial tokens.
- **Action Chunking:** Outputs 16-step predictive chunks ($320\text{ ms}$ prediction horizon) of end-effector 6D poses, gripper aperture forces, and foot contact schedules at $50\text{ Hz}$.

---

## 3. Layer 2: 1 kHz Whole-Body Control & Residual Physics RL

- **Processor:** Dual lockstep ARM Cortex-R82 cores running a hardened real-time microkernel (Zephyr RTOS).
- **Control Law:** Contact-Implicit Model Predictive Control (CI-MPC) solved via quadratic programming (QP) at $1{,}000\text{ Hz}$ combined with a pre-trained **Residual Physics Reinforcement Learning (RL) policy**:
  $$\tau_{\text{cmd}} = \tau_{\text{MPC}}(\mathbf{q}, \dot{\mathbf{q}}, \mathbf{x}_{\text{ref}}) + \pi_{\text{residual}}(\mathbf{s}_t) + \mathbf{J}^T \mathbf{F}_{\text{contact}}$$
- **Latency Budget (10,000-Cycle Monte Carlo RTOS Benchmark):** Strict deterministic execution:
  - Sensor DMA ingest: $115\ \mu\text{s}$ mean
  - Pinocchio Spatial FK/ID: $175\ \mu\text{s}$ mean
  - OSQP CI-MPC Solver (warm/active-set jitter): $328\ \mu\text{s}$ mean
  - Residual RL ONNX Policy + SPE gPTP Bus: $164\ \mu\text{s}$ mean
  - **Total Loop Latency:** **$781.9\ \mu\text{s}$ mean ($956.5\ \mu\text{s}$ p99, $979.4\ \mu\text{s}$ p99.9 — 0 deadline misses across 10,000 ticks)**.

---

## 4. Layer 3: Sub-2ms Causal Edge Tactile Slip Reflex

The 22-DoF Apex-Dex hands feature an autonomous reflex circuit on the forearm FPGA (Xilinx Artix UltraScale+):
- Reads direct raw CMOS sub-ROI frame differences from the silicone GelSkin elastomer.
- Using causal delayed micro-slip velocity $v_{\text{slip}}$ and transmitted shear load $F_s$ (with **zero oracle knowledge** of surface friction $\mu$ or arm jerk), an online Stribeck friction observer ($\hat{\mu} = 0.191$ vs $0.22$ oily steel) and PI tendon reflex trigger in **$1.75\text{ ms}$**, arresting object slip at **$0.26\text{ mm}$** ($3{,}854\text{ mm}$ open-loop drop).
