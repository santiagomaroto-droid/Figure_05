# Figure 05: Deterministic Communication Bus Specification
## Real-Time Single-Pair Ethernet (1000BASE-T1) Dual Counter-Rotating Ring

**Author / Lead Researcher:** Santiago Maroto  
**Document Revision:** v13.0.0 | **Protocol:** *Figure-Sync Real-Time Protocol (FSRP v3.2)*  
**Target Hardware:** Distributed GaN Joint Inverters & Central RTOS Microkernel  

---

```
                       FIGURE-SYNC DETERMINISTIC RING TOPOLOGY
                    ============================================

                               Central Brain (Dual Drive Thor)
                              ┌──────────────────────────────┐
                              │  gPTP Master Clock (±25 ns)  │
                              └──────────────┬───────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │ Primary Ring (Clockwise / 1000BASE-T1)    │
                       ▼                                           ▼
             ┌──────────────────┐                         ┌──────────────────┐
             │ Node 01: Hip Yaw │                         │ Node 56: Hand    │
             └────────┬─────────┘                         └────────▲─────────┘
                      │                                            │
                      ▼                                            │ Secondary Ring
             ┌──────────────────┐                                  │ (Counter-CW)
             │ Node 02: Hip Roll│                                  │ [Auto-Healing]
             └────────┬─────────┘                                  │
                      │                                            │
                      └──────────────────────► ◄───────────────────┘
```

---

## 1. Physical Layer: Single-Pair Ethernet (1000BASE-T1)

Humanoid robots face severe packaging constraints. Conventional EtherCAT and multi-conductor harnesses require 8 to 12 conductors per joint, leading to mechanical bulk, bending fatigue, and joint binding.

Figure 05 standardizes on **IEEE 802.3bp (1000BASE-T1)**:
- **Conductors:** Exactly **one unshielded twisted pair (UTP)** for high-speed differential bidirectional 1 Gbps data.
- **Power Delivery:** Power-over-Data-Lines (PoDL / IEEE 802.3bu) delivering 48V auxiliary logic power over the same single pair.
- **Continuous Rotary Joints:** High-frequency gold-on-gold slip rings with through-bore Ø12 mm hollow shafts. No external wire harnesses exposed to pinch points or abrasion.

---

## 2. Clock Synchronization: IEEE 802.1AS-Rev (gPTP)

All 56 distributed GaN inverters must sample phase currents and latch encoder positions synchronously to prevent phase jitter in Whole-Body Control:
- **Master Grandmaster Clock:** Lockstep ARM Cortex-R82 RTOS hardware timer with temperature-compensated crystal oscillator (TCXO).
- **Network Jitter:** **$< 35\text{ ns}$** maximum phase deviation across all 56 nodes over the full ring traversal.
- **Synchronized Current Sampling:** All 56 FOC motor inverters sample phase currents simultaneously at the exact start of every $20\text{ kHz}$ PWM cycle.

---

## 3. Real-Time Telemetry Packet Frame Layout (64 Bytes Deterministic)

Every joint node exchanges a compact 64-byte telemetry and command datagram every **$100\text{ µs}$ ($10\text{ kHz}$ update rate)**:

```
Byte Offset      Field Name            Type          Resolution / Range
──────────────────────────────────────────────────────────────────────────────────
0x00 - 0x03      PTP Timestamp         uint32        1.0 nanosecond LSB
0x04 - 0x07      Absolute Joint Pos    int32         20-bit (0.00034° / LSB)
0x08 - 0x09      Joint Velocity        int16         0.01 rad/s LSB (±327.67 rad/s)
0x0A - 0x0B      Target Torque Cmd     int16         0.05 Nm LSB (±1638.4 Nm)
0x0C - 0x0D      Torque Feedback       int16         0.05 Nm LSB (Strain gauge bridge)
0x0E - 0x0F      Phase Current I_rms   uint16        0.01 A LSB (0 to 655.35 A)
0x10 - 0x11      DC Bus Voltage        uint16        0.05 V LSB (0 to 1000.0 V)
0x12 - 0x13      GaN Junction Temp     int16         0.1 °C LSB (-40°C to +200°C)
0x14 - 0x15      Stator Core Temp      int16         0.1 °C LSB (-40°C to +200°C)
0x16 - 0x17      Status & Error Word   uint16        Bitfield (OCP, OVP, OTP, Slip)
0x18 - 0x3B      Tactile / Sensor Ext  byte[36]      GelSkin 3D vector fields (Hands)
0x3C - 0x3F      CRC-32 Checksum       uint32        IEEE 802.3 Ethernet Polynomial
```

---

## 4. Fault-Tolerant Auto-Healing Ring Topology (< 4 µs Failover)

In harsh industrial environments, a joint disconnection must never cause a robot collapse:
- **Dual Counter-Rotating Ring:** Primary ring routes data clockwise; secondary ring routes data counter-clockwise.
- **Hardware Physical-Layer Link Loss Detection:** Rather than waiting multiple slow upper-layer control ticks, if any joint PHY detects continuous 1000BASE-T1 PAM3 carrier/eye-opening loss in `< 2.4 µs`:
  - The upstream and downstream transceivers loop back their TX/RX lines directly in hardware within a total of **$3.8\text{ µs}$** (verified across 56 nodes in `simulate_safety_and_bus_hil.py`).
  - The ring seamlessly morphs into a bidirectional bus line within a single `100 µs` cycle budget.
  - Zero dropped `10 kHz` control frames; the robot maintains stable balancing without safety trip.
