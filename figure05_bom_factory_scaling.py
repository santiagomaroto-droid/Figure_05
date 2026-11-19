"""
Figure 05 - Industrial Bill of Materials (BOM) & Factory Scaling Engine
======================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Group
Target: Brett Adcock & Kyle Edelberg, Figure AI

Calculates:
1. Itemized component-level BOM across all 56 DoF and sub-assemblies.
2. Volume scaling economics from 1k to 50k units/year.
3. Validates the $18,450 USD factory BOM target at scale.
4. Generates publication-quality manufacturing cost curves and CSV dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_factory_bom():
    print("=================================================================")
    print("FIGURE 05: INDUSTRIAL BILL OF MATERIALS & FACTORY SCALING ENGINE")
    print("=================================================================")

    # Subsystem items: [Category, Subsystem, Qty, Cost_1k, Cost_10k, Cost_50k]
    bom_data = [
        # Actuation Family
        ["Actuation", "Class Gamma Actuator (Hip, Knee, Lumbar - 510 Nm)", 12, 1150.0, 540.0, 310.0],
        ["Actuation", "Class Beta Actuator (Shoulder, Elbow, Ankle - 220 Nm)", 16, 750.0, 360.0, 205.0],
        ["Actuation", "Class Alpha Actuator (Neck, Wrist, Forearm - 55 Nm)", 8, 380.0, 190.0, 115.0],
        ["Actuation", "Apex-Dex Micro-Tendon Actuators (In-Metacarpal)", 20, 160.0, 75.0, 42.0],
        
        # Power & Structural Energy Storage
        ["Power", "2.2 kWh Structural Pelvis Monocoque (4680 Arrays)", 1, 3600.0, 1850.0, 1200.0],
        ["Power", "Integrated 800V GaN Distribution PDU & Inverters", 1, 2200.0, 1100.0, 720.0],
        ["Power", "Active Regenerative Braking Unit & Bidirectional DC-DC", 1, 880.0, 440.0, 280.0],

        # Compute & Perception
        ["Compute", "Dual Drive Thor SoCs (2000 TFLOPS FP8) + Carrier Board", 1, 3900.0, 2600.0, 1850.0],
        ["Compute", "Dual Lockstep Cortex-R82 Real-Time Microkernel Board", 1, 780.0, 390.0, 240.0],
        ["Perception", "Wide-Baseline Stereo RGB-D RealSense Arrays (Head & Torso)", 4, 340.0, 170.0, 105.0],
        ["Perception", "Apex-Dex GelSkin Optical-Tactile Finger Arrays (120 Hz)", 10, 210.0, 95.0, 58.0],
        ["Perception", "Dual BiSS-C 20-Bit Absolute Encoders & 6-Axis IMUs", 36, 110.0, 48.0, 28.0],

        # Structural Chassis & Materials
        ["Structure", "Topologically Optimized 7075-T6 Unibody Castings", 1, 4400.0, 2100.0, 1280.0],
        ["Structure", "Compression-Molded Carbon Fiber Exterior Fairings", 1, 2600.0, 1300.0, 790.0],
        ["Structure", "Titanium Ti-6Al-4V Joint Pivot Spindles & Hardware", 1, 1450.0, 750.0, 440.0],
        ["Harnessing", "Single-Pair Ethernet (1000BASE-T1) SPE Slip-Ring Bus", 1, 1050.0, 520.0, 310.0]
    ]

    columns = ["Category", "Component", "Qty", "UnitCost_1k", "UnitCost_10k", "UnitCost_50k"]
    df = pd.DataFrame(bom_data, columns=columns)

    # Calculate Total Line Costs
    df["Total_1k"] = df["Qty"] * df["UnitCost_1k"]
    df["Total_10k"] = df["Qty"] * df["UnitCost_10k"]
    df["Total_50k"] = df["Qty"] * df["UnitCost_50k"]

    # Export to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "figure05_detailed_bom.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved detailed CSV BOM to: {csv_path}")

    # Subsystem category rollups at 50k scale
    category_summary = df.groupby("Category")["Total_50k"].sum().sort_values(ascending=False)
    
    total_cost_1k = df["Total_1k"].sum()
    total_cost_10k = df["Total_10k"].sum()
    total_cost_50k = df["Total_50k"].sum()

    print(f"[*] BOM Cost at 1,000 units/year:   ${total_cost_1k:,.2f} USD")
    print(f"[*] BOM Cost at 10,000 units/year:  ${total_cost_10k:,.2f} USD")
    print(f"[*] BOM Cost at 50,000 units/year:  ${total_cost_50k:,.2f} USD (Target: $18,450.00)")

    assert not df.isna().any().any(), "BOM CSV contains NaN entries!"
    assert total_cost_1k > total_cost_10k > total_cost_50k, "Economy-of-scale learning curve must decrease monotonically!"
    assert total_cost_50k <= 18450.0, f"50k BOM exceeded $18,450 ceiling: ${total_cost_50k:,.2f}"

    # Generate Visualizations
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#07090E')

    for ax in axs:
        ax.set_facecolor('#0F172A')
        ax.tick_params(colors='#94A3B8')
        ax.xaxis.label.set_color('#E2E8F0')
        ax.yaxis.label.set_color('#E2E8F0')
        ax.title.set_color('#38BDF8')
        for spine in ax.spines.values():
            spine.set_color('#334155')

    # Subplot 1: Production Volume Scaling Curve
    volumes = [500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    # Learning curve model: C(V) = C0 * (V / V0)^b
    b_learning = np.log(total_cost_50k / total_cost_1k) / np.log(50000 / 1000)
    costs_curve = total_cost_1k * (np.array(volumes) / 1000.0) ** b_learning

    axs[0].plot(volumes, costs_curve / 1000.0, marker='o', color='#38BDF8', lw=2.8, markersize=7, label='Unit Manufacturing Cost')
    axs[0].axhline(y=18.450, color='#10B981', linestyle='--', lw=1.8, label='Target Parity ($18,450 USD)')
    axs[0].set_xscale('log')
    axs[0].set_title('Figure 05 Unit Manufacturing Cost vs Annual Volume', fontsize=11, fontweight='bold')
    axs[0].set_xlabel('Annual Production Volume (Units/Year)')
    axs[0].set_ylabel('Total Robot BOM Cost ($k USD)')
    axs[0].legend(facecolor='#1E293B', edgecolor='#475569', labelcolor='#F8FAFC')
    axs[0].grid(True, color='#1E293B', which='both')
    axs[0].annotate(f'50,000 units: ${total_cost_50k:,.0f}', 
                    xy=(50000, total_cost_50k/1000.0), xytext=(20000, 32),
                    color='#F8FAFC', fontweight='bold',
                    arrowprops=dict(facecolor='#10B981', edgecolor='#10B981', arrowstyle="->", lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#1E293B', edgecolor='#10B981'))

    # Subplot 2: Subsystem Cost Breakdown at 50,000 Units/Year
    labels = category_summary.index.tolist()
    values = category_summary.values.tolist()
    colors_pie = ['#38BDF8', '#0284C7', '#10B981', '#F59E0B', '#A855F7', '#EC4899']

    bars = axs[1].bar(labels, values, color=colors_pie[:len(labels)], edgecolor='#38BDF8', width=0.55)
    axs[1].set_title(f'BOM Cost Distribution at 50k Scale (Total: ${total_cost_50k:,.0f})', fontsize=11, fontweight='bold')
    axs[1].set_ylabel('Subsystem Cost ($ USD)')
    axs[1].tick_params(axis='x', rotation=25)
    axs[1].grid(axis='y', color='#1E293B')
    for b in bars:
        yval = b.get_height()
        axs[1].text(b.get_x() + b.get_width()/2.0, yval + 150, f'${yval:,.0f}', ha='center', va='bottom', color='#F8FAFC', fontweight='bold', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(__file__), "figure05_manufacturing_cost_curve.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved high-resolution Manufacturing Cost Curve to: {chart_path}")

if __name__ == "__main__":
    calculate_factory_bom()
