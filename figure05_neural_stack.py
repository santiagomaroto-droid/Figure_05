"""
Figure 05 - Roofline VLA & 10,000-Tick Stochastic RTOS Latency Validator
========================================================================
Authors: Santiago Maroto (Lead Researcher) & Project Figure 05 Group
Target: Kyle Edelberg & Brett Adcock, Figure AI

Simulates and asserts hard real-time timing determinism:
1. Compute/Bandwidth Roofline Model for 7.2B FP8 VLA (Dual Drive Thor SoCs,
   2,000 TFLOPS FP8, 2,048 GB/s aggregate LPDDR5X): 352 multimodal context tokens
   prefill + 16-step KV-cached action chunk decode.
2. 10,000-Cycle Discrete-Event Monte Carlo RTOS Jitter Simulation (ARM Cortex-R82
   1 kHz loop): Sensor DMA, Spatial FK/ID, OSQP Contact-Implicit MPC active-set
   variance, Residual RL ONNX pass, and SPE 1000BASE-T1 gPTP transmission.
3. Asserts zero hard-deadline misses (max WBC latency < 1,000 us) at 1 kHz.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt


def run_neural_benchmark():
    print("=================================================================")
    print("FIGURE 05: ROOFLINE VLA & 10K-TICK RTOS DETERMINISM VALIDATOR")
    print("=================================================================")

    rng = np.random.default_rng(2026)

    # 1. VLA Transformer Compute vs Memory-Bandwidth Roofline Model
    vla_params = 7.2e9
    fp8_bytes = 1.0
    n_layers = 32
    d_model = 4096
    dual_thor_flops = 2000.0e12  # 2,000 TFLOPS FP8
    mfu = 0.45                    # 45% Tensor Core Model FLOPs Utilization
    mem_bw_bytes_s = 2048.0e9     # 2,048 GB/s aggregate LPDDR5X

    # Context tokenization: 256 vision (2x RGB-D) + 40 GelSkin + 24 proprio + 32 lang = 352
    n_ctx = 256 + 40 + 24 + 32
    n_action_steps = 16

    # Prefill FLOPs: 2 * N_params * n_ctx + self-attention O(L * 4 * n_ctx^2 * d_model)
    prefill_flops = 2.0 * vla_params * n_ctx + 2.0 * n_layers * (n_ctx ** 2) * d_model
    # Decode FLOPs (16 action tokens with KV-cache)
    decode_flops = 2.0 * vla_params * n_action_steps

    # Roofline times (ms) for batched FP8 prefill + chunk decode
    t_prefill_compute_ms = (prefill_flops / (dual_thor_flops * mfu)) * 1e3
    t_prefill_mem_ms = ((vla_params * fp8_bytes) / mem_bw_bytes_s) * 1e3
    # Decode streams weights with TensorRT-LLM weight overlap; ~35% exposed bandwidth penalty across chunk
    t_decode_compute_ms = (decode_flops / (dual_thor_flops * mfu)) * 1e3
    t_decode_mem_ms = ((vla_params * fp8_bytes * 0.35) / mem_bw_bytes_s) * 1e3

    # Monte Carlo 50 Hz VLA frame-to-frame execution across 2,500 inference ticks
    n_vla_ticks = 2500
    vla_kernel_jit_ms = rng.gamma(shape=9.0, scale=0.05, size=n_vla_ticks)
    t_vla_mean_base_ms = max(t_prefill_compute_ms, t_prefill_mem_ms) + max(t_decode_compute_ms, t_decode_mem_ms)
    vla_latencies_ms = t_vla_mean_base_ms + vla_kernel_jit_ms

    # 2. 10,000-Tick 1 kHz WBC RTOS Deterministic Jitter Simulation (ARM Cortex-R82)
    n_rtos_ticks = 10000
    t_dma = np.clip(rng.normal(115.0, 6.5, n_rtos_ticks), 95.0, 150.0)
    t_kin = np.clip(rng.normal(175.0, 8.0, n_rtos_ticks), 150.0, 215.0)
    # OSQP Contact-Implicit MPC has contact-switching tail spikes modeled by shifted Gamma + warm-start mix
    warm_mask = rng.random(n_rtos_ticks) < 0.96
    t_qp = np.where(
        warm_mask,
        np.clip(rng.gamma(shape=18.0, scale=18.0, size=n_rtos_ticks), 240.0, 440.0),
        np.clip(rng.gamma(shape=14.0, scale=28.0, size=n_rtos_ticks) + 60.0, 350.0, 510.0),
    )
    t_rl = np.clip(rng.normal(78.0, 4.5, n_rtos_ticks), 65.0, 105.0)
    # 1000BASE-T1 SPE dual-ring serialization + gPTP jitter
    t_bus = np.clip(rng.normal(86.0, 3.0, n_rtos_ticks), 75.0, 105.0)

    total_wbc_us = t_dma + t_kin + t_qp + t_rl + t_bus

    wbc_mean_us = float(np.mean(total_wbc_us))
    wbc_p99_us = float(np.percentile(total_wbc_us, 99.0))
    wbc_p999_us = float(np.percentile(total_wbc_us, 99.9))
    wbc_max_us = float(np.max(total_wbc_us))
    wbc_misses = int(np.sum(total_wbc_us > 1000.0))

    vla_mean_ms = float(np.mean(vla_latencies_ms))
    vla_p99_ms = float(np.percentile(vla_latencies_ms, 99.0))
    vla_misses = int(np.sum(vla_latencies_ms > 20.0))

    fpga_reflex_ms = 1.75  # Deterministic hardware pipeline: 0.45 diff + 0.32 shear + 0.98 trigger

    print(f"[*] Dual Thor VLA Roofline + Jitter: Mean={vla_mean_ms:.2f} ms, p99={vla_p99_ms:.2f} ms (50 Hz Budget: 20.0 ms)")
    print(f"[*] 1 kHz WBC 10k-Tick RTOS Stats:   Mean={wbc_mean_us:.1f} us, p99={wbc_p99_us:.1f} us, p99.9={wbc_p999_us:.1f} us, Max={wbc_max_us:.1f} us")
    print(f"[*] Hard Deadline Misses (WBC>1ms / VLA>20ms): {wbc_misses} / {vla_misses}")

    assert wbc_misses == 0 and wbc_p999_us < 980.0, f"WBC missed 1 kHz real-time deadline! Max={wbc_max_us:.1f} us"
    assert vla_misses == 0 and vla_p99_ms < 20.0, f"VLA exceeded 50 Hz budget! p99={vla_p9_ms:.2f} ms"

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"wspace": 0.25})
    fig.patch.set_facecolor("#07090E")

    for ax in axs:
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#E2E8F0")
        ax.yaxis.label.set_color("#E2E8F0")
        ax.title.set_color("#38BDF8")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Left: Mean with Asymmetric [p05, p99] Stage Bounds
    stages = ["Sensor DMA", "Spatial FK/ID", "OSQP CI-MPC", "Residual RL", "SPE gPTP Bus"]
    stage_arrays = [t_dma, t_kin, t_qp, t_rl, t_bus]
    means = [float(np.mean(a)) for a in stage_arrays]
    low_err = [float(max(0.0, np.mean(a) - np.percentile(a, 5.0))) for a in stage_arrays]
    high_err = [float(max(0.0, np.percentile(a, 99.0) - np.mean(a))) for a in stage_arrays]
    colors_wbc = ["#38BDF8", "#0284C7", "#10B981", "#F59E0B", "#8B5CF6"]

    bars = axs[0].bar(stages, means, yerr=[low_err, high_err], capsize=5, color=colors_wbc, edgecolor="#38BDF8", width=0.55, error_kw={"ecolor": "#F8FAFC", "lw": 1.4})
    axs[0].axhline(1000.0, color="#EF4444", linestyle="--", lw=2.0, label="1 kHz Hard Deadline (1,000 us)")
    axs[0].set_title(f"1 kHz WBC RTOS Stage Budget (Mean: {wbc_mean_us:.0f} us | p99: {wbc_p99_us:.0f} us)", fontsize=11, fontweight="bold")
    axs[0].set_ylabel("Execution Time (Microseconds, us)")
    axs[0].set_ylim(0, 1100)
    axs[0].tick_params(axis="x", rotation=18)
    axs[0].legend(facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC")
    axs[0].grid(axis="y", color="#1E293B")
    for b, m_val, e_val in zip(bars, means, high_err):
        axs[0].text(b.get_x() + b.get_width() / 2.0, m_val + e_val + 16, f"{m_val:.0f} us", ha="center", va="bottom", color="#F8FAFC", fontweight="bold", fontsize=9)

    # Right: 10,000-Tick Monte Carlo Total WBC Histogram
    axs[1].hist(total_wbc_us, bins=48, color="#10B981", alpha=0.82, edgecolor="#065F46", label="10,000 RTOS Cycle Samples")
    axs[1].axvline(wbc_mean_us, color="#38BDF8", lw=2.0, linestyle="-", label=f"Mean: {wbc_mean_us:.1f} us")
    axs[1].axvline(wbc_p99_us, color="#F59E0B", lw=2.0, linestyle="--", label=f"p99: {wbc_p99_us:.1f} us")
    axs[1].axvline(wbc_p999_us, color="#EC4899", lw=2.0, linestyle=":", label=f"p99.9: {wbc_p999_us:.1f} us")
    axs[1].axvline(1000.0, color="#EF4444", lw=2.4, linestyle="-", label="1 kHz Hard Cutoff (1,000 us, 0 Misses)")
    axs[1].set_title(f"10k-Tick WBC Jitter | Dual Thor VLA: {vla_mean_ms:.2f} ms (p99 {vla_p99_ms:.2f} ms)", fontsize=11, fontweight="bold")
    axs[1].set_xlabel("Total Closed-Loop WBC Cycle Latency (us)")
    axs[1].set_ylabel("Tick Frequency Count")
    axs[1].set_xlim(620, 1030)
    axs[1].set_ylim(0, 620)
    axs[1].legend(loc="upper left", facecolor="#1E293B", edgecolor="#475569", labelcolor="#F8FAFC", fontsize=8.5)
    axs[1].grid(True, color="#1E293B")

    chart_path = os.path.join(os.path.dirname(__file__), "figure05_neural_latency_breakdown.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved high-resolution Neural Latency chart to: {chart_path}")

    neural_metrics = {
        "vla_model_parameters": "7.2B",
        "precision": "FP8 (TensorRT-LLM Quantized)",
        "dual_thor_tflops": 2000.0,
        "vla_context_tokens": n_ctx,
        "vla_mean_inference_ms": round(vla_mean_ms, 2),
        "vla_p99_inference_ms": round(vla_p99_ms, 2),
        "vla_frequency_hz": 50,
        "wbc_monte_carlo_ticks": n_rtos_ticks,
        "wbc_mean_microseconds": round(wbc_mean_us, 1),
        "wbc_p99_microseconds": round(wbc_p99_us, 1),
        "wbc_p999_microseconds": round(wbc_p999_us, 1),
        "wbc_max_microseconds": round(wbc_max_us, 1),
        "wbc_deadline_misses": wbc_misses,
        "wbc_safety_headroom_percent": round((1000.0 - wbc_p99_us) / 10.0, 1),
        "edge_slip_reflex_ms": fpga_reflex_ms,
    }
    with open(os.path.join(os.path.dirname(__file__), "figure05_neural_metrics.json"), "w") as f:
        json.dump(neural_metrics, f, indent=4)
    print("[+] Successfully exported neural benchmark metrics.")


if __name__ == "__main__":
    run_neural_benchmark()
