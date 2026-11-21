import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
dtype = torch.float32
bytes_per_elem = 4  # FP32 = 4 bytes


def sync():
    """Ensure asynchronous GPU operations are complete."""
    if device.type == "mps":
        torch.mps.synchronize()


print(f"Benchmarking empirical Roofline on device: {device} ({dtype})\n")

# Measure memory bandwidth slope (STREAM triad: C = A + alpha * B)
# 100M elements (~400 MB per vector, ~1.2 GB total traffic), larger than
# CPU/GPU cache sizes
vec_size = 100_000_000
vA = torch.randn(vec_size, device=device, dtype=dtype)
vB = torch.randn(vec_size, device=device, dtype=dtype)
alpha = 2.5

# Warmup
for _ in range(5):
    vC = torch.add(vA, vB, alpha=alpha)
sync()

# Timed runs, take minimum time to eliminate OS jitter
bw_times = []
for _ in range(25):
    sync()
    t0 = time.perf_counter()
    vC = torch.add(vA, vB, alpha=alpha)
    sync()
    bw_times.append(time.perf_counter() - t0)

min_bw_time = min(bw_times)
# Traffic: read vA (4B) + read vB (4B) + write vC (4B) = 12 bytes per element
total_bytes_bw = 3 * bytes_per_elem * vec_size
peak_bandwidth_gbs = (total_bytes_bw / min_bw_time) / 1e9

# Free large vectors before running matrix sweeps
del vA, vB, vC
if device.type == "mps":
    torch.mps.empty_cache()

print(f"Empirical Memory Bandwidth (Slope): {peak_bandwidth_gbs:.2f} GB/s")

# Sweep matrix sizes (N x N) to measure compute ceiling & ridge point on roofline
sizes = [128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
if device.type == "cpu":
    # Cap CPU sweep at 4096 so it completes quickly
    sizes = [s for s in sizes if s <= 4096]

measured_ai = []
measured_gflops = []

for N in sizes:
    A = torch.randn(N, N, device=device, dtype=dtype)
    B = torch.randn(N, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(5):
        C = A @ B
    sync()

    # Adjust iterations so small matrices still get accurate timer resolution
    iters = 50 if N <= 1024 else 20
    gemm_times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        C = A @ B
        sync()
        gemm_times.append(time.perf_counter() - t0)

    min_gemm_time = min(gemm_times)

    # Roofline: FLOPs = 2*N^3, compulsory bytes read = 3 * b * N^2
    flops = 2.0 * (N**3)
    compulsory_bytes = 3.0 * bytes_per_elem * (N**2)

    ai = flops / compulsory_bytes
    gflops = (flops / min_gemm_time) / 1e9

    measured_ai.append(ai)
    measured_gflops.append(gflops)
    print(f"N = {N:4d} | Measured arithmetic intensity = {ai:6.1f} FLOPs/B | Measured gflops = {gflops:8.1f} GFLOPs/s")

    del A, B, C
    if device.type == "mps":
        torch.mps.empty_cache()

peak_compute_gflops = max(measured_gflops)
ridge_point_ai = peak_compute_gflops / peak_bandwidth_gbs

print(f"\nEmpirical compute ceiling (Max):    {peak_compute_gflops:.2f} GFLOPs/s")
print(f"Roofline ridge point):                 {ridge_point_ai:.2f} FLOPs/Byte")

# Plot the Empirical Roofline Model (Log-Log)
ai_axis = np.logspace(-1, 3.2, 500)  # 0.1 to ~1585 FLOPs/Byte
bandwidth_roof = peak_bandwidth_gbs * ai_axis
compute_roof = np.full_like(ai_axis, peak_compute_gflops)
roofline = np.minimum(bandwidth_roof, compute_roof)

plt.figure(figsize=(10, 6), dpi=150)

# Plot the empirical Roofline boundary
plt.loglog(ai_axis, roofline, color="#1f77b4", linewidth=2.5, label="Empirical Roofline boundary")

# Plot dashed extensions to highlight the intersection (Ridge Point)
plt.loglog(ai_axis, bandwidth_roof, color="#1f77b4", linestyle="--", alpha=0.35)
plt.loglog(ai_axis, compute_roof, color="#1f77b4", linestyle="--", alpha=0.35)

# Plot the ridge point
plt.scatter(
    [ridge_point_ai],
    [peak_compute_gflops],
    color="#d62728",
    zorder=5,
    s=60,
    label=f"Ridge Point ({ridge_point_ai:.1f} FLOPs/B)",
)

# Plot measured GEMM sweep points
plt.scatter(
    measured_ai, measured_gflops, color="#ff7f0e", edgecolor="black", zorder=6, s=70, label="PyTorch GEMM (N x N)"
)

for i, N in enumerate(sizes):
    plt.annotate(
        f"N={N}",
        (measured_ai[i], measured_gflops[i]),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
        fontsize=8,
    )

# Labels and annotations
plt.title(f"Empirical Roofline Model — iMac ({device.type.upper()}, FP32)", fontsize=13, fontweight="bold")
plt.xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=11)
plt.ylabel("Performance (GFLOPs / s)", fontsize=11)
plt.ylim(bottom=1, top=peak_compute_gflops * 2.2)
plt.xlim(left=0.1, right=1500)
plt.grid(True, which="both", linestyle=":", alpha=0.6)

# Text callouts for Slope and Ceiling
plt.text(
    0.25,
    peak_bandwidth_gbs * 0.35,
    f"Bandwidth Slope\n{peak_bandwidth_gbs:.1f} GB/s",
    rotation=33,
    fontsize=9.5,
    color="#1f77b4",
    fontweight="bold",
)
plt.text(
    ridge_point_ai * 1.4,
    peak_compute_gflops * 1.12,
    f"Compute Ceiling: {peak_compute_gflops:.1f} GFLOPs/s",
    fontsize=9.5,
    color="#1f77b4",
    fontweight="bold",
)

plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("imac_roofline.png")
plt.show()
