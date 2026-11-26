import time

import torch

BYTES_TO_DTYPE: dict[int, torch.dtype] = {
    2: torch.float16,
    4: torch.float32,
    # 8: torch.float64, # MPS supports float32 as the largest size
}

PEAK_FLOPS = 2.6e12  # FLOP/s, M1 GPU
PEAK_BW = 68e9  # bytes/s

# Choose "mps" for Apple Silicon GPU, or "cpu" for CPU + AMX/AVX
float_bytes = 4  # 32-bit float
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = BYTES_TO_DTYPE[float_bytes]


def sync():
    if device.type == "mps":
        torch.mps.synchronize()


def measure_peak_bandwidth() -> float:
    vec_len = 32_000_000  # 100_000
    vA = torch.randn(vec_len, device=device, dtype=dtype)
    vB = torch.randn(vec_len, device=device, dtype=dtype)
    vC = torch.empty_like(vA)
    alpha = 2.5

    # Warmup
    for _ in range(5):
        torch.add(vA, vB, alpha=alpha, out=vC)
    sync()

    # Timed
    times = []
    for _ in range(25):
        start_time = time.perf_counter()
        torch.add(vA, vB, alpha=alpha, out=vC)
        sync()
        times.append(time.perf_counter() - start_time)

    # Memory accessed: float_bytes*3*vec_len
    total_bytes = 3 * float_bytes * vec_len
    peak_bandwidth_gbps = total_bytes / min(times) / 1.0e9  # use min time

    # Clear large memory allocation
    del vA, vB, vC
    if device.type == "mps":
        torch.mps.empty_cache()

    return peak_bandwidth_gbps


def measure_gflops() -> tuple[dict[int, float], dict[int, float]]:

    sizes = [128, 256, 512, 1024, 2048, 3072, 4096, 6144, 8192]
    result_runtime: dict[int, float] = {}
    measured_ai: dict[int, float] = {}
    measured_gflops: dict[int, float] = {}

    for N in sizes:
        A = torch.randn(N, N, device=device, dtype=dtype)
        B = torch.randn(N, N, device=device, dtype=dtype)

        # Warmup
        for _ in range(5):
            C = A @ B
        sync()

        # Benchmark
        runs = 20
        runtimes = []
        for _ in range(runs):
            sync()
            start = time.perf_counter()
            C = A @ B
            sync()
            runtimes.append(time.perf_counter() - start)

        result_runtime[N] = min(runtimes)

        # Roofline: FLOPs = 2*N^3, compulsory bytes read = 3 * b * N^2
        flops_computed = 2.0 * (N**3)
        bytes_accessed = 3.0 * float_bytes * (N**2)

        arithmetic_intensity = flops_computed / bytes_accessed
        gflops = (flops_computed / result_runtime[N]) / 1e9

        measured_ai[N] = arithmetic_intensity
        measured_gflops[N] = gflops

        del A, B, C
        if device.type == "mps":
            torch.mps.empty_cache()

    return (measured_ai, measured_gflops)


if __name__ == "__main__":

    peak_bandwidth_gbps = measure_peak_bandwidth()
    measured_ai, measured_gflops = measure_gflops()

    peak_compute_gflops = max(measured_gflops.values())
    ridge_point_ai = peak_compute_gflops / peak_bandwidth_gbps

    print(f"\nMeasured compute ceiling (Max):    {peak_compute_gflops:.2f} GFLOPs/s")
    print(f"Roofline ridge point:                 {ridge_point_ai:.2f} FLOPs/Byte")
