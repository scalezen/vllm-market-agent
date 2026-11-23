import time

import torch

BYTES_TO_DTYPE: dict[int, torch.dtype] = {
    2: torch.float16,
    4: torch.float32,
    8: torch.float64,
}

PEAK_FLOPS = 2.6e12   # FLOP/s, M1 GPU
PEAK_BW = 68e9        # bytes/s

# Choose "mps" for Apple Silicon GPU, or "cpu" for CPU + AMX/AVX
float_bytes = 4 # 16-bit float
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = BYTES_TO_DTYPE[float_bytes]

def sync():
    if device.type == "mps":
        torch.mps.synchronize()

def measure_peak_bandwidth()->float:

    vec_len = 100_000
    vA = torch.randn(vec_len, device=device, dtype=dtype)
    vB = torch.randn(vec_len, device=device, dtype=dtype)
    alpha = 2.5

    # Warmup
    for _ in range(5):
        vC = torch.add(vA, vB, alpha=alpha)

    # Timed
    times = []
    for _ in range(25):
        start_time = time.perf_counter()
        vC = torch.add(vA, vB, alpha=alpha)
        sync()
        times.append(time.perf_counter() - start_time)

    # Memory accessed: float_bytes*3*vec_len
    total_bytes = 3 * float_bytes * vec_len
    peak_bandwidth_gbps =  total_bytes / min(times) / 1.0e9 # use min time

    # Clear large memory allocation
    del vA, vB, vC
    if device.type == "mps":
        torch.mps.empty_cache()

    return peak_bandwidth_gbps

def measure_gflops()->float:

    sizes = [128, 256, 512, 1024, 2058, 3072, 4096, 6144, 8192]
    result_runtime: dict[int, float] = {}

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
            elapsed = (time.perf_counter() - start) / runs

        result_runtime[N] = min(runtimes)

