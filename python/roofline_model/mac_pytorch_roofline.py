import time
import torch

# Choose "mps" for Apple Silicon GPU, or "cpu" for CPU + AMX/AVX
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32

N = 4096  # Increase to 8192 if benchmarking an Apple Silicon GPU
A = torch.randn(N, N, device=device, dtype=dtype)
B = torch.randn(N, N, device=device, dtype=dtype)

def sync():
    if device.type == "mps":
        torch.mps.synchronize()

# Warmup 
for _ in range(5):
    C = A @ B
sync()

# Benchmark
runs = 20
sync()
start = time.perf_counter()
for _ in range(runs):
    C = A @ B
sync()
elapsed = (time.perf_counter() - start) / runs

gflops = (2 * (N ** 3)) / (elapsed * 1e9)
print(f"Device: {device} | Avg time: {elapsed*1000:.2f} ms | Peak: {gflops:.1f} GFLOP/s")
