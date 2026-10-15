#!/usr/bin/env python3
"""Runs the full config sweep behind bench_results.csv in one command -- the
committed record of exactly which configs, concurrencies and reps produced it
(CLAUDE.md rule 2: no performance claim without a reproducible script).

    python bench/run_sweep.py                    # every config below, appended to bench/bench_results.csv
    python bench/run_sweep.py --csv other.csv    # write elsewhere instead
    python bench/run_sweep.py --concurrencies 1 2 --reps 1   # a cheap smoke test

Each entry in CONFIGS is turned into one
`python bench/bench_serving.py --start-server <flags> --concurrencies ... --reps ... --csv ...`
call (see bench_serving.py's own docstring for what each flag does); the
command is printed before it runs, so the exact invocation is always visible,
not just implied. Add a config by adding a row to CONFIGS.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

# Every config currently represented in bench_results.csv. Extra CLI flags
# passed straight through to bench_serving.py; label is auto-derived there.
CONFIGS: list[list[str]] = [
    [],  # baseline
    ["--continuous-batching", "--max-num-seqs", "8"],
    ["--kv-cache-quantization", "--kv-cache-quantization-bits", "4"],
    ["--kv-cache-quantization", "--kv-cache-quantization-bits", "8"],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=BENCH_DIR / "bench_results.csv")
    parser.add_argument("--concurrencies", nargs="+", default=["1", "2", "4", "8", "16"])
    parser.add_argument("--reps", default="1")
    args = parser.parse_args()

    for i, extra_flags in enumerate(CONFIGS):
        cmd = [
            sys.executable,
            str(BENCH_DIR / "bench_serving.py"),
            "--start-server",
            "--concurrencies",
            *args.concurrencies,
            "--reps",
            args.reps,
            "--csv",
            str(args.csv.resolve()),
            *extra_flags,
        ]
        print(f"\n=== config {i + 1}/{len(CONFIGS)} ===\n+ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        if i + 1 < len(CONFIGS):
            time.sleep(2)  # let the port fully release before the next --start-server


if __name__ == "__main__":
    main()
