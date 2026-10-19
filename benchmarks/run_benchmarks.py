#!/usr/bin/env python3
"""Run benchmark_impl.py once per config, in order.

    python benchmarks/run_benchmarks.py benchmarks/configs/*.yaml

Each config runs in a fresh process, so nothing carries over between runs: no
memory held by the last run, and no client tied to a closed event loop or a
stopped server (the module-level aclient can't be reused across asyncio.run).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", type=Path, nargs="+")
    configs = parser.parse_args().configs

    for i, config in enumerate(configs):
        cmd = [sys.executable, str(BENCHMARKS_DIR / "benchmark_impl.py"), str(config)]
        print(f"\n=== {i + 1}/{len(configs)}: {config.name} ===\n+ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        if i + 1 < len(configs):
            time.sleep(2)  # let the port fully release before the next server starts


if __name__ == "__main__":
    main()
