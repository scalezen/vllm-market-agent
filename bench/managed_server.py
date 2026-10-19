"""Starts a vLLM-MLX server for one benchmark run and stops it afterwards."""
import os
import subprocess
import sys
import time
from pathlib import Path

from market_agent.config import BASE_URL, client


def _server_reachable() -> bool:
    try:
        client.models.list()
        return True
    except Exception:
        return False


class ManagedServer:
    """Starts `script` with the config's server settings and waits until it is ready;
    stops it on exit. Refuses to reuse a server it didn't start, so the recorded
    config is always the one that served the requests."""

    def __init__(self, server: dict, script: Path, log_path: Path):
        # vllm-mlx lives next to whatever Python runs this script; put that directory
        # first so the serve script finds it even without an activated environment.
        env_bin = str(Path(sys.executable).parent)
        bits = server["kv_cache_quantization_bits"]
        self.env = {
            **os.environ,
            "PATH": f"{env_bin}:{os.environ.get('PATH', '')}",
            "MODEL": server["model"],
            "PARSER": server["tool_call_parser"],
            "REASONING_PARSER": server["reasoning_parser"],
            "CONTINUOUS_BATCHING": "true" if server["continuous_batching"] else "false",
            "MAX_NUM_SEQS": str(server["max_num_seqs"] or ""),
            "KV_CACHE_QUANTIZATION": "true" if server["kv_cache_quantization"] else "false",
            "KV_CACHE_QUANTIZATION_BITS": str(bits or ""),
            "PREFIX_CACHE": "true" if server["prefix_cache"] else "false",
        }
        self.script = script
        self.timeout_s = server["ready_timeout_s"]
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "ManagedServer":
        if _server_reachable():
            raise RuntimeError(f"Something is already serving on {BASE_URL}. Stop it first.")
        log = self.log_path.open("w")
        self.proc = subprocess.Popen(
            [str(self.script)], cwd=self.script.parent, env=self.env, stdout=log, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"Server exited before it was ready; see {self.log_path}:\n{self._tail()}")
            if _server_reachable():
                return self
            time.sleep(2)
        self._stop()
        raise TimeoutError(f"Server not ready within {self.timeout_s}s; see {self.log_path}:\n{self._tail()}")

    def _tail(self, n: int = 20) -> str:
        try:
            return "\n".join(self.log_path.read_text().splitlines()[-n:])
        except OSError:
            return "(no log)"

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)

    def __exit__(self, *exc_info) -> None:
        self._stop()
