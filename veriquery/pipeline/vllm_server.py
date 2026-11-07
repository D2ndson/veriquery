"""Start / stop / check the local vLLM server that serves the base model plus every trained LoRA adapter.

vLLM serves the base weights as "base" and each adapter under its own name (writer-sft, writer-dpo, repairer, ...), so
one process answers for all of them (multi-LoRA). It must be restarted to pick up a newly trained adapter, which
`ensure()` does automatically. GPU training and vLLM cannot share the GPU: the runner stops vLLM before training.

NOTE: written against vLLM's documented OpenAI server flags; it cannot be tested without a GPU, so run
`python -m veriquery.pipeline.vllm_server status` first and read logs/vllm.log if something looks off.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from veriquery.config import load_models_config, path

PID_FILE = Path("/tmp/veriquery_vllm.pid")     # global on purpose: quick and full runs must see the same server
SIG_FILE = Path("/tmp/veriquery_vllm.sig")


def pipeline_cfg() -> dict:
    import yaml
    return yaml.safe_load(path("configs/pipeline.yaml").read_text())["vllm"]


def served_models(port: int = 8000, timeout: float = 3.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=timeout) as r:
            return [m["id"] for m in json.load(r)["data"]]
    except Exception:
        return []


def is_up(port: int = 8000) -> bool:
    return bool(served_models(port))


def adapters_on_disk() -> dict[str, Path]:
    root = path("artifacts/adapters")
    if not root.is_dir():
        return {}
    return {p.name: p for p in sorted(root.glob("*")) if (p / "adapter_config.json").is_file()}


def signature(adapters: dict[str, Path]) -> str:
    """Identity of the adapter set (names, locations, weight-file mtimes): a changed adapter forces a restart."""
    sig = {}
    for n, p in adapters.items():
        w = Path(p) / "adapter_model.safetensors"
        sig[n] = [str(Path(p).resolve()), w.stat().st_mtime_ns if w.is_file() else 0]
    return json.dumps(sig, sort_keys=True)


def stop() -> None:
    pf = PID_FILE
    if pf.is_file():
        try:
            pid = int(pf.read_text())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            for _ in range(60):
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            else:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pf.unlink(missing_ok=True)
    SIG_FILE.unlink(missing_ok=True)
    time.sleep(2)


def start(adapters: dict[str, Path] | None = None, wait: float = 900.0) -> None:
    cfg = pipeline_cfg()
    try:  # training in this same process keeps GPU memory cached; give it back before vLLM starts
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    base = load_models_config()["training"]["base_model"]
    adapters = adapters if adapters is not None else adapters_on_disk()
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", base, "--served-model-name", "base",
           "--port", str(cfg["port"]), "--max-model-len", str(cfg["max_model_len"]),
           "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]), "--dtype", "bfloat16"]
    if adapters:
        cmd += ["--enable-lora", "--max-lora-rank", str(cfg["max_lora_rank"]), "--max-loras", str(max(2, len(adapters))),
                "--lora-modules", *[f"{n}={p}" for n, p in adapters.items()]]
    log = path("logs/vllm.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vllm] starting: serving base + {sorted(adapters)}  (log: {log})")
    with log.open("a") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "HF_TOKEN": os.environ.get("HF_TOKEN", "")})
    PID_FILE.write_text(str(proc.pid))
    SIG_FILE.write_text(signature(adapters))
    t0 = time.time()
    while time.time() - t0 < wait:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}. Last log lines:\n" + "\n".join(
                log.read_text().splitlines()[-25:]))
        if is_up(cfg["port"]):
            print(f"[vllm] up after {time.time() - t0:.0f}s: {served_models(cfg['port'])}")
            return
        time.sleep(5)
    raise TimeoutError(f"vLLM did not come up within {wait:.0f}s; see {log}")


def ensure(extra_adapters: dict[str, Path] | None = None) -> None:
    """Make sure the server is up and serves every adapter currently on disk (restart if one is missing)."""
    port = pipeline_cfg()["port"]
    want = {**adapters_on_disk(), **(extra_adapters or {})}
    have = set(served_models(port))
    same = SIG_FILE.is_file() and SIG_FILE.read_text() == signature(want)
    if have and set(want) <= have and same:
        return
    if have:
        print("[vllm] adapter set changed (or foreign server) -> restarting server")
        stop()
    start(want)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print("up:" if is_up(pipeline_cfg()["port"]) else "down", served_models(pipeline_cfg()["port"]),
              "adapters on disk:", list(adapters_on_disk()))
    elif cmd == "start":
        ensure()
    elif cmd == "stop":
        stop()
