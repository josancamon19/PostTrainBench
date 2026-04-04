#!/usr/bin/env python3
"""Run baseline evaluations on RunPod GPUs.

Cost-efficient: spins up N reusable GPU pods, distributes evals via work-stealing queue.
Each pod installs packages once, then runs multiple evaluations sequentially.

Usage:
    python scripts/run_baselines.py                          # Run all missing baselines
    python scripts/run_baselines.py --model llama3.2-1b --benchmark gsm8k
    python scripts/run_baselines.py --type instruct --gpu a100
    python scripts/run_baselines.py --dry-run --workers 1
"""

import argparse
import atexit
import json
import os
import queue
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import runpod

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from constants import BASE_SCORES, BENCHMARKS, INSTRUCT_BASELINES, MODELS  # noqa: E402

GPU_TYPE_MAP = {
    "h100": "NVIDIA H100 80GB HBM3",
    "a100": "NVIDIA A100 80GB PCIe",
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "l40s": "NVIDIA L40S",
    "4090": "NVIDIA GeForce RTX 4090",
}

RUNPOD_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"

_SSH_PREAMBLE = (
    'export PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:'
    '/usr/sbin:/usr/bin:/sbin:/bin:$PATH"; '
    "[ -f /etc/rp_environment ] && . /etc/rp_environment 2>/dev/null; "
)

# Track active pods for cleanup on exit
_active_pods: set[str] = set()


def _cleanup_pods():
    for pod_id in list(_active_pods):
        try:
            runpod.terminate_pod(pod_id)
            print(f"  Cleaned up pod {pod_id}")
        except Exception as e:
            print(f"  WARNING: Failed to clean up pod {pod_id}: {e}")


atexit.register(_cleanup_pods)


def instruct_model_id(base_model_id: str) -> str:
    """Derive instruct model ID from base model ID."""
    if base_model_id.endswith("-Base"):
        return base_model_id.removesuffix("-Base")
    return f"{base_model_id}-Instruct"


def has_score(model_id: str, benchmark_id: str, model_type: str) -> bool:
    base_id = model_id.replace("-Instruct", "")
    if model_type == "base":
        return (base_id, benchmark_id) in BASE_SCORES
    return (base_id, benchmark_id) in INSTRUCT_BASELINES


class RunPodWorker:
    """Manages a RunPod pod for running multiple evaluations via SSH."""

    def __init__(self, worker_id: int, gpu_type: str, disk_gb: int = 80):
        self.worker_id = worker_id
        self.pod_id: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self.gpu_type = gpu_type
        self.disk_gb = disk_gb
        self._setup_done = False

        ed25519 = Path.home() / ".ssh" / "id_ed25519"
        rsa = Path.home() / ".ssh" / "id_rsa"
        self._key = str(ed25519 if ed25519.exists() else rsa)

    def _log(self, msg: str):
        print(f"  [worker-{self.worker_id}] {msg}", flush=True)

    def start(self):
        env = {}
        for var in ("HF_TOKEN", "OPENAI_API_KEY"):
            if os.environ.get(var):
                env[var] = os.environ[var]

        pod = runpod.create_pod(
            name=f"ptb-baseline-{self.worker_id}-{int(time.time()) % 10000}",
            image_name=RUNPOD_IMAGE,
            gpu_type_id=self.gpu_type,
            gpu_count=1,
            cloud_type="SECURE",
            support_public_ip=True,
            start_ssh=True,
            container_disk_in_gb=self.disk_gb,
            ports="22/tcp",
            env=env or None,
        )
        self.pod_id = pod["id"]
        _active_pods.add(self.pod_id)
        self._log(f"Created pod {self.pod_id}")

        self._wait_ready()
        self._wait_ssh()
        self._log(f"Pod ready: {self.ssh_host}:{self.ssh_port}")

    def _wait_ready(self, timeout: int = 600):
        start = time.time()
        while time.time() - start < timeout:
            pod = runpod.get_pod(self.pod_id)
            for port in (pod.get("runtime") or {}).get("ports") or []:
                if port.get("privatePort") == 22 and port.get("isIpPublic"):
                    self.ssh_host = port["ip"]
                    self.ssh_port = port["publicPort"]
                    return
            time.sleep(5)
        raise TimeoutError(f"Pod {self.pod_id} not ready after {timeout}s")

    def _wait_ssh(self, timeout: int = 300):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = self.ssh("echo ready", timeout=10)
                if r.returncode == 0:
                    return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError(f"SSH not ready on {self.pod_id}")

    def _ssh_args(self) -> list[str]:
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=5",
            "-i", self._key,
            "-p", str(self.ssh_port),
        ]

    def _scp_args(self) -> list[str]:
        return [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-i", self._key,
            "-P", str(self.ssh_port),
        ]

    def ssh(self, cmd: str, timeout: int | None = None) -> subprocess.CompletedProcess:
        full = _SSH_PREAMBLE + cmd
        args = ["ssh", *self._ssh_args(), f"root@{self.ssh_host}", f"bash -l -c {shlex.quote(full)}"]
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)

    def ssh_stream(self, cmd: str, timeout: int | None = None) -> int:
        """Run command with streaming stdout. Returns exit code."""
        full = _SSH_PREAMBLE + cmd
        args = ["ssh", *self._ssh_args(), f"root@{self.ssh_host}", f"bash -l -c {shlex.quote(full)}"]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        start = time.time()
        for line in proc.stdout:
            print(f"    {line}", end="", flush=True)
            if timeout and time.time() - start > timeout:
                proc.kill()
                proc.wait()
                raise TimeoutError(f"Command timed out after {timeout}s")
        proc.wait()
        return proc.returncode

    def scp_up(self, local: str | Path, remote: str):
        self.ssh(f"mkdir -p {shlex.quote(str(Path(remote).parent))}")
        args = ["scp", *self._scp_args(), str(local), f"root@{self.ssh_host}:{remote}"]
        subprocess.run(args, capture_output=True, check=True)

    def scp_up_dir(self, local_dir: str | Path, remote_dir: str):
        self.ssh(f"mkdir -p {shlex.quote(remote_dir)}")
        items = [str(p) for p in Path(local_dir).iterdir()]
        if not items:
            return
        args = ["scp", "-r", *self._scp_args(), *items, f"root@{self.ssh_host}:{remote_dir}"]
        subprocess.run(args, capture_output=True, check=True)

    def setup(self):
        """Install ML packages (one-time per pod)."""
        if self._setup_done:
            return
        self._log("Installing packages (one-time)...")
        cmds = [
            "rm -f /usr/lib/python*/EXTERNALLY-MANAGED",
            "pip install --no-cache vllm",
            (
                "pip install --no-cache accelerate bitsandbytes datasets evaluate "
                "inspect-ai inspect-evals lm-eval matplotlib ninja 'openai>=2.26.0' "
                "packaging pandas peft scikit-learn transformers trl"
            ),
            "mkdir -p /app",
        ]
        for cmd in cmds:
            self._log(f"  {cmd[:80]}...")
            r = self.ssh(cmd, timeout=900)
            if r.returncode != 0:
                raise RuntimeError(f"Setup failed: {r.stderr[:500]}")
        self._setup_done = True
        self._log("Setup complete.")

    def run_eval(self, model_key: str, model_type: str, benchmark_id: str) -> dict | None:
        base_id = MODELS[model_key].model_id
        model_id = instruct_model_id(base_id) if model_type == "instruct" else base_id
        label = f"{benchmark_id}/{model_id}"

        self._log(f"RUNNING {label}...")

        task_dir = SRC_DIR / "tasks" / benchmark_id
        eval_script = task_dir / "evaluate.py"
        if not eval_script.exists():
            self._log(f"SKIP {label} (no evaluate.py)")
            return None

        # Clean workspace and upload eval files
        self.ssh("rm -rf /app/* 2>/dev/null; mkdir -p /app")
        self.scp_up(eval_script, "/app/evaluate.py")

        eval_code = task_dir / "evaluation_code"
        if eval_code.is_dir():
            self.scp_up_dir(eval_code, "/app/evaluation_code")

        templates = SRC_DIR / "harbor_template" / "environment" / "templates"
        if templates.is_dir():
            self.scp_up_dir(templates, "/app/templates")

        # Build eval command
        limit = "-1"
        conns = {
            "llama3.2-1b": 128, "llama3.2-3b": 64, "llama3.1-8b": 32,
            "qwen3-8b": 32, "qwen3-30b-a3b": 16,
        }
        max_conn = conns.get(model_key, 32)
        eval_timeout = 7200

        if benchmark_id == "bfcl":
            limit = "400"

        if benchmark_id in ("arenahardwriting", "healthbench"):
            extra_args = ""
            eval_timeout = 14400
        else:
            extra_args = f"--gpu-memory-utilization 0.9 --max-connections {max_conn} --max-tokens 2048"

        cmd = (
            f"cd /app && python3 -X int_max_str_digits=0 evaluate.py "
            f"--model-path {model_id} --limit {limit} {extra_args} "
            f"--json-output-file /tmp/metrics.json"
        )

        try:
            exit_code = self.ssh_stream(cmd, timeout=eval_timeout)
            if exit_code != 0:
                self._log(f"ERROR {label}: eval exited with code {exit_code}")

            r = self.ssh("cat /tmp/metrics.json", timeout=10)
            if r.returncode != 0:
                self._log(f"ERROR {label}: metrics.json not found")
                return None

            metrics = json.loads(r.stdout)
            score = metrics.get("accuracy", 0)
            self._log(f"DONE {label}: {score:.4f} ({score * 100:.1f}%)")
            return {
                "model_key": model_key,
                "model_type": model_type,
                "model_id": model_id,
                "benchmark_id": benchmark_id,
                "score": score,
                "metrics": metrics,
            }
        except Exception as e:
            self._log(f"ERROR {label}: {e}")
            return None

    def terminate(self):
        if self.pod_id:
            try:
                runpod.terminate_pod(self.pod_id)
                _active_pods.discard(self.pod_id)
                self._log(f"Terminated pod {self.pod_id}")
            except Exception as e:
                self._log(f"WARNING: Failed to terminate {self.pod_id}: {e}")
            self.pod_id = None


def worker_fn(worker: RunPodWorker, job_queue: queue.Queue, results: list):
    """Worker thread: start pod, install packages, process jobs from queue."""
    try:
        worker.start()
        worker.setup()
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                break
            result = worker.run_eval(*job)
            if result:
                results.append(result)
            job_queue.task_done()
    except Exception as e:
        worker._log(f"FATAL: {e}")
        raise
    finally:
        worker.terminate()


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations on RunPod GPUs")
    parser.add_argument("--model", type=str, default=None, help="Model key (e.g. llama3.2-1b, qwen3-8b)")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark ID (e.g. gsm8k)")
    parser.add_argument("--type", type=str, default=None, choices=["base", "instruct"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if score already exists")
    parser.add_argument("--workers", type=int, default=3, help="Number of RunPod pods to spin up")
    parser.add_argument("--gpu", type=str, default="h100", help=f"GPU type: {', '.join(GPU_TYPE_MAP)}")
    parser.add_argument("--disk-gb", type=int, default=80, help="Container disk size in GB")
    parser.add_argument("--output", type=str, default="scripts/baseline_results.json")
    args = parser.parse_args()

    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    gpu_type = GPU_TYPE_MAP.get(args.gpu, args.gpu)

    model_keys = [args.model] if args.model else list(MODELS.keys())
    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS.keys())
    types = [args.type] if args.type else ["base", "instruct"]

    # Build and filter job list
    actual_jobs = []
    for model_key in model_keys:
        if model_key not in MODELS:
            print(f"  SKIP unknown model: {model_key}")
            continue
        base_id = MODELS[model_key].model_id
        for model_type in types:
            model_id = instruct_model_id(base_id) if model_type == "instruct" else base_id
            for benchmark_id in benchmarks:
                if not args.force and has_score(base_id, benchmark_id, model_type):
                    print(f"  SKIP {benchmark_id}/{model_id} (already have score)")
                elif not (SRC_DIR / "tasks" / benchmark_id / "evaluate.py").exists():
                    print(f"  SKIP {benchmark_id}/{model_id} (no evaluate.py)")
                else:
                    actual_jobs.append((model_key, model_type, benchmark_id))

    n_workers = min(args.workers, len(actual_jobs)) if actual_jobs else 0

    print(f"\nRunning {len(actual_jobs)} evaluations across {n_workers} RunPod pods "
          f"(gpu={args.gpu}, disk={args.disk_gb}GB, dry_run={args.dry_run})\n")

    if args.dry_run:
        for model_key, model_type, benchmark_id in actual_jobs:
            base_id = MODELS[model_key].model_id
            model_id = instruct_model_id(base_id) if model_type == "instruct" else base_id
            print(f"  WOULD RUN {benchmark_id}/{model_id}")
        return

    if not actual_jobs:
        print("Nothing to run.")
        return

    # Distribute jobs via work-stealing queue
    job_queue: queue.Queue = queue.Queue()
    for job in actual_jobs:
        job_queue.put(job)

    results: list[dict] = []
    workers = [RunPodWorker(i, gpu_type, args.disk_gb) for i in range(n_workers)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(worker_fn, w, job_queue, results) for w in workers]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  Worker failed: {e}")

    # Save results
    if results:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)

        existing.extend(results)
        with open(output_path, "w") as f:
            json.dump(existing, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"\n{'Model':<40} {'Benchmark':<20} {'Score':>10}")
        print("-" * 72)
        for r in sorted(results, key=lambda x: (x["model_id"], x["benchmark_id"])):
            print(f"{r['model_id']:<40} {r['benchmark_id']:<20} {r['score'] * 100:>9.1f}%")

        print("\nTo update constants.py, add these entries:")
        for r in sorted(results, key=lambda x: (x["model_id"], x["benchmark_id"])):
            base_id = r["model_id"].replace("-Instruct", "")
            # For Qwen3 models, the instruct version removes -Base suffix
            if not r["model_id"].endswith("-Base") and r["model_type"] == "instruct":
                base_id = f"{r['model_id']}-Base" if "Qwen" in r["model_id"] else base_id
            dict_name = "INSTRUCT_BASELINES" if r["model_type"] == "instruct" else "BASE_SCORES"
            print(f'    # {dict_name}\n    ("{base_id}", "{r["benchmark_id"]}"): {r["score"]:.3f},')


if __name__ == "__main__":
    main()
