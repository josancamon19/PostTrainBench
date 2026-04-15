"""Generate Harbor-compatible tasks for running PostTrainBench evaluations."""

import contextlib
import json
import shutil
from pathlib import Path

import chz

from constants import (
    BASE_SCORES,
    BENCHMARKS,
    INSTRUCT_BASELINES,
    MODELS,
    REGRESSION_EVALS,
    BenchmarkInfo,
    ModelInfo,
)

TEMPLATE_DIR = Path(__file__).parent / "harbor_template"
SRC_DIR = Path(__file__).parent


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(self, output_dir: Path, num_hours: float = 10, mode: str = "gpu", include_target: bool = False):
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.mode = mode  # "gpu", "tinker", or "gpu-runpod"
        self.include_target = include_target

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        bench_file = SRC_DIR / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def _template(self, *parts: str) -> Path:
        """Resolve a template path, using .tinker variant when in tinker mode.

        E.g. ("task.toml") -> "task.tinker.toml", ("instruction.md") -> "instruction.tinker.md".
        gpu-runpod shares GPU templates (no variant suffix).
        """
        if self._template_mode == "tinker":
            filename = parts[-1]
            name, dot, ext = filename.rpartition(".")
            tinker_filename = f"{name}.tinker.{ext}" if dot else f"{filename}.tinker"
            prefixed = [*list(parts[:-1]), tinker_filename]
            candidate = TEMPLATE_DIR / Path(*prefixed)
            if candidate.exists():
                return candidate
        return TEMPLATE_DIR / Path(*parts)

    @property
    def _models(self) -> dict[str, ModelInfo]:
        if self.mode == "tinker":
            return {k: v for k, v in MODELS.items() if v.tinker}
        return MODELS

    @property
    def _template_mode(self) -> str:
        """The template mode to use — gpu-runpod shares GPU templates."""
        return "gpu" if self.mode == "gpu-runpod" else self.mode

    def generate_task_toml(self, task_dir: Path, benchmark_id: str = "", task_id: str = "") -> None:
        content = self._template("task.toml").read_text()
        if benchmark_id in ("arenahardwriting", "healthbench"):
            if "[environment.env]" in content:
                content = content.replace(
                    "[environment.env]",
                    '[environment.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"',
                    1,
                )
            else:
                content += '\n[environment.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
        # Add prebuilt docker_image for RunPod tasks (GPU mode uses Dockerfile directly)
        if self.mode == "gpu-runpod" and task_id:
            image = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
            content = content.replace(
                "[environment]",
                f'[environment]\ndocker_image = "{image}"',
                1,
            )
        (task_dir / "task.toml").write_text(content)

    def generate_instruction(
        self, task_dir: Path, model_info: ModelInfo, benchmark_info: BenchmarkInfo, benchmark_id: str = ""
    ) -> None:
        content = self._template("instruction.md").read_text()
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        timeout_sec = self._read_agent_timeout_sec()
        content = content.replace("{num_hours}", str(timeout_sec / 3600))
        content = content.replace("{setup_other}", benchmark_info.setup_note)
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content = content.replace(
                "{openai_restriction}",
                "- IMPORTANT: You are NOT allowed to use the OpenAI API for anything but the evaluation script.\n",
            )
        else:
            content = content.replace("{openai_restriction}", "")

        # Add target score inline if available and enabled
        target = INSTRUCT_BASELINES.get((model_info.model_id, benchmark_id))
        if self.include_target and target is not None:
            target_pct = f"{target * 100:.1f}%"
            content = content.replace(
                "{target_line}",
                f" Aim for at least **{target_pct}**. Keep iterating until you reach it or run out of time.",
            )
        else:
            content = content.replace("{target_line}", "")

        (task_dir / "instruction.md").write_text(content)

    def _read_agent_timeout_sec(self) -> int:
        """Read agent timeout from the task.toml template (single source of truth)."""
        import tomllib

        content = self._template("task.toml").read_text()
        config = tomllib.loads(content)
        return int(config["agent"]["timeout_sec"])

    def generate_timer_sh(self, env_dir: Path) -> None:
        timeout_sec = self._read_agent_timeout_sec()
        timer_script = f"""#!/bin/bash

TIMEOUT_SEC={timeout_sec}

# Container age from pid 1's elapsed time. /proc/uptime is NOT namespaced
# in RunPod/Docker, so it reports host uptime (days/weeks) and would
# immediately fire "Timer expired!".
ELAPSED=$(ps -o etimes= -p 1 2>/dev/null | tr -d ' ')
if ! [[ "$ELAPSED" =~ ^[0-9]+$ ]]; then
    # Fallback: first-call timestamp
    START_FILE="$(dirname "$0")/.timer_start"
    if [ ! -f "$START_FILE" ]; then
        date +%s > "$START_FILE"
    fi
    START_DATE=$(cat "$START_FILE")
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_DATE))
fi
REMAINING=$((TIMEOUT_SEC - ELAPSED))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
        timer_path = env_dir / "timer.sh"
        timer_path.write_text(timer_script)
        timer_path.chmod(0o755)

    def generate_environment(
        self, task_dir: Path, benchmark_id: str, model_info: ModelInfo, benchmark_info: BenchmarkInfo
    ) -> None:
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "gpu-runpod":
            # No Dockerfile — use setup.sh to install packages at runtime
            shutil.copy(TEMPLATE_DIR / "environment" / "setup.sh", env_dir / "setup.sh")
            # Background GPU sampler, started from setup.sh
            sampler_src = TEMPLATE_DIR / "environment" / "gpu_sampler.sh"
            if sampler_src.exists():
                shutil.copy(sampler_src, env_dir / "gpu_sampler.sh")
        else:
            dockerfile_name = "tinker.Dockerfile" if self.mode == "tinker" else "gpu.Dockerfile"
            shutil.copy(TEMPLATE_DIR / "environment" / dockerfile_name, env_dir / "Dockerfile")
            dockerignore_src = TEMPLATE_DIR / "environment" / ".dockerignore"
            if dockerignore_src.exists():
                shutil.copy(dockerignore_src, env_dir / ".dockerignore")

        if self.mode == "tinker":
            eval_src = SRC_DIR / "tasks" / benchmark_id / "evaluate_tinker.py"
            if eval_src.exists():
                shutil.copy(eval_src, env_dir / "evaluate.py")
            else:
                raise FileNotFoundError(f"evaluate_tinker.py not found: {eval_src}")
            # Copy shared tinker evaluation module
            tinker_eval_src = SRC_DIR / "tinker_util.py"
            if tinker_eval_src.exists():
                shutil.copy(tinker_eval_src, env_dir / "tinker_util.py")
        else:
            eval_src = SRC_DIR / "tasks" / benchmark_id / "evaluate.py"
            if eval_src.exists():
                shutil.copy(eval_src, env_dir / "evaluate.py")
            else:
                raise FileNotFoundError(f"evaluate.py not found: {eval_src}")

            templates_src = TEMPLATE_DIR / "environment" / "templates"
            if templates_src.exists():
                shutil.copytree(templates_src, env_dir / "templates", dirs_exist_ok=True)
            else:
                raise FileNotFoundError(f"templates directory not found: {templates_src}")

        eval_code_src = SRC_DIR / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, env_dir / "evaluation_code", dirs_exist_ok=True)

        task_context_src = SRC_DIR / "tasks" / benchmark_id / "task_context"
        if task_context_src.is_dir():
            for item in task_context_src.iterdir():
                dst = env_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dst)

        # NOTE: contamination judge files are copied into tests/ (not environment/)
        # so the agent cannot read or tamper with the judge logic.

        self.generate_timer_sh(env_dir)

        target = INSTRUCT_BASELINES.get((model_info.model_id, benchmark_id))
        base = BASE_SCORES.get((model_info.model_id, benchmark_id))
        max_conn = model_info.max_connections_long if benchmark_info.long_generation else model_info.max_connections
        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
            "mode": self.mode,
            "max_connections": max_conn,
        }
        if target is not None:
            metadata["target_score"] = target
        if base is not None:
            metadata["base_score"] = base

        # Regression suite: evals the verifier runs on final_model, excluding the target.
        # Only for GPU modes (tinker mode has a different eval flow).
        if self.mode != "tinker":
            regression_ids = [bid for bid in REGRESSION_EVALS if bid != benchmark_id]
            metadata["regression_benchmarks"] = regression_ids
            metadata["regression_baselines"] = {
                bid: BASE_SCORES.get((model_info.model_id, bid))
                for bid in regression_ids
            }

        (env_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def generate_tests(self, task_dir: Path, benchmark_id: str) -> None:
        """Copy pristine eval files into tests/ so the verifier uses untampered copies."""
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(self._template("tests", "test.sh"), test_sh_dst)
        test_sh_dst.chmod(0o755)

        verifier_src = TEMPLATE_DIR / "tests" / "verifier.py"
        if verifier_src.exists():
            shutil.copy(verifier_src, tests_dir / "verifier.py")

        if self.mode == "tinker":
            eval_src = SRC_DIR / "tasks" / benchmark_id / "evaluate_tinker.py"
        else:
            eval_src = SRC_DIR / "tasks" / benchmark_id / "evaluate.py"
        if eval_src.exists():
            shutil.copy(eval_src, tests_dir / "evaluate.py")
        if self.mode == "tinker":
            tinker_eval_src = SRC_DIR / "tinker_util.py"
            if tinker_eval_src.exists():
                shutil.copy(tinker_eval_src, tests_dir / "tinker_util.py")

        if self.mode != "tinker":
            templates_src = TEMPLATE_DIR / "environment" / "templates"
            if templates_src.exists():
                shutil.copytree(templates_src, tests_dir / "templates", dirs_exist_ok=True)

            # Regression suite evaluators. Each lives under tests/regression/<id>/evaluate.py
            # so regression_suite.py can invoke them without touching the target's evaluate.py.
            # Source preference: regression-only evals (mmlu/ifeval/truthfulqa) live under
            # harbor_template/regressions/; benchmarks that are also training targets
            # (gsm8k/humaneval/etc.) reuse src/tasks/<id>/evaluate.py.
            regression_root = tests_dir / "regression"
            for reg_id in REGRESSION_EVALS:
                if reg_id == benchmark_id:
                    continue
                regression_only_src = TEMPLATE_DIR / "regressions" / reg_id / "evaluate.py"
                task_src = SRC_DIR / "tasks" / reg_id / "evaluate.py"
                src = regression_only_src if regression_only_src.exists() else task_src
                if not src.exists():
                    continue
                dst_dir = regression_root / reg_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst_dir / "evaluate.py")
                reg_code = SRC_DIR / "tasks" / reg_id / "evaluation_code"
                if reg_code.is_dir():
                    shutil.copytree(reg_code, dst_dir / "evaluation_code", dirs_exist_ok=True)
            regression_runner = TEMPLATE_DIR / "tests" / "regression_suite.py"
            if regression_runner.exists():
                shutil.copy(regression_runner, tests_dir / "regression_suite.py")
            compute_parser = TEMPLATE_DIR / "tests" / "compute_parser.py"
            if compute_parser.exists():
                shutil.copy(compute_parser, tests_dir / "compute_parser.py")

        eval_code_src = SRC_DIR / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, tests_dir / "evaluation_code", dirs_exist_ok=True)

        # Copy metadata into tests/ so the verifier reads a tamper-proof copy
        env_metadata = task_dir / "environment" / "metadata.json"
        if env_metadata.exists():
            shutil.copy(env_metadata, tests_dir / "metadata.json")

        # Contamination judge + shared runner
        contam_src = TEMPLATE_DIR / "tests" / "contamination"
        if contam_src.is_dir():
            shutil.copytree(contam_src, tests_dir / "contamination", dirs_exist_ok=True)
        contam_sh = TEMPLATE_DIR / "tests" / "contamination.sh"
        if contam_sh.exists():
            shutil.copy(contam_sh, tests_dir / "contamination.sh")

    def generate_task(self, benchmark_id: str, model_key: str) -> Path:
        models = self._models
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in models:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(models.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = models[model_key]

        with contextlib.suppress(FileNotFoundError):
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )

        task_id = f"{benchmark_id}-{model_info.short_name}"
        if self.mode == "tinker":
            task_id += "-tinker"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        self.generate_task_toml(task_dir, benchmark_id, task_id)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id)

        solution_src = self._template("solution.sh")
        if solution_src.exists():
            solution_dir = task_dir / "solution"
            solution_dir.mkdir(parents=True, exist_ok=True)
            dst = solution_dir / "solve.sh"
            content = solution_src.read_text()
            content = content.replace("{instruct_id}", model_info.instruct_id or "")
            dst.write_text(content)
            dst.chmod(0o755)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def _benchmarks_for_mode(self) -> list[str]:
        if self._template_mode == "tinker":
            return [bid for bid in BENCHMARKS if (SRC_DIR / "tasks" / bid / "evaluate_tinker.py").exists()]
        return list(BENCHMARKS)

    def generate_all_tasks(self) -> list[Path]:
        tasks = []
        models = self._models
        for benchmark_id in self._benchmarks_for_mode():
            for model_key, model_info in models.items():
                key = (model_info.model_id, benchmark_id)
                target = INSTRUCT_BASELINES.get(key)
                base = BASE_SCORES.get(key)
                if target is None or base is None:
                    continue
                if self.include_target and base >= target:
                    print(
                        f"  WARNING: Skipping {benchmark_id}/{model_info.short_name}: "
                        f"base ({base:.3f}) >= target ({target:.3f}). "
                        f"Check if eval datasets match between base and instruct measurements."
                    )
                    continue
                tasks.append(self.generate_task(benchmark_id, model_key))
        return tasks


def list_available_tasks(mode: str = "gpu") -> list[str]:
    models = {k: v for k, v in MODELS.items() if v.tinker} if mode == "tinker" else MODELS
    suffix = "-tinker" if mode == "tinker" else ""
    if mode == "tinker":
        benchmarks = [bid for bid in BENCHMARKS if (SRC_DIR / "tasks" / bid / "evaluate_tinker.py").exists()]
    else:
        benchmarks = list(BENCHMARKS)
    return [f"{bid}-{models[mk].short_name}{suffix}" for bid in benchmarks for mk in models]


# --- CLI ---


def generate(
    benchmark: str | None = None,
    model: str | None = None,
    output: Path = Path("./datasets/posttrainbench"),
    num_hours: float | None = None,
    all: bool = False,
    list: bool = False,
    mode: str = "all",
    include_target: bool = True,
) -> None:
    """Generate Harbor tasks for PostTrainBench.

    Args:
        benchmark: Benchmark to generate (e.g. gsm8k, humaneval, aime2025).
        model: Base model to generate (e.g. qwen3-1.7b, smollm3-3b).
        output: Output directory for generated tasks.
        num_hours: Number of hours for the training task. Defaults to 10 for gpu mode, 1 for tinker mode.
        all: Generate tasks for all benchmark + model combinations.
        list: List available benchmarks and models.
        mode: Export mode: "gpu", "tinker", "gpu-runpod", or "all" (default, exports gpu+tinker+gpu-runpod).
        include_target: Include instruct baseline target score in the instruction (tinker mode only).
    """
    all_modes = ["tinker", "gpu", "gpu-runpod"]
    modes = all_modes if mode == "all" else [mode]

    if list:
        for m in modes:
            models = {k: v for k, v in MODELS.items() if v.tinker} if m == "tinker" else MODELS
            print(f"Mode: {m}")
            print("\nAvailable benchmarks:")
            for bm_id, bm_info in BENCHMARKS.items():
                print(f"  {bm_id}: {bm_info.benchmark_name}")
            print("\nAvailable models:")
            for model_key, model_info in models.items():
                print(f"  {model_key}: {model_info.model_id}")
            print("\nAvailable task combinations:")
            for task_id in list_available_tasks(m):
                print(f"  {task_id}")
            print()
        return

    total_tasks = []
    for m in modes:
        mode_output = output / m
        hours = num_hours if num_hours is not None else (1 if m == "tinker" else 10)
        adapter = PostTrainBenchAdapter(output_dir=mode_output, num_hours=hours, mode=m, include_target=include_target)

        if all:
            print(f"Generating {m} tasks to {mode_output}/...")
            tasks = adapter.generate_all_tasks()
            total_tasks.extend(tasks)
            print(f"  Generated {len(tasks)} {m} tasks.")
        else:
            if not benchmark or not model:
                print("Either all=true or both benchmark=X and model=Y are required")
                return
            task_dir = adapter.generate_task(benchmark, model)
            total_tasks.append(task_dir)

    if all and total_tasks:
        print(f"\nTotal: {len(total_tasks)} tasks.")
        print("\nTo run a task with Harbor:")
        print(f"  harbor run --path {total_tasks[0]} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")
        return

    if not all and total_tasks:
        print("\nTo run this task with Harbor:")
        print(f"  harbor run --path {total_tasks[0]} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")


if __name__ == "__main__":
    chz.entrypoint(generate)
