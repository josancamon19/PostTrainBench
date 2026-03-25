"""Generate Harbor-compatible tasks for running PostTrainBench evaluations."""

import contextlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import chz

BASE_TEMPLATE_DIR = Path(__file__).parent / "base_template"
HARBOR_TEMPLATE_DIR = Path(__file__).parent / "harbor_template"
PRIME_RL_TEMPLATE_DIR = Path(__file__).parent / "prime_rl_template"
SRC_DIR = Path(__file__).parent

CLAUDE_CLAUSE = (
    "\nYou are running in a non-interactive mode. "
    "So make sure every process you are running finishes before you write your last message.\n"
)


@dataclass
class BenchmarkInfo:
    task_id: str
    benchmark_name: str
    setup_note: str = ""


@dataclass
class ModelInfo:
    model_id: str
    short_name: str


BENCHMARKS = {
    "gsm8k": BenchmarkInfo(
        task_id="gsm8k",
        benchmark_name="GSM8K (Grade School Math 8K)",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
        "Do not be alarmed by this, this is normal behavior for inspect-ai. "
        "Also if you run into issues with the evaluate.py script, this is likely "
        "due to memory constraints on the GPU. In this case please decrease "
        "--max-connections or --max-tokens.\n",
    ),
    "humaneval": BenchmarkInfo(
        task_id="humaneval",
        benchmark_name="HumanEval",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
        "Do not be alarmed by this, this is normal behavior for inspect-ai.\n",
    ),
    "aime2025": BenchmarkInfo(
        task_id="aime2025",
        benchmark_name="AIME 2025",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
        "Do not be alarmed by this, this is normal behavior for inspect-ai.\n",
    ),
    "gpqamain": BenchmarkInfo(
        task_id="gpqamain",
        benchmark_name="GPQA",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
        "Do not be alarmed by this, this is normal behavior for inspect-ai.\n",
    ),
    "bfcl": BenchmarkInfo(
        task_id="bfcl",
        benchmark_name="Berkeley Function Calling Leaderboard",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
        "Do not be alarmed by this, this is normal behavior for inspect-ai.\n",
    ),
    "arenahardwriting": BenchmarkInfo(
        task_id="arenahardwriting",
        benchmark_name="Arena-Hard-v2.0 (Writing)",
        setup_note="",
    ),
    "healthbench": BenchmarkInfo(
        task_id="healthbench",
        benchmark_name="HealthBench",
        setup_note="",
    ),
}

MODELS = {
    "qwen3-1.7b": ModelInfo(model_id="Qwen/Qwen3-1.7B-Base", short_name="qwen3-1.7b"),
    "qwen3-4b": ModelInfo(model_id="Qwen/Qwen3-4B-Base", short_name="qwen3-4b"),
    "smollm3-3b": ModelInfo(model_id="HuggingFaceTB/SmolLM3-3B-Base", short_name="smollm3-3b"),
    "gemma3-4b": ModelInfo(model_id="google/gemma-3-4b-pt", short_name="gemma3-4b"),
}


PRIME_RL_MODELS = {
    "qwen3-4b": ModelInfo(model_id="Qwen/Qwen3-4B-Instruct-2507", short_name="qwen3-4b"),
    "smollm3-3b": ModelInfo(model_id="HuggingFaceTB/SmolLM3-3B", short_name="smollm3-3b"),
    "llama3.2-3b": ModelInfo(model_id="meta-llama/Llama-3.2-3B-Instruct", short_name="llama3.2-3b"),
}

PRIME_RL_BENCHMARKS = {"gsm8k", "aime2025"}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: float = 10,
        include_claude_clause: bool = True,
        mode: str = "default",
    ):
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.mode = mode
        if mode == "prime-rl":
            self.template_dir = PRIME_RL_TEMPLATE_DIR
        else:
            self.template_dir = HARBOR_TEMPLATE_DIR

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        bench_file = SRC_DIR / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def generate_task_toml(self, task_dir: Path, benchmark_id: str = "") -> None:
        content = (self.template_dir / "task.toml").read_text()
        agent_timeout = self.num_hours * 3600
        content = content.replace("timeout_sec = 36000.0", f"timeout_sec = {float(agent_timeout)}")
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content += '\n[agent.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
        (task_dir / "task.toml").write_text(content)

    def generate_instruction(
        self, task_dir: Path, model_info: ModelInfo, benchmark_info: BenchmarkInfo, benchmark_id: str = ""
    ) -> None:
        content = (self.template_dir / "instruction.md").read_text()
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        content = content.replace("{num_hours}", str(self.num_hours))
        content = content.replace("{setup_other}", benchmark_info.setup_note)
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content = content.replace(
                "{openai_restriction}",
                "- IMPORTANT: You are NOT allowed to use the OpenAI API for anything but the evaluation script.\n",
            )
        else:
            content = content.replace("{openai_restriction}", "")
        if self.include_claude_clause:
            content += CLAUDE_CLAUSE
        (task_dir / "instruction.md").write_text(content)

    def generate_timer_sh(self, env_dir: Path) -> None:
        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}

START_FILE="$(dirname "$0")/.timer_start"
if [ ! -f "$START_FILE" ]; then
    date +%s > "$START_FILE"
fi
START_DATE=$(cat "$START_FILE")

DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

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

    def _copy_dir(self, src: Path, dst: Path) -> None:
        """Copy all files from src into dst, creating dirs as needed."""
        if not src.exists():
            return
        for item in src.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src)
                dest = dst / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(item, dest)

    def generate_environment(
        self, task_dir: Path, benchmark_id: str, model_info: ModelInfo, benchmark_info: BenchmarkInfo
    ) -> None:
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy base template (shared files: templates, judge, dockerignore)
        self._copy_dir(BASE_TEMPLATE_DIR / "environment", env_dir)

        # 2. Overlay mode-specific files (Dockerfile, etc.)
        self._copy_dir(self.template_dir / "environment", env_dir)

        # 3. Benchmark-specific files
        eval_src = SRC_DIR / "tasks" / benchmark_id / "evaluate.py"
        if eval_src.exists():
            shutil.copy(eval_src, env_dir / "evaluate.py")
        else:
            raise FileNotFoundError(f"evaluate.py not found: {eval_src}")

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

        self.generate_timer_sh(env_dir)

        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
            "mode": self.mode,
        }
        (env_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def generate_tests(self, task_dir: Path, benchmark_id: str) -> None:
        """Copy pristine eval files into tests/ so the verifier uses untampered copies."""
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy base template tests (test.sh, verifier.py)
        self._copy_dir(BASE_TEMPLATE_DIR / "tests", tests_dir)

        # 2. Overlay mode-specific tests (if any)
        self._copy_dir(self.template_dir / "tests", tests_dir)

        # Make test.sh executable
        test_sh = tests_dir / "test.sh"
        if test_sh.exists():
            test_sh.chmod(0o755)

        # 3. Benchmark-specific eval files
        eval_src = SRC_DIR / "tasks" / benchmark_id / "evaluate.py"
        if eval_src.exists():
            shutil.copy(eval_src, tests_dir / "evaluate.py")

        templates_src = BASE_TEMPLATE_DIR / "environment" / "templates"
        if templates_src.exists():
            shutil.copytree(templates_src, tests_dir / "templates", dirs_exist_ok=True)

        eval_code_src = SRC_DIR / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, tests_dir / "evaluation_code", dirs_exist_ok=True)

    def generate_task(self, benchmark_id: str, model_key: str) -> Path:
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if self.mode == "prime-rl" and benchmark_id not in PRIME_RL_BENCHMARKS:
            raise ValueError(
                f"Benchmark {benchmark_id!r} not yet supported in prime-rl mode. "
                f"Available: {sorted(PRIME_RL_BENCHMARKS)}"
            )

        benchmark_info = BENCHMARKS[benchmark_id]
        models = PRIME_RL_MODELS if self.mode == "prime-rl" else MODELS
        if model_key not in models:
            raise ValueError(f"Unknown model for {self.mode} mode: {model_key}. Available: {list(models.keys())}")
        model_info = models[model_key]

        with contextlib.suppress(FileNotFoundError):
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )

        prefix = "prime-rl-" if self.mode == "prime-rl" else ""
        task_id = f"{prefix}{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        self.generate_task_toml(task_dir, benchmark_id)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        tasks = []
        benchmarks = PRIME_RL_BENCHMARKS if self.mode == "prime-rl" else BENCHMARKS
        models = PRIME_RL_MODELS if self.mode == "prime-rl" else MODELS
        for benchmark_id in benchmarks:
            for model_key in models:
                tasks.append(self.generate_task(benchmark_id, model_key))
        return tasks


def list_available_tasks() -> list[str]:
    return [f"{bid}-{MODELS[mk].short_name}" for bid in BENCHMARKS for mk in MODELS]


# --- CLI ---


def generate(
    benchmark: str | None = None,
    model: str | None = None,
    output: Path = Path("./datasets/posttrainbench"),
    num_hours: float = 10,
    mode: str = "default",
    all: bool = False,
    list: bool = False,
) -> None:
    """Generate Harbor tasks for PostTrainBench.

    Args:
        benchmark: Benchmark to generate (e.g. gsm8k, humaneval, aime2025).
        model: Base model to generate (e.g. qwen3-1.7b, smollm3-3b).
        output: Output directory for generated tasks.
        num_hours: Number of hours for the training task.
        mode: Task mode. "default" for raw GPU training, "prime-rl" for prime-rl based training.
        all: Generate tasks for all benchmark + model combinations.
        list: List available benchmarks and models.
    """
    if list:
        print("Available benchmarks:")
        benchmarks = PRIME_RL_BENCHMARKS if mode == "prime-rl" else BENCHMARKS
        for bm_id in benchmarks:
            bm_info = BENCHMARKS[bm_id]
            print(f"  {bm_id}: {bm_info.benchmark_name}")
        print("\nAvailable models:")
        models = PRIME_RL_MODELS if mode == "prime-rl" else MODELS
        for model_key, model_info in models.items():
            print(f"  {model_key}: {model_info.model_id}")
        print(f"\nMode: {mode}")
        return

    env_type = "daytona" if mode == "prime-rl" else "modal"
    adapter = PostTrainBenchAdapter(output_dir=output, num_hours=num_hours, mode=mode)

    if all:
        print(f"Generating all {mode} tasks to {output}/...")
        tasks = adapter.generate_all_tasks()
        print(f"\nGenerated {len(tasks)} tasks.")
        print("\nTo run a task with Harbor:")
        print(f"  harbor run --path {tasks[0]} --agent claude-code --model anthropic/claude-sonnet-4 --env {env_type}")
        return

    if not benchmark or not model:
        print("Either all=true or both benchmark=X and model=Y are required")
        return

    task_dir = adapter.generate_task(benchmark, model)
    print("\nTo run this task with Harbor:")
    print(f"  harbor run --path {task_dir} --agent claude-code --model anthropic/claude-sonnet-4 --env {env_type}")


if __name__ == "__main__":
    chz.entrypoint(generate)
