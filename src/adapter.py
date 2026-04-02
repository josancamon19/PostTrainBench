"""Generate Harbor-compatible tasks for running PostTrainBench evaluations."""

import contextlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import chz

TEMPLATE_DIR = Path(__file__).parent / "harbor_template"
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

# Models available on Tinker (subset that Tinker API supports)
# GSM8K reference scores (instruct version, 8-shot CoT): target after post-training
TINKER_MODELS = {
    # "qwen3-4b-instruct": ModelInfo(
    #     model_id="Qwen/Qwen3-4B-Instruct-2507", short_name="qwen3-4b-instruct"
    # ),  # Instruction, Dense — GSM8K: 82.2% (already instruct)
    "llama3.1-8b": ModelInfo(
        model_id="meta-llama/Llama-3.1-8B", short_name="llama3.1-8b"
    ),  # Base, Dense — base: 3.1%, instruct ref: 84.5%
    "llama3.2-3b": ModelInfo(
        model_id="meta-llama/Llama-3.2-3B", short_name="llama3.2-3b"
    ),  # Base, Dense — base: 6.1%, instruct ref: 77.7%
    "llama3.2-1b": ModelInfo(
        model_id="meta-llama/Llama-3.2-1B", short_name="llama3.2-1b"
    ),  # Base, Dense — base: 3.6%, instruct ref: 44.4%
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(self, output_dir: Path, num_hours: float = 10, include_claude_clause: bool = True, mode: str = "gpu"):
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.mode = mode  # "gpu" (default) or "tinker"

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        bench_file = SRC_DIR / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def _template(self, *parts: str) -> Path:
        """Resolve a template path, using .tinker variant when in tinker mode.

        E.g. ("task.toml") -> "task.tinker.toml", ("environment", "Dockerfile") -> "environment/Dockerfile.tinker".
        """
        if self.mode == "tinker":
            filename = parts[-1]
            name, dot, ext = filename.rpartition(".")
            if dot:
                tinker_filename = f"{name}.tinker.{ext}"
            else:
                tinker_filename = f"{filename}.tinker"
            prefixed = list(parts[:-1]) + [tinker_filename]
            candidate = TEMPLATE_DIR / Path(*prefixed)
            if candidate.exists():
                return candidate
        return TEMPLATE_DIR / Path(*parts)

    @property
    def _models(self) -> dict[str, ModelInfo]:
        return TINKER_MODELS if self.mode == "tinker" else MODELS

    def generate_task_toml(self, task_dir: Path, benchmark_id: str = "") -> None:
        import re

        content = self._template("task.toml").read_text()
        agent_timeout = self.num_hours * 3600
        # Replace the agent timeout (last timeout_sec in [agent] section)
        content = re.sub(
            r"(\[agent\].*?timeout_sec\s*=\s*)[\d.]+",
            rf"\g<1>{float(agent_timeout)}",
            content,
            count=1,
            flags=re.DOTALL,
        )
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content += '\n[agent.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'
        (task_dir / "task.toml").write_text(content)

    def generate_instruction(
        self, task_dir: Path, model_info: ModelInfo, benchmark_info: BenchmarkInfo, benchmark_id: str = ""
    ) -> None:
        content = self._template("instruction.md").read_text()
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
        timeout_sec = int(self.num_hours * 3600)
        timer_script = f"""#!/bin/bash

TIMEOUT_SEC={timeout_sec}

# Use container boot time as the start time (synced with agent timeout)
START_DATE=$(awk '/btime/{{print $2}}' /proc/stat 2>/dev/null)
if [ -z "$START_DATE" ]; then
    # Fallback for non-Linux: first-call timestamp
    START_FILE="$(dirname "$0")/.timer_start"
    if [ ! -f "$START_FILE" ]; then
        date +%s > "$START_FILE"
    fi
    START_DATE=$(cat "$START_FILE")
fi

DEADLINE=$((START_DATE + TIMEOUT_SEC))
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

    def generate_environment(
        self, task_dir: Path, benchmark_id: str, model_info: ModelInfo, benchmark_info: BenchmarkInfo
    ) -> None:
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(self._template("environment", "Dockerfile"), env_dir / "Dockerfile")
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

        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(self._template("tests", "test.sh"), test_sh_dst)
        test_sh_dst.chmod(0o755)

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

        eval_code_src = SRC_DIR / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, tests_dir / "evaluation_code", dirs_exist_ok=True)

        judge_src = TEMPLATE_DIR / "judge" / "contamination_judge.py"
        if judge_src.exists():
            shutil.copy(judge_src, tests_dir / "contamination_judge.py")
        judge_prompt_src = TEMPLATE_DIR / "judge" / "contamination_judge_prompt.txt"
        if judge_prompt_src.exists():
            shutil.copy(judge_prompt_src, tests_dir / "contamination_judge_prompt.txt")

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

        self.generate_task_toml(task_dir, benchmark_id)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id)

        solution_src = self._template("solution.sh")
        if solution_src.exists():
            solution_dir = task_dir / "solution"
            solution_dir.mkdir(parents=True, exist_ok=True)
            dst = solution_dir / "solve.sh"
            shutil.copy(solution_src, dst)
            dst.chmod(0o755)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        tasks = []
        models = self._models
        for benchmark_id in BENCHMARKS:
            for model_key in models:
                tasks.append(self.generate_task(benchmark_id, model_key))
        return tasks


def list_available_tasks(mode: str = "gpu") -> list[str]:
    models = TINKER_MODELS if mode == "tinker" else MODELS
    suffix = "-tinker" if mode == "tinker" else ""
    return [f"{bid}-{models[mk].short_name}{suffix}" for bid in BENCHMARKS for mk in models]


# --- CLI ---


def generate(
    benchmark: str | None = None,
    model: str | None = None,
    output: Path = Path("./datasets/posttrainbench"),
    num_hours: float = 10,
    all: bool = False,
    list: bool = False,
    mode: str = "gpu",
) -> None:
    """Generate Harbor tasks for PostTrainBench.

    Args:
        benchmark: Benchmark to generate (e.g. gsm8k, humaneval, aime2025).
        model: Base model to generate (e.g. qwen3-1.7b, smollm3-3b).
        output: Output directory for generated tasks.
        num_hours: Number of hours for the training task.
        all: Generate tasks for all benchmark + model combinations.
        list: List available benchmarks and models.
        mode: Export mode: "gpu" (default, local GPU training) or "tinker" (Tinker API training).
    """
    if list:
        models = TINKER_MODELS if mode == "tinker" else MODELS
        print(f"Mode: {mode}")
        print("\nAvailable benchmarks:")
        for bm_id, bm_info in BENCHMARKS.items():
            print(f"  {bm_id}: {bm_info.benchmark_name}")
        print("\nAvailable models:")
        for model_key, model_info in models.items():
            print(f"  {model_key}: {model_info.model_id}")
        print("\nAvailable task combinations:")
        for task_id in list_available_tasks(mode):
            print(f"  {task_id}")
        return

    adapter = PostTrainBenchAdapter(output_dir=output, num_hours=num_hours, mode=mode)

    if all:
        print(f"Generating all tasks to {output}/...")
        tasks = adapter.generate_all_tasks()
        print(f"\nGenerated {len(tasks)} tasks.")
        print("\nTo run a task with Harbor:")
        print(f"  harbor run --path {tasks[0]} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")
        return

    if not benchmark or not model:
        print("Either all=true or both benchmark=X and model=Y are required")
        return

    task_dir = adapter.generate_task(benchmark, model)
    print("\nTo run this task with Harbor:")
    print(f"  harbor run --path {task_dir} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")


if __name__ == "__main__":
    chz.entrypoint(generate)
