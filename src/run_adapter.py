"""Generate Harbor-compatible tasks for running PostTrainBench evaluations."""

from pathlib import Path

import chz

from adapter import (
    PostTrainBenchAdapter,
    BENCHMARKS,
    MODELS,
    list_available_tasks,
)


def generate(
    benchmark: str | None = None,
    model: str | None = None,
    output: Path = Path("./datasets/posttrainbench"),
    num_hours: int = 10,
    all: bool = False,
    list: bool = False,
) -> None:
    """Generate Harbor tasks for PostTrainBench.

    Args:
        benchmark: Benchmark to generate (e.g. gsm8k, humaneval, aime2025).
        model: Base model to generate (e.g. qwen3-1.7b, smollm3-3b).
        output: Output directory for generated tasks.
        num_hours: Number of hours for the training task.
        all: Generate tasks for all benchmark + model combinations.
        list: List available benchmarks and models.
    """
    if list:
        print("Available benchmarks:")
        for bm_id, bm_info in BENCHMARKS.items():
            print(f"  {bm_id}: {bm_info.benchmark_name}")
        print("\nAvailable models:")
        for model_key, model_info in MODELS.items():
            print(f"  {model_key}: {model_info.model_id}")
        print("\nAvailable task combinations:")
        for task_id in list_available_tasks():
            print(f"  {task_id}")
        return

    adapter = PostTrainBenchAdapter(output_dir=output, num_hours=num_hours)

    if all:
        print(f"Generating all tasks to {output}/...")
        tasks = adapter.generate_all_tasks()
        print(f"\nGenerated {len(tasks)} tasks.")
        print(f"\nTo run a task with Harbor:")
        print(f"  harbor run --path {tasks[0]} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")
        return

    if not benchmark or not model:
        print("Either --all or both --benchmark and --model are required")
        return

    task_dir = adapter.generate_task(benchmark, model)
    print(f"\nTo run this task with Harbor:")
    print(f"  harbor run --path {task_dir} --agent claude-code --model anthropic/claude-sonnet-4 --env modal")


if __name__ == "__main__":
    chz.entrypoint(generate)
