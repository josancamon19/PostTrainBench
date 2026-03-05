"""Run PostTrainBench trials on Modal.

Convenience wrapper around Harbor's Trial API that handles:
  - Automatic HF cache volume creation and population (idempotent)
  - Volume mounting for the shared HuggingFace cache
  - Batch execution with bounded concurrency

Usage:
    # Single trial (HF cache volume is set up automatically)
    python run_job.py \
        --task-dir ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
        --agent claude-code \
        --model anthropic/claude-sonnet-4 \
        --modal-secret my-api-keys

    # Skip automatic cache population (volume already populated)
    python run_job.py \
        --task-dir ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
        --agent claude-code \
        --model anthropic/claude-sonnet-4 \
        --no-ensure-cache

    # Without HF cache volume (models downloaded at runtime)
    python run_job.py \
        --task-dir ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
        --agent claude-code \
        --model anthropic/claude-sonnet-4 \
        --no-hf-cache

    # Multiple trials from a directory of tasks
    python run_job.py \
        --tasks-root ./tasks \
        --agent claude-code \
        --model anthropic/claude-sonnet-4 \
        --n-concurrent 4
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from modal_volume import DEFAULT_VOLUME_NAME, ensure_hf_cache

# Path inside the container where the Modal volume is mounted
HF_CACHE_VOLUME_MOUNT = "/hf-cache-volume"


def _build_trial_config(
    task_dir: Path,
    agent_name: str,
    model_name: str,
    hf_cache_volume: str | None,
    modal_secrets: list[str],
    agent_env: dict[str, str],
    trials_dir: Path,
) -> TrialConfig:
    """Build a TrialConfig for a PostTrainBench trial on Modal."""
    env_kwargs: dict[str, Any] = {}
    if hf_cache_volume:
        env_kwargs["volumes"] = {HF_CACHE_VOLUME_MOUNT: hf_cache_volume}
    if modal_secrets:
        env_kwargs["secrets"] = modal_secrets

    return TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(
            name=agent_name,
            model_name=model_name,
            env=agent_env,
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.MODAL,
            kwargs=env_kwargs,
        ),
    )


async def _run_single_trial(
    task_dir: Path,
    agent_name: str,
    model_name: str,
    hf_cache_volume: str | None,
    modal_secrets: list[str],
    agent_env: dict[str, str],
    trials_dir: Path,
) -> None:
    """Create a Trial and run it."""
    config = _build_trial_config(
        task_dir=task_dir,
        agent_name=agent_name,
        model_name=model_name,
        hf_cache_volume=hf_cache_volume,
        modal_secrets=modal_secrets,
        agent_env=agent_env,
        trials_dir=trials_dir,
    )

    trial = Trial(config)

    print(f"Running trial: {config.trial_name}")
    print(f"  Task: {task_dir}")
    print(f"  Agent: {agent_name} ({model_name})")
    if hf_cache_volume:
        print(f"  HF cache volume: {hf_cache_volume} -> {HF_CACHE_VOLUME_MOUNT}")

    result = await trial.run()

    print(f"\nTrial completed: {result.trial_name}")
    if result.verifier_result:
        print(f"  Rewards: {result.verifier_result.rewards}")
    if result.exception_info:
        print(f"  Exception: {result.exception_info.exception_type}: "
              f"{result.exception_info.exception_message}")


async def _run_batch(
    task_dirs: list[Path],
    agent_name: str,
    model_name: str,
    hf_cache_volume: str | None,
    modal_secrets: list[str],
    agent_env: dict[str, str],
    trials_dir: Path,
    n_concurrent: int,
) -> None:
    """Run multiple trials with bounded concurrency."""
    semaphore = asyncio.Semaphore(n_concurrent)

    async def run_with_semaphore(task_dir: Path) -> None:
        async with semaphore:
            await _run_single_trial(
                task_dir=task_dir,
                agent_name=agent_name,
                model_name=model_name,
                hf_cache_volume=hf_cache_volume,
                modal_secrets=modal_secrets,
                agent_env=agent_env,
                trials_dir=trials_dir,
            )

    async with asyncio.TaskGroup() as tg:
        for task_dir in task_dirs:
            tg.create_task(run_with_semaphore(task_dir))


def _parse_agent_env(env_args: list[str] | None) -> dict[str, str]:
    """Parse KEY=VALUE environment variable arguments."""
    env = {}
    for arg in env_args or []:
        if "=" not in arg:
            raise ValueError(f"Invalid env format '{arg}', expected KEY=VALUE")
        key, _, value = arg.partition("=")
        # Resolve ${VAR} references from the host environment
        if value.startswith("${") and value.endswith("}"):
            host_var = value[2:-1]
            resolved = os.environ.get(host_var)
            if resolved is None:
                raise RuntimeError(
                    f"Environment variable {host_var} not set on host"
                )
            value = resolved
        env[key] = value
    return env


def main():
    parser = argparse.ArgumentParser(
        description="Run PostTrainBench trials on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Task selection (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task-dir", "-t",
        type=Path,
        help="Path to a single task directory",
    )
    task_group.add_argument(
        "--tasks-root",
        type=Path,
        help="Path to directory containing multiple task directories",
    )

    # Agent configuration
    parser.add_argument(
        "--agent", "-a",
        type=str,
        required=True,
        help="Agent name (e.g. claude-code, codex, aider)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name (e.g. anthropic/claude-sonnet-4)",
    )

    # Modal configuration
    parser.add_argument(
        "--hf-cache-volume",
        type=str,
        default=DEFAULT_VOLUME_NAME,
        help=f"Modal volume name for the HuggingFace cache "
             f"(default: {DEFAULT_VOLUME_NAME}). "
             "Use --no-hf-cache to disable.",
    )
    parser.add_argument(
        "--no-hf-cache",
        action="store_true",
        help="Don't use a HF cache volume (models downloaded at runtime)",
    )
    parser.add_argument(
        "--no-ensure-cache",
        action="store_true",
        help="Skip automatic cache population "
             "(assumes the volume is already populated)",
    )
    parser.add_argument(
        "--modal-secret",
        type=str,
        action="append",
        default=[],
        dest="modal_secrets",
        help="Modal secret name to mount (can be repeated)",
    )

    # Agent environment
    parser.add_argument(
        "--ae", "--agent-env",
        type=str,
        action="append",
        default=[],
        dest="agent_env_args",
        help="Environment variable for the agent as KEY=VALUE (can be repeated). "
             "Use ${VAR} to reference host environment variables.",
    )

    # Execution
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=Path("trials"),
        help="Directory to store trial results (default: ./trials)",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Number of concurrent trials when using --tasks-root (default: 1)",
    )

    args = parser.parse_args()

    hf_cache_volume = None if args.no_hf_cache else args.hf_cache_volume

    # Ensure the HF cache volume is created and populated before running
    # trials.  This is idempotent (skips already-cached resources).
    if hf_cache_volume and not args.no_ensure_cache:
        ensure_hf_cache()

    agent_env = _parse_agent_env(args.agent_env_args)

    if args.task_dir:
        asyncio.run(
            _run_single_trial(
                task_dir=args.task_dir,
                agent_name=args.agent,
                model_name=args.model,
                hf_cache_volume=hf_cache_volume,
                modal_secrets=args.modal_secrets,
                agent_env=agent_env,
                trials_dir=args.trials_dir,
            )
        )
    else:
        task_dirs = sorted([
            d for d in args.tasks_root.iterdir()
            if d.is_dir() and (d / "task.toml").exists()
        ])
        if not task_dirs:
            raise FileNotFoundError(
                f"No task directories found in {args.tasks_root}"
            )
        print(f"Found {len(task_dirs)} tasks in {args.tasks_root}")
        asyncio.run(
            _run_batch(
                task_dirs=task_dirs,
                agent_name=args.agent,
                model_name=args.model,
                hf_cache_volume=hf_cache_volume,
                modal_secrets=args.modal_secrets,
                agent_env=agent_env,
                trials_dir=args.trials_dir,
                n_concurrent=args.n_concurrent,
            )
        )


if __name__ == "__main__":
    main()
