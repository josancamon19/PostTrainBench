# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

PostTrainBench is a benchmark that measures CLI agents' ability to post-train pre-trained LLMs. Agents get 10 hours on an H100 GPU to improve a base model's performance on a given benchmark. The project uses the **Harbor** framework for task orchestration and supports GPU (vLLM/Modal) and Tinker (remote API/Daytona) execution modes.

## Commands

```bash
# Setup
uv sync                    # Install dependencies into .venv
source .venv/bin/activate

# Lint
ruff check .               # Line length 120, Python 3.13 target
ruff check --fix .         # Auto-fix

# Generate Harbor tasks
python src/adapter.py mode=all all=true include_target=false      # All modes (gpu + tinker + runpod)
python src/adapter.py mode=gpu all=true include_target=false      # All GPU tasks
python src/adapter.py mode=runpod all=true include_target=false   # All RunPod tasks (no Dockerfile)
python src/adapter.py mode=gpu benchmark=gsm8k model=llama3.1-8b  # Single task
# Output goes to datasets/posttrainbench/{gpu,tinker,runpod}/{task_id}/

# Build Docker images (base + per-task, pushes to GHCR)
bash src/build.sh

# Run jobs
python -m daytona run src/configs/job-gpu.yaml      # GPU mode (Modal)
python -m daytona run src/configs/job-tinker.yaml    # Tinker mode (Daytona)
python -m daytona run src/configs/job-gpu-runpod.yaml # GPU on RunPod

# Compute baseline scores
python scripts/run_baselines.py
```

## Architecture

### Core Flow

1. **`src/adapter.py`** — The main entry point. `PostTrainBenchAdapter` generates Harbor-compatible task directories from templates + constants. Uses `chz` for CLI arg parsing. Each generated task includes a `task.toml`, `instruction.md`, Dockerfile, evaluation scripts, and a verifier.

2. **`src/constants.py`** — Defines `BENCHMARKS` (7 tasks), `MODELS`/`TINKER_MODELS`, `BASE_SCORES`, and `INSTRUCT_BASELINES`. This is where you register new benchmarks and models.

3. **`src/harbor_template/`** — Templates that `adapter.py` fills in:
   - `task.toml` / `task.tinker.toml` — Harbor task metadata, resources, timeouts
   - `instruction.md` / `instruction.tinker.md` — Agent prompt with placeholders (`{model}`, `{benchmark}`, `{num_hours}`, etc.)
   - `environment/base/gpu.Dockerfile` — GPU base: CUDA 12.9, Python 3.10, vLLM, full ML stack
   - `environment/base/tinker.Dockerfile` — Tinker base: Python 3.11, Tinker SDK
   - `environment/gpu.Dockerfile` — Per-task GPU layer (copies evaluate.py, prompt templates)
   - `environment/tinker.Dockerfile` — Per-task Tinker layer
   - `environment/runpod-ssh.gpu.Dockerfile` — RunPod SSH overlay (wraps any GPU image)
   - `tests/test.sh` — Verifier with 3-phase eval retry (reduces token limits on GPU OOM)
   - `tests/verifier.py` — Validation helpers (model existence, metadata)
   - `judge/contamination_judge.py` — AI judge detecting data tampering and model substitution

4. **`src/tasks/`** — Each subdirectory is a benchmark containing `benchmark.txt`, `evaluate.py` (GPU/Inspect AI), and optionally `evaluate_tinker.py`, `task_context/`, `evaluation_code/`.

5. **`src/harbor_patch.py`** — Monkeypatches Harbor's Modal environment to resolve task env vars.

### Execution Modes

- **GPU** (`mode=gpu`): Local vLLM inference on H100s via Modal. Pre-built Docker images on GHCR tagged as `ghcr.io/josancamon19/posttrainbench-gpu:{task_id}`. Exported tasks include a Dockerfile.
- **RunPod** (`mode=runpod`): Same as GPU but uses prebuilt GHCR images directly (`-gpu:{task_id}-runpod`). Exported tasks have NO Dockerfile — only `docker_image` in task.toml.
- **Tinker** (`mode=tinker`): Remote API for training/inference via Daytona. No local GPU needed.

### Verification Pipeline

The verifier (`tests/test.sh`) runs after the agent finishes:
1. Checks `final_model/config.json` exists
2. Runs the contamination judge (checks for test data usage and model substitution)
3. Evaluates with 3-phase retry: Phase 1 (default tokens, 4 attempts) → Phase 2 (reduced tokens, 3 attempts) → Phase 3 (further reduced, 2 attempts)
4. Outputs: `metrics.json`, `reward.txt` (0-1 score), `contamination_judgement.txt`

### Job Configs

YAML files in `src/configs/` define agent, environment type, orchestrator, and dataset paths. Key fields: `agents[].model_name`, `environment.type` (modal/runpod/daytona), `orchestrator.n_concurrent_trials`.

## Adding a New Benchmark

1. Create `src/tasks/{task_name}/` with `benchmark.txt` and `evaluate.py` (use Inspect AI; see existing tasks)
2. Register in `src/constants.py`: add to `BENCHMARKS` dict, add baseline scores to `BASE_SCORES` and `INSTRUCT_BASELINES`
3. Run `python scripts/run_baselines.py` to compute baselines
4. Push to trigger CI image rebuild

## Environment Variables

```
OPENAI_API_KEY    — Required for contamination judge + arenahardwriting/healthbench evals
HF_TOKEN          — HuggingFace model access
ANTHROPIC_API_KEY — Claude Code agent
TINKER_API_KEY    — Tinker training API
DAYTONA_API_KEY   — Daytona remote execution
```

## Notes

- The 6 benchmarks: GSM8K, HumanEval, AIME 2025, GPQA, Arena-Hard Writing, HealthBench
- Models: Qwen3-1.7B, Qwen3-4B, SmolLM3-3B, Gemma-3-4B (see `constants.py` for current list)
- `evaluate.py` ERROR warnings from inspect-ai are normal
- CI (`.github/workflows/build-images.yml`) builds base + per-task Docker images on push to main when Dockerfiles or adapter.py change
- `datasets/`, `results/`, `jobs/` directories are gitignored (generated outputs)
