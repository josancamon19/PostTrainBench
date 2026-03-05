# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations on cloud GPUs.

## Supported Benchmarks

| Benchmark ID | Name | Type | Notes |
|-------------|------|------|-------|
| gsm8k | GSM8K (Grade School Math 8K) | inspect-ai | |
| humaneval | HumanEval | inspect-ai | |
| aime2025 | AIME 2025 | inspect-ai | |
| gpqamain | GPQA | inspect-ai | |
| bfcl | Berkeley Function Calling Leaderboard | inspect-ai | Includes `bfcl_evaluation_code.py` via task_context |
| arenahardwriting | Arena-Hard-v2.0 (Writing) | vLLM + OpenAI judge | Requires `OPENAI_API_KEY` for agent |
| healthbench | HealthBench | vLLM + OpenAI judge | Requires `OPENAI_API_KEY` for agent |

## Supported Models

| Key | HuggingFace Model ID |
|-----|---------------------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

Total: **28 tasks** (7 benchmarks x 4 models).

## File Layout

```
src/harbor_adapter/
├── adapter.py         # Task generation logic (benchmarks, models, templates)
├── run_adapter.py     # CLI: generate Harbor tasks from PostTrainBench config
├── run_job.py         # CLI: run trials on Modal (convenience wrapper for cache + volume)
├── modal_volume.py    # Modal volume creation and HF cache population
├── __init__.py        # Public API exports
├── template/          # Task templates (Dockerfile, instruction.md, tests, etc.)
└── tasks/             # Generated task directories (created by run_adapter.py)
```

## Installation

```bash
# Use the included pyproject.toml file to get the python environment with harbor and modal
uv sync
```

## Quick Start

### 1. Generate tasks

```bash
cd src/harbor_adapter

# Generate a single task
python run_adapter.py --benchmark gsm8k --model qwen3-1.7b --output ./tasks

# Or generate all 28 task combinations
python run_adapter.py --all --output ./tasks

# List available benchmarks and models
python run_adapter.py --list
```

### 2. Set API keys

```bash
python -m modal setup                # Modal cloud setup
export ANTHROPIC_API_KEY=<your-key>  # For Claude agent
export OPENAI_API_KEY=<your-key>     # For contamination judge (codex CLI) + arenahardwriting/healthbench eval
```

### 3. Run trials

Use `harbor run` to run trials on Modal. The overlay setup and timer initialization are handled automatically inside the Docker image via `BASH_ENV` (no external hooks needed).

```bash
cd src/harbor_adapter

# With HF cache volume (mount the pre-populated Modal volume)
harbor run \
    --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --env modal \
    --ek 'volumes={"/hf-cache-volume":"posttrainbench-hf-cache"}'

# Without HF cache volume (models downloaded at runtime)
harbor run \
    --path ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --env modal
```

Alternatively, use `run_job.py` as a convenience wrapper that handles cache population and volume mounting automatically:

```bash
# Single trial (HF cache volume is set up automatically on first run)
python run_job.py \
    --task-dir ./tasks/posttrainbench-gsm8k-qwen3-1.7b \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --modal-secret my-api-keys

# Multiple trials with concurrency
python run_job.py \
    --tasks-root ./tasks \
    --agent claude-code \
    --model anthropic/claude-sonnet-4 \
    --n-concurrent 4

# Skip cache population (volume already populated from a previous run)
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
```

## HF Cache Volume

PostTrainBench tasks fine-tune large models that benefit from a pre-populated HuggingFace cache. The adapter uses a **Modal volume** (`posttrainbench-hf-cache`) shared across all trials, containing pre-downloaded models and datasets from `containers/download_hf_cache/resources.json`.

### How it works

1. **Volume creation**: `modal_volume.py` creates the Modal volume with `create_if_missing=True` and downloads all models (via `huggingface_hub.snapshot_download`) and datasets (via `datasets.load_dataset`) into it.

2. **Copy-on-write isolation**: Each trial sandbox mounts the volume at `/hf-cache-volume` (read-only base layer). The `BASH_ENV` script (`setup-overlay.sh`) runs automatically on the first `bash -c` command and sets up a `fuse-overlayfs` overlay:
   - **Lower layer**: `/hf-cache-volume` (shared Modal volume, read-only)
   - **Upper layer**: `/tmp/hf-overlay/upper` (local to the sandbox)
   - **Merged at**: `/hf-home` (what the agent sees as `HF_HOME`)

   This means all trials can read from the shared ~500GB cache, but writes (new model checkpoints, additional downloads) are isolated per-trial and don't pollute the shared cache.

3. **Fallback**: If `fuse-overlayfs` is unavailable, the script falls back to `cp -as` (symlink tree).

### Populating the volume

The cache is populated automatically when you run `run_job.py` (unless `--no-ensure-cache` or `--no-hf-cache` is passed). The population is idempotent -- already-cached resources are skipped, so repeated runs are fast.

To populate the volume separately (recommended for the first time, since it can take hours):

```bash
# Download everything (14 models + 400+ datasets)
modal run src/harbor_adapter/modal_volume.py

# Models only (faster, ~50GB)
modal run src/harbor_adapter/modal_volume.py --models-only

# Datasets only
modal run src/harbor_adapter/modal_volume.py --datasets-only
```

### Volume name

The default volume name is `posttrainbench-hf-cache` (defined as `DEFAULT_VOLUME_NAME` in `modal_volume.py`). You can override it in `run_job.py` with `--hf-cache-volume <name>`, but you are responsible for populating a non-default volume yourself.

## Overlay Setup

The Docker image includes a `BASH_ENV` script (`setup-overlay.sh`) that runs automatically on the first `bash -c` command inside the container. It is idempotent (guarded by a flag file `/tmp/.overlay_done`) and performs two tasks:

1. **HF cache overlay** -- Mounts the fuse-overlayfs as described above (or falls back to symlinks). This runs before the agent's first command so its setup time doesn't count against the agent's timer.
2. **Timer sentinel** -- Creates `.timer_start` with the current Unix timestamp. This ensures `timer.sh` starts counting from the moment the agent begins, not from when the task was generated.

This approach works natively with `harbor run` and the Harbor registry -- no custom hooks or wrapper scripts are needed.

## API Key Requirements

| Key | Used By | Required For |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | Agent (Claude) | All benchmarks |
| `OPENAI_API_KEY` | Contamination judge (codex CLI), evaluation judge | All benchmarks (judge), arenahardwriting/healthbench (agent eval) |

- The verifier receives `OPENAI_API_KEY` as both `OPENAI_API_KEY` and `CODEX_API_KEY` (codex CLI reads `CODEX_API_KEY`).
- For arenahardwriting and healthbench, `OPENAI_API_KEY` is also passed to the agent environment since their `evaluate.py` scripts call the OpenAI API for judging.

## `run_job.py` Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--task-dir, -t` | (required\*) | Path to a single task directory |
| `--tasks-root` | (required\*) | Path to directory containing multiple tasks |
| `--agent, -a` | (required) | Agent name (e.g. `claude-code`, `codex`, `aider`) |
| `--model, -m` | (required) | Model name (e.g. `anthropic/claude-sonnet-4`) |
| `--hf-cache-volume` | `posttrainbench-hf-cache` | Modal volume name for the HF cache |
| `--no-hf-cache` | `false` | Don't mount a HF cache volume |
| `--no-ensure-cache` | `false` | Skip automatic cache population |
| `--modal-secret` | `[]` | Modal secret name (repeatable) |
| `--ae, --agent-env` | `[]` | Agent env var as `KEY=VALUE` (repeatable, supports `${VAR}`) |
| `--trials-dir` | `./trials` | Directory for trial results |
| `--n-concurrent` | `1` | Concurrent trials (with `--tasks-root`) |

\* `--task-dir` and `--tasks-root` are mutually exclusive; one is required.

## Task Structure

Each generated task follows Harbor's standard format:

```
posttrainbench-gsm8k-qwen3-1.7b/
├── task.toml              # Task configuration (GPU, timeout, env vars)
├── instruction.md         # Instructions for the agent
├── environment/
│   ├── Dockerfile         # Container definition (CUDA + vLLM + ML packages)
│   ├── .dockerignore      # Excludes Dockerfile and setup-overlay.sh from COPY
│   ├── setup-overlay.sh   # BASH_ENV script: overlay setup + timer sentinel
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── contamination_judge.py  # Generates judge prompt for codex CLI
│   ├── timer.sh           # Countdown timer (sentinel-file based)
│   ├── metadata.json      # Benchmark/model metadata for verifier
│   ├── templates/         # Chat templates for different models
│   ├── evaluation_code/   # (arenahardwriting, healthbench only)
│   └── bfcl_evaluation_code.py  # (bfcl only, from task_context)
└── tests/
    └── test.sh            # Verifier: contamination judge + 3-phase eval retry
```

## Evaluation Retry Logic

The verifier (`test.sh`) uses a 3-phase evaluation retry strategy matching `run_task.sh`:

| Phase | Max Attempts | Token Limits |
|-------|-------------|-------------|
| 1 | 4 | Default |
| 2 | 3 | Reduced (see below) |
| 3 | 2 | Further reduced (see below) |

Token limits per benchmark:

| Benchmark | Phase 2 | Phase 3 |
|-----------|---------|---------|
| aime2025 | `--max-tokens 12000` | `--max-tokens 8000` |
| arenahardwriting | `--max-new-tokens 12288` | `--max-new-tokens 8192` |
| bfcl | `--max-tokens 12000` | `--max-tokens 8000` |
| gpqamain | `--max-tokens 12000` | `--max-tokens 8000` |
| gsm8k | `--max-tokens 3000` | `--max-tokens 2000` |
| healthbench | `--max-new-tokens 12288` | `--max-new-tokens 8192` |
| humaneval | `--max-tokens 3000` | `--max-tokens 2000` |

GPU processes are killed between attempts to free VRAM.

## Contamination Judge

The contamination judge uses OpenAI's Codex CLI to analyze the agent's code:

```bash
codex --search -a never exec --json -c model_reasoning_summary=detailed \
    --skip-git-repo-check --yolo --model "gpt-5.1-codex" "$JUDGE_PROMPT"
```

It checks for:
- **Data contamination**: Using benchmark test data for training
- **Model violations**: Using a different model than the specified base model

Codex reads the workspace code and writes `contamination_judgement.txt` and `disallowed_model_judgement.txt` directly. The judge prompt is synced with `src/disallowed_usage_judge/prompt.txt`.

## Timer

The timer uses a sentinel-file approach: on the first `bash timer.sh` call, the current timestamp is recorded in `.timer_start`. This ensures the countdown is accurate even if the task is generated long before the agent starts.

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| GPU | 1x H100 | Configured in task.toml |
| Memory | 64 GB | |
| Storage | 100 GB | |
| Agent timeout | 10 hours | Adjustable via `--num-hours` |
| Verifier timeout | 3 hours | Accommodates 3-phase retry |
| Internet | Enabled | |

## Scoring

The verifier extracts the accuracy metric from `metrics.json` as the reward (0-1 scale). Results are stored in:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score
- `/logs/verifier/contamination_judgement.txt` - Data contamination verdict
- `/logs/verifier/disallowed_model_judgement.txt` - Model usage verdict
