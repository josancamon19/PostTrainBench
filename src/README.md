# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations. Supports two modes: **GPU** (local vLLM) and **Tinker** (remote API).

## Supported Benchmarks

| Benchmark ID | Name | Samples | Metric |
|-------------|------|---------|--------|
| gsm8k | GSM8K (Grade School Math 8K) | 1,319 | Accuracy (exact match) |
| humaneval | HumanEval | 164 | pass@1 (code execution) |
| aime2025 | AIME 2025 | 30 | Accuracy (exact match) |
| gpqamain | GPQA Main | 448 | Accuracy (multiple choice) |
| bfcl | Berkeley Function Calling Leaderboard | 400 | Accuracy (function name match) |
| arenahardwriting | Arena-Hard-v2.0 (Writing) | - | OpenAI judge (GPU mode only) |
| healthbench | HealthBench | - | OpenAI judge (GPU mode only) |

## Supported Models

### Tinker Mode (remote API, no GPU required)

| Key | Model ID | Type | GSM8K | HumanEval | AIME 2025 | GPQA | BFCL |
|-----|----------|------|-------|-----------|-----------|------|------|
| llama3.1-8b | meta-llama/Llama-3.1-8B | Base | 3.3% | 22.0% | 0.0% | 17.2% | 78.3% |
| llama3.2-3b | meta-llama/Llama-3.2-3B | Base | 5.8% | 0.6% | 0.0% | 23.7% | 96.0% |
| llama3.2-1b | meta-llama/Llama-3.2-1B | Base | 3.7% | 0.0% | 0.0% | 11.4% | 22.5% |

#### Reference scores (instruct/trained versions, for context)

| Model | GSM8K | HumanEval | AIME 2025 | GPQA | BFCL |
|-------|-------|-----------|-----------|------|------|
| Qwen3-4B-Instruct-2507 | 82.5% | 83.5% | 16.7% | 46.2% | - |
| Qwen3-8B (Hybrid) | - | - | 36.7% | - | - |
| Llama-3.3-70B-Instruct | - | - | 10.0% | - | 100% |
| Qwen3.5-397B-A17B | - | - | 43.3% | - | - |

### GPU Mode (local vLLM, H100 required)

| Key | Model ID |
|-----|----------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

## Quick Start

### 1. Install

```bash
uv sync
```

### 2. Generate tasks

```bash
# Tinker mode (no GPU needed)
python src/adapter.py mode=tinker all=true

# Single task
python src/adapter.py mode=tinker benchmark=gsm8k model=llama3.2-1b

# GPU mode
python src/adapter.py mode=gpu all=true

# List available options
python src/adapter.py list=true mode=tinker
```

### 3. Set API keys

```bash
python -m modal setup                # Modal cloud setup
export ANTHROPIC_API_KEY=<your-key>  # For Claude agent
export TINKER_API_KEY=<your-key>     # For Tinker mode
export OPENAI_API_KEY=<your-key>     # For contamination judge + arenahardwriting/healthbench
```

### 4. Run with Harbor

```bash
# Run all GSM8K Tinker tasks
harbor run -c job.yaml

# Run a single task
harbor run \
    -p datasets/posttrainbench/gsm8k-llama3.2-1b-tinker \
    -a claude-code \
    -m anthropic/claude-sonnet-4 \
    --env modal

# Test with oracle (dummy solution, verifies pipeline)
harbor run -c job.yaml -a oracle -p datasets/posttrainbench -t "gsm8k-llama3.2-1b-tinker"
```

## Task Structure

### Tinker Mode

```
gsm8k-llama3.2-1b-tinker/
├── task.toml              # Task config (no GPU, timeouts, env vars)
├── instruction.md         # Agent instructions
├── solution/
│   └── solve.sh           # Oracle dummy solution (saves base weights)
├── environment/
│   ├── Dockerfile         # Python + tinker + tinker-cookbook + datasets
│   ├── evaluate.py        # Benchmark evaluation script
│   ├── tinker_util.py     # Shared Tinker evaluation utilities
│   ├── timer.sh           # Countdown timer (synced with agent timeout)
│   └── metadata.json      # Benchmark/model metadata
└── tests/
    ├── test.sh            # Verifier: evaluate checkpoint + contamination judge
    ├── evaluate.py        # Pristine copy of evaluation script
    ├── tinker_util.py     # Shared utilities for verifier
    ├── contamination_judge.py
    └── contamination_judge_prompt.txt
```

### GPU Mode

```
gsm8k-qwen3-1.7b/
├── task.toml              # Task config (1x H100, timeouts, env vars)
├── instruction.md         # Agent instructions
├── environment/
│   ├── Dockerfile         # CUDA + vLLM + ML packages
│   ├── evaluate.py        # Benchmark evaluation (inspect-ai + vLLM)
│   ├── timer.sh           # Countdown timer
│   ├── metadata.json      # Benchmark/model metadata
│   └── templates/         # Chat templates per model
└── tests/
    ├── test.sh            # Verifier: contamination judge + 3-phase eval retry
    ├── evaluate.py        # Pristine copy
    ├── templates/         # Chat templates
    ├── contamination_judge.py
    └── contamination_judge_prompt.txt
```

## API Key Requirements

| Key | Used By | Required For |
|-----|---------|-------------|
| `ANTHROPIC_API_KEY` | Agent (Claude) | All benchmarks |
| `TINKER_API_KEY` | Tinker API (training + inference) | Tinker mode |
| `OPENAI_API_KEY` | Contamination judge, evaluation judge | All benchmarks (judge), arenahardwriting/healthbench (agent eval) |

## Contamination Judge

The verifier runs a contamination judge (in `tests/`, not visible to the agent) using OpenAI's Codex CLI to check for:
- **Data contamination**: Using benchmark test data for training
- **Model violations**: Using a different model than the specified base model

## Scoring

The verifier extracts accuracy from `metrics.json` as the reward (0-1 scale). Results:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score
- `/logs/verifier/contamination_judgement.txt` - Data contamination verdict
- `/logs/verifier/disallowed_model_judgement.txt` - Model usage verdict
