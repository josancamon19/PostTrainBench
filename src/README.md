# PostTrainBench Harbor Adapter

This adapter generates [Harbor](https://harborframework.com)-compatible tasks for running PostTrainBench evaluations.

## Modes

The adapter supports two modes:

| Mode | Flag | Compute | Training | Environment |
|------|------|---------|----------|-------------|
| **default** | `mode=default` | H100 GPU via Modal | Raw GPU — agent writes training scripts | `modal` |
| **prime-rl** | `mode=prime-rl` | CPU-only via Daytona | Hosted RL via Prime Intellect — agent writes TOML configs | `daytona` |

## Template Structure

Templates use a **base + overlay** pattern. Shared files live in `base_template/`, and mode-specific files in `harbor_template/` or `prime_rl_template/`. Mode-specific files override base files when both exist.

```
src/
  base_template/                  # Shared across all modes
    environment/
      .dockerignore
      contamination_judge.py
      contamination_judge_prompt.txt
      templates/*.jinja            # Chat templates (qwen3, gemma3, smollm)
    tests/
      test.sh                      # Verifier: contamination judge + eval retry
      verifier.py
  harbor_template/                 # default mode overrides
    environment/
      Dockerfile                   # CUDA + vLLM + full ML stack
    instruction.md                 # Raw GPU training instructions
    task.toml                      # 1x H100, 64GB RAM
  prime_rl_template/               # prime-rl mode overrides
    environment/
      Dockerfile                   # CPU-only, prime CLI + SDK
    instruction.md                 # prime-rl workflow instructions
    task.toml                      # No GPU, 8GB RAM, PRIME_API_KEY
```

## Supported Benchmarks

### Default mode (all 7 benchmarks)

| Benchmark ID | Name | Type |
|-------------|------|------|
| gsm8k | GSM8K (Grade School Math 8K) | inspect-ai |
| humaneval | HumanEval | inspect-ai |
| aime2025 | AIME 2025 | inspect-ai |
| gpqamain | GPQA | inspect-ai |
| bfcl | Berkeley Function Calling Leaderboard | inspect-ai |
| arenahardwriting | Arena-Hard-v2.0 (Writing) | vLLM + OpenAI judge |
| healthbench | HealthBench | vLLM + OpenAI judge |

### Prime-rl mode (2 benchmarks with existing prime-rl environments)

| Benchmark ID | Name |
|-------------|------|
| gsm8k | GSM8K (Grade School Math 8K) |
| aime2025 | AIME 2025 |

## Supported Models

### Default mode

| Key | HuggingFace Model ID |
|-----|---------------------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

### Prime-rl mode (models available on Prime Intellect Lab)

| Key | HuggingFace Model ID |
|-----|---------------------|
| qwen3-4b | Qwen/Qwen3-4B-Instruct-2507 |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B |
| llama3.2-3b | meta-llama/Llama-3.2-3B-Instruct |

## Quick Start

### Default mode (raw GPU training)

```bash
# Generate tasks
uv run python src/adapter.py benchmark=gsm8k model=qwen3-1.7b

# Set API keys
export ANTHROPIC_API_KEY=<your-key>
export OPENAI_API_KEY=<your-key>

# Run
harbor run --path ./datasets/posttrainbench/gsm8k-qwen3-1.7b \
    --agent claude-code --model anthropic/claude-sonnet-4 --env modal
```

### Prime-rl mode (hosted RL training)

```bash
# Generate tasks
uv run python src/adapter.py mode=prime-rl benchmark=gsm8k model=qwen3-4b

# Set API keys
export OPENAI_API_KEY=<your-key>
export PRIME_API_KEY=<your-key>

# Run
harbor run --path ./datasets/posttrainbench/prime-rl-gsm8k-qwen3-4b \
    --agent codex --model openai/gpt-5.1-codex --env daytona
```

### CLI reference

```bash
# List available benchmarks and models for a mode
uv run python src/adapter.py mode=prime-rl list=true

# Generate all tasks for a mode
uv run python src/adapter.py mode=prime-rl all=true

# Custom output dir and time limit
uv run python src/adapter.py mode=prime-rl benchmark=gsm8k model=qwen3-4b \
    output=./my_tasks num_hours=0.5
```

## API Key Requirements

| Key | Default mode | Prime-rl mode |
|-----|-------------|---------------|
| `ANTHROPIC_API_KEY` | Agent (Claude) | Agent (Claude), if using claude-code |
| `OPENAI_API_KEY` | Contamination judge, arenahardwriting/healthbench eval | Contamination judge, agent (if using codex) |
| `PRIME_API_KEY` | — | Agent (prime CLI for hosted training) |

## Scoring

The verifier extracts the accuracy metric from `metrics.json` as the reward (0-1 scale). Results are stored in:
- `/logs/verifier/metrics.json` - Full evaluation metrics
- `/logs/verifier/reward.txt` - Accuracy score
- `/logs/verifier/contamination_judgement.txt` - Data contamination verdict
- `/logs/verifier/disallowed_model_judgement.txt` - Model usage verdict
