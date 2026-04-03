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
| arenahardwriting | Arena-Hard-v2.0 (Writing) | 250 | Winrate vs baseline (OpenAI judge) |
| healthbench | HealthBench | 245 | Rubric-based accuracy (OpenAI judge) |

## Supported Models

### Tinker Mode (remote API, no GPU required)

| Key | Model ID | Type | GSM8K | HumanEval | AIME 2025 | GPQA | BFCL | Arena-Hard | HealthBench |
|-----|----------|------|-------|-----------|-----------|------|------|------------|-------------|
| llama3.1-8b | meta-llama/Llama-3.1-8B | Base | 3.3% | 22.0% | 0.0% | 17.2% | 78.3% | 1.7% | 19.3% |
| llama3.2-3b | meta-llama/Llama-3.2-3B | Base | 5.8% | 0.6% | 0.0% | 23.7% | 96.0% | 0.4% | 14.0% |
| llama3.2-1b | meta-llama/Llama-3.2-1B | Base | 3.7% | 0.0% | 0.0% | 11.4% | 22.5% | 1.4% | 5.8% |

#### Target reference scores (official instruct versions of the same models)

These are the scores Meta achieved with full post-training. They represent the upper bound of what's achievable.

| Model | GSM8K | HumanEval | AIME 2025 | GPQA | BFCL | Arena-Hard | HealthBench |
|-------|-------|-----------|-----------|------|------|------------|-------------|
| Llama-3.1-8B-Instruct | 84.5% | 72.6% | - | 30.4% | 76.1% | 18.3% | - |
| Llama-3.2-3B-Instruct | 77.7% | - | - | 32.8% | 67.0% | - | - |
| Llama-3.2-1B-Instruct | 44.4% | - | - | 27.2% | 25.7% | - | - |

Sources: Meta official model cards. BFCL: v1 for 8B, v2 for 3B/1B. Arena-Hard: v0.1 style-controlled winrate (LMSYS).

### GPU Mode (local vLLM, H100 required)

| Key | Model ID |
|-----|----------|
| qwen3-1.7b | Qwen/Qwen3-1.7B-Base |
| qwen3-4b | Qwen/Qwen3-4B-Base |
| smollm3-3b | HuggingFaceTB/SmolLM3-3B-Base |
| gemma3-4b | google/gemma-3-4b-pt |

## API Key Requirements

| Key | Used By | Required For |
|-----|---------|-------------|
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
