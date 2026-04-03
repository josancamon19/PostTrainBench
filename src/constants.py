"""PostTrainBench constants: benchmarks, models, and scoring baselines."""

from dataclasses import dataclass


@dataclass
class BenchmarkInfo:
    task_id: str
    benchmark_name: str
    setup_note: str = ""


@dataclass
class ModelInfo:
    model_id: str
    short_name: str


# setup_note: GPU-mode specific notes from original posttrainbench (inspect-ai warnings).
# Included in instruction.md via {setup_other} placeholder. Not relevant for tinker mode.
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

# GPU models - same as Tinker for apples-to-apples comparison
MODELS = {
    # "qwen3-1.7b": ModelInfo(model_id="Qwen/Qwen3-1.7B-Base", short_name="qwen3-1.7b"),
    # "qwen3-4b": ModelInfo(model_id="Qwen/Qwen3-4B-Base", short_name="qwen3-4b"),
    # "smollm3-3b": ModelInfo(model_id="HuggingFaceTB/SmolLM3-3B-Base", short_name="smollm3-3b"),
    # "gemma3-4b": ModelInfo(model_id="google/gemma-3-4b-pt", short_name="gemma3-4b"),
    "llama3.1-8b": ModelInfo(model_id="meta-llama/Llama-3.1-8B", short_name="llama3.1-8b"),
    "llama3.2-3b": ModelInfo(model_id="meta-llama/Llama-3.2-3B", short_name="llama3.2-3b"),
    "llama3.2-1b": ModelInfo(model_id="meta-llama/Llama-3.2-1B", short_name="llama3.2-1b"),
}

TINKER_MODELS = {
    "llama3.1-8b": ModelInfo(
        model_id="meta-llama/Llama-3.1-8B", short_name="llama3.1-8b"
    ),
    "llama3.2-3b": ModelInfo(
        model_id="meta-llama/Llama-3.2-3B", short_name="llama3.2-3b"
    ),
    "llama3.2-1b": ModelInfo(
        model_id="meta-llama/Llama-3.2-1B", short_name="llama3.2-1b"
    ),
}

# Official instruct model scores (from Meta model cards) for target-setting.
# Keys: (model_id, benchmark_id) -> score as float (0-1 scale).
# Only includes pairs with official published numbers.
INSTRUCT_BASELINES: dict[tuple[str, str], float] = {
    # Llama-3.1-8B-Instruct
    ("meta-llama/Llama-3.1-8B", "gsm8k"): 0.845,
    ("meta-llama/Llama-3.1-8B", "humaneval"): 0.726,
    ("meta-llama/Llama-3.1-8B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.1-8B", "gpqamain"): 0.304,
    ("meta-llama/Llama-3.1-8B", "bfcl"): 0.761,
    ("meta-llama/Llama-3.1-8B", "arenahardwriting"): 0.507,
    ("meta-llama/Llama-3.1-8B", "healthbench"): 0.234,
    # Llama-3.2-3B-Instruct
    ("meta-llama/Llama-3.2-3B", "gsm8k"): 0.777,
    ("meta-llama/Llama-3.2-3B", "humaneval"): 0.506,
    ("meta-llama/Llama-3.2-3B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.2-3B", "gpqamain"): 0.328,
    ("meta-llama/Llama-3.2-3B", "bfcl"): 0.670,
    ("meta-llama/Llama-3.2-3B", "arenahardwriting"): 0.433,
    ("meta-llama/Llama-3.2-3B", "healthbench"): 0.256,
    # Llama-3.2-1B-Instruct
    ("meta-llama/Llama-3.2-1B", "gsm8k"): 0.444,
    ("meta-llama/Llama-3.2-1B", "humaneval"): 0.348,
    ("meta-llama/Llama-3.2-1B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.2-1B", "gpqamain"): 0.272,
    ("meta-llama/Llama-3.2-1B", "bfcl"): 0.257,
    ("meta-llama/Llama-3.2-1B", "arenahardwriting"): 0.200,
    ("meta-llama/Llama-3.2-1B", "healthbench"): 0.129,
}

# Base model scores (measured via Tinker evaluation).
BASE_SCORES: dict[tuple[str, str], float] = {
    # Llama-3.1-8B
    ("meta-llama/Llama-3.1-8B", "gsm8k"): 0.033,
    ("meta-llama/Llama-3.1-8B", "humaneval"): 0.220,
    ("meta-llama/Llama-3.1-8B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.1-8B", "gpqamain"): 0.172,
    ("meta-llama/Llama-3.1-8B", "bfcl"): 0.783,
    ("meta-llama/Llama-3.1-8B", "arenahardwriting"): 0.017,
    ("meta-llama/Llama-3.1-8B", "healthbench"): 0.193,
    # Llama-3.2-3B
    ("meta-llama/Llama-3.2-3B", "gsm8k"): 0.058,
    ("meta-llama/Llama-3.2-3B", "humaneval"): 0.006,
    ("meta-llama/Llama-3.2-3B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.2-3B", "gpqamain"): 0.237,
    ("meta-llama/Llama-3.2-3B", "bfcl"): 0.960,
    ("meta-llama/Llama-3.2-3B", "arenahardwriting"): 0.004,
    ("meta-llama/Llama-3.2-3B", "healthbench"): 0.140,
    # Llama-3.2-1B
    ("meta-llama/Llama-3.2-1B", "gsm8k"): 0.037,
    ("meta-llama/Llama-3.2-1B", "humaneval"): 0.0,
    ("meta-llama/Llama-3.2-1B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.2-1B", "gpqamain"): 0.114,
    ("meta-llama/Llama-3.2-1B", "bfcl"): 0.225,
    ("meta-llama/Llama-3.2-1B", "arenahardwriting"): 0.014,
    ("meta-llama/Llama-3.2-1B", "healthbench"): 0.058,
}

# Base model scores (measured via GPU/vLLM evaluation, temp=0.6 from model generation_config).
# For comparison with Tinker scores (temp=0.0) above.
BASE_SCORES_GPU: dict[tuple[str, str], float] = {
    # Llama-3.1-8B
    ("meta-llama/Llama-3.1-8B", "gsm8k"): 0.041,
    ("meta-llama/Llama-3.1-8B", "humaneval"): 0.091,
    ("meta-llama/Llama-3.1-8B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.1-8B", "gpqamain"): 0.062,
    ("meta-llama/Llama-3.1-8B", "bfcl"): 0.288,
    ("meta-llama/Llama-3.1-8B", "arenahardwriting"): 0.0,
    ("meta-llama/Llama-3.1-8B", "healthbench"): 0.196,
    # Llama-3.2-3B
    ("meta-llama/Llama-3.2-3B", "gsm8k"): 0.040,
    ("meta-llama/Llama-3.2-3B", "humaneval"): 0.213,
    ("meta-llama/Llama-3.2-3B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.2-3B", "gpqamain"): 0.004,
    ("meta-llama/Llama-3.2-3B", "bfcl"): 0.238,
    ("meta-llama/Llama-3.2-3B", "arenahardwriting"): 0.001,
    ("meta-llama/Llama-3.2-3B", "healthbench"): 0.149,
    # Llama-3.2-1B
    ("meta-llama/Llama-3.2-1B", "gsm8k"): 0.021,
    ("meta-llama/Llama-3.2-1B", "humaneval"): 0.085,
    ("meta-llama/Llama-3.2-1B", "aime2025"): 0.0,
    ("meta-llama/Llama-3.2-1B", "gpqamain"): 0.0,
    ("meta-llama/Llama-3.2-1B", "bfcl"): 0.273,
    ("meta-llama/Llama-3.2-1B", "arenahardwriting"): 0.0,
    ("meta-llama/Llama-3.2-1B", "healthbench"): 0.059,
}
