"""PostTrainBench constants: benchmarks, models, and scoring baselines."""

from dataclasses import dataclass


@dataclass
class BenchmarkInfo:
    task_id: str
    benchmark_name: str
    setup_note: str = ""
    long_generation: bool = False  # Uses ~16k output tokens (affects max_connections)


@dataclass
class ModelInfo:
    model_id: str
    short_name: str
    instruct_id: str = ""  # HF model ID of the instruct variant (for baseline oracle)
    tinker: bool = True  # Whether available on Tinker (remote API mode)
    # Max concurrent vLLM requests on H100 80GB (scaled by model size + KV cache)
    max_connections: int = 64  # for short-generation benchmarks (~2k tokens)
    max_connections_long: int = 8  # for long-generation benchmarks (~16k tokens)


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
    "arenahardwriting": BenchmarkInfo(
        task_id="arenahardwriting",
        benchmark_name="Arena-Hard-v2.0 (Writing)",
        setup_note="",
        long_generation=True,
    ),
    "healthbench": BenchmarkInfo(
        task_id="healthbench",
        benchmark_name="HealthBench",
        setup_note="",
        long_generation=True,
    ),
}

MODELS = {
    # commented due to base not being available on tinker
    # "qwen3-1.7b": ModelInfo(model_id="Qwen/Qwen3-1.7B-Base", short_name="qwen3-1.7b", tinker=False),
    # "qwen3-4b": ModelInfo(model_id="Qwen/Qwen3-4B-Base", short_name="qwen3-4b", tinker=False),
    # "smollm3-3b": ModelInfo(model_id="HuggingFaceTB/SmolLM3-3B-Base", short_name="smollm3-3b", tinker=False),
    # "gemma3-4b": ModelInfo(model_id="google/gemma-3-4b-pt", short_name="gemma3-4b", tinker=False),
    "llama3.1-8b": ModelInfo(
        model_id="meta-llama/Llama-3.1-8B",
        short_name="llama3.1-8b",
        instruct_id="meta-llama/Llama-3.1-8B-Instruct",
        max_connections=64,
        max_connections_long=8,
    ),
    "llama3.2-3b": ModelInfo(
        model_id="meta-llama/Llama-3.2-3B",
        short_name="llama3.2-3b",
        instruct_id="meta-llama/Llama-3.2-3B-Instruct",
        max_connections=128,
        max_connections_long=16,
    ),
    "llama3.2-1b": ModelInfo(
        model_id="meta-llama/Llama-3.2-1B",
        short_name="llama3.2-1b",
        instruct_id="meta-llama/Llama-3.2-1B-Instruct",
        max_connections=256,
        max_connections_long=32,
    ),
    "qwen3-8b": ModelInfo(
        model_id="Qwen/Qwen3-8B-Base",
        short_name="qwen3-8b",
        instruct_id="Qwen/Qwen3-8B",
        max_connections=64,
        max_connections_long=8,
    ),
    "qwen3-30b-a3b": ModelInfo(
        model_id="Qwen/Qwen3-30B-A3B-Base",
        short_name="qwen3-30b-a3b",
        instruct_id="Qwen/Qwen3-30B-A3B",
        max_connections=16,
        max_connections_long=4,
    ),
}

# Baseline scores per model. Base measured via Tinker (temp=0). Target = instruct model scores.
# Llama instruct: GPU oracle + Meta model cards. Qwen instruct: Tinker (temp=0, /no_think for gsm8k).
# fmt: off
SCORES: dict[str, dict] = {
    "meta-llama/Llama-3.1-8B": {
        "instruct_id": "meta-llama/Llama-3.1-8B-Instruct",
        "benchmarks": {
            #                    base    target
            "gsm8k":           (0.033,  0.845),
            "humaneval":       (0.213,  0.659),
            # target = 1/30, real instruct score is 0.0, kinda an impossible task.
            "aime2025":        (0.0,    0.033),
            "gpqamain":        (0.181,  0.304),
            "arenahardwriting":(0.017,  0.467),
            "healthbench":     (0.214,  0.234),
        },
    },
    "meta-llama/Llama-3.2-3B": {
        "instruct_id": "meta-llama/Llama-3.2-3B-Instruct",
        "benchmarks": {
            "gsm8k":           (0.061,  0.777), # ✅
            "humaneval":       (0.006,  0.524), # ✅
            # "aime2025":        (0.0,    0.0),
            "gpqamain":        (0.252,  0.328), # ✅
            "arenahardwriting":(0.005,  0.433), # ✅
            "healthbench":     (0.134,  0.256), # ✅
        },
    },
    "meta-llama/Llama-3.2-1B": {
        "instruct_id": "meta-llama/Llama-3.2-1B-Instruct",
        "benchmarks": {
            "gsm8k":           (0.036,  0.466), # ✅
            "humaneval":       (0.0,    0.354), # ✅
            # "aime2025":        (0.0,    0.0),
            "gpqamain":        (0.132,  0.225), # ✅
            "arenahardwriting":(0.0,    0.200), # ✅
            "healthbench":     (0.054,  0.139), # ✅
        },
    },
    "Qwen/Qwen3-8B-Base": {
        "instruct_id": "Qwen/Qwen3-8B",
        "benchmarks": {
            # "gsm8k":           (0.913,  0.875),  # base > target (instruct /no_think)
            "humaneval":       (0.024,  0.457),
            "aime2025":        (0.167,  0.400),
            "gpqamain":        (0.388,  0.534),
            "arenahardwriting":(0.341,  0.820),
            "healthbench":     (0.287,  0.542),
        },
    },
    "Qwen/Qwen3-30B-A3B-Base": {
        "instruct_id": "Qwen/Qwen3-30B-A3B",
        "benchmarks": {
            # "gsm8k":           (0.908,  0.895),  # base > target (instruct /no_think)
            "humaneval":       (0.006,  0.720),
            "aime2025":        (0.100,  0.500),
            "gpqamain":        (0.462,  0.580),
            "arenahardwriting":(0.475,  0.860),
            "healthbench":     (0.301,  0.580),
        },
    },
}
# fmt: on

# Backward-compatible flat dicts (used by adapter.py and verifier)
BASE_SCORES: dict[tuple[str, str], float] = {
    (model_id, bid): scores[0] for model_id, data in SCORES.items() for bid, scores in data["benchmarks"].items()
}

INSTRUCT_BASELINES: dict[tuple[str, str], float] = {
    (model_id, bid): scores[1] for model_id, data in SCORES.items() for bid, scores in data["benchmarks"].items()
}

# Regression suite: evals the verifier runs against final_model to detect catastrophic
# forgetting and domain generalization. Not training targets. Two layers:
#   Layer A (breadth, orthogonal to the STEM targets): mmlu, ifeval, truthfulqa
#   Layer B (cross-target check on existing cheap benchmarks): gsm8k, humaneval, gpqamain
# Long-gen OpenAI-graded benchmarks (arenahardwriting, healthbench) are excluded due to
# cost + latency per trial.
REGRESSION_EVALS: list[str] = [
    "mmlu",
    "ifeval",
    "truthfulqa",
    "gsm8k",
    "humaneval",
    "gpqamain",
]

# Baselines for Layer A regression evals, measured via Tinker (0-shot chat,
# temp=0, full test/validation sets: MMLU=14,042, TruthfulQA MC1=817 samples).
# Used for forgetting_penalty computation.
#
# Caveat: Llama base models under-perform at 0-shot chat format (they weren't
# trained on it), so the MMLU numbers here are noticeably lower than Meta's
# reported 8-shot completion-style baselines. These numbers are what matters
# for THIS benchmark, because the verifier evaluates trained models in the
# same 0-shot chat format — apples-to-apples.
#
# Llama-3.2-1B on TruthfulQA @ 0.938 still looks suspicious at full 817
# samples — the 1B model evidently outputs a fixed letter that correlates
# with TruthfulQA's truthful-answer positions. Shuffling choice positions
# per sample would correct this; left as follow-up. For now the high
# baseline just means forgetting is easy to flag on that row — a trained
# model that actually answers the questions will score lower, and that
# drop would show up as forgetting penalty even though semantically the
# post-training is *improving* the model. Handle with care when reading
# leaderboard for this model/benchmark combo.
#
# IFEval needs google/instruction_following_eval (not resolvable on 3.13);
# will be measured on GPU in a follow-up pass. Entry omitted here so the
# regression suite skips it in the forgetting calculation instead of
# counting as 0.
#
# Layer B (gsm8k, humaneval, gpqamain) baselines reuse BASE_SCORES — not
# duplicated here.
# fmt: off
REGRESSION_BASE_SCORES: dict[tuple[str, str], float] = {
    ("Qwen/Qwen3-30B-A3B-Base", "mmlu"):       0.771,
    ("Qwen/Qwen3-30B-A3B-Base", "truthfulqa"): 0.606,
    ("Qwen/Qwen3-8B-Base",      "mmlu"):       0.723,
    ("Qwen/Qwen3-8B-Base",      "truthfulqa"): 0.752,
    ("meta-llama/Llama-3.1-8B", "mmlu"):       0.107,
    ("meta-llama/Llama-3.1-8B", "truthfulqa"): 0.494,
    ("meta-llama/Llama-3.2-3B", "mmlu"):       0.219,
    ("meta-llama/Llama-3.2-3B", "truthfulqa"): 0.632,
    ("meta-llama/Llama-3.2-1B", "mmlu"):       0.199,
    ("meta-llama/Llama-3.2-1B", "truthfulqa"): 0.938,  # position-bias artifact (see note)
}
# fmt: on
