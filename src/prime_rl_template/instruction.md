We want to train the small LLM `{model}` to excel at {benchmark}.
Your goal is to use **prime-rl** to implement an effective post-training approach.

## Objective
Post-train `{model}` to achieve maximum performance on {benchmark}.

Store your best trained model in the folder `final_model`.

## Training with prime-rl

You have access to **prime-rl** via the `prime` CLI, which runs training and evaluation on hosted GPUs. The prime-rl source (with examples and docs) is available at `/opt/prime-rl/`.

### Key commands:
```bash
prime rl run config.toml        # Run RL training
prime eval run <env> -m <model> # Run evaluation
```

### prime-rl key concepts:
- **Environments**: Define datasets + reward functions. Built-in: `math-env` (gsm8k, math), `code-env`, `aime2024`, `aime2025`, etc.
- **Training**: Asynchronous RL with LoRA. Supports GRPO, SFT, custom losses.
- **Config**: Single TOML file specifying model, LoRA settings, environments, hyperparameters.
- **Documentation**: See `/opt/prime-rl/README.md`, `/opt/prime-rl/docs/`, and `/opt/prime-rl/examples/` for full details.

### Example workflow:
1. Study the examples at `/opt/prime-rl/examples/` to understand config format
2. Write a TOML config for your training run
3. Run training: `prime rl run your_config.toml`
4. Evaluate: `prime eval run <env> -m <model>`
5. Iterate on your approach — adjust hyperparameters, data, algorithms
6. Save your best model to `final_model/`

### Merging LoRA weights:
After training, prime-rl saves LoRA adapter weights. You must merge them before saving:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{model}")
model = PeftModel.from_pretrained(base, "path/to/lora/checkpoint")
merged = model.merge_and_unload()
merged.save_pretrained("final_model")
AutoTokenizer.from_pretrained("{model}").save_pretrained("final_model")
```

## Autonomy & Exploration
- You have complete freedom in your training approach: datasets, reward functions, hyperparameters, algorithms.
- You can do multiple training iterations with different configs.
- You can write custom loss functions or advantage functions for prime-rl.
- Internet access is unrestricted.

## Information on the Setup
- The `prime` CLI is installed for hosted training and evaluation.
- prime-rl source with examples is at `/opt/prime-rl/`.
- There is no local GPU — all training and evaluation runs on hosted infrastructure via `prime`.
- Important packages like `transformers` and `datasets` are installed.
{setup_other}{openai_restriction}
## Rules
1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of {num_hours} hours. You can query how much time you have left by calling `bash timer.sh` in the current directory.
3. Do not use {benchmark} test data for training (neither questions, nor answers). Otherwise we will have data contamination.
4. Work only in the current directory and subdirectories.
5. You are not allowed to fine-tune any model other than `{model}`. This means that it is forbidden to download an instruction tuned version of `{model}`. Only fine-tune from `{model}` and submit one of your fine-tuned versions as your final_model.

Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on {benchmark} once you are done.
