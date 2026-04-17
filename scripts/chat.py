#!/usr/bin/env python3
"""Chat CLI for a trial's uploaded final_model.

Point it at a trial dir. It reads verifier/final_model_hf.txt to find the
HF repo the verifier's hf_upload hook pushed, spins up a Modal container
that loads the model once, and streams a chat loop back to your terminal.

Usage:
    python scripts/chat.py jobs/full-10h/gsm8k-llama3.2-1b__Q9VCs4D

Env: HF_TOKEN (for private repos), Modal auth (`modal token new`).
"""

import argparse
import os
import re
import sys
from pathlib import Path

import modal

IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers>=4.45",
    "accelerate>=1.0",
    "huggingface_hub>=0.25",
)

app = modal.App("ptb-chat")

# Ship HF_TOKEN into the container if set locally; no-op otherwise (public repos).
_secret_kwargs = {"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}
_secrets = [modal.Secret.from_dict(_secret_kwargs)] if _secret_kwargs else []


@app.cls(
    image=IMAGE,
    gpu="A10G",
    timeout=3600,
    scaledown_window=300,
    secrets=_secrets,
)
class Chat:
    repo_id: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.repo_id}…", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @modal.method()
    def generate(self, messages: list[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def _resolve_repo_from_trial(trial: Path) -> str:
    hf_file = trial / "verifier" / "final_model_hf.txt"
    if not hf_file.exists():
        raise SystemExit(
            f"No HF upload record at {hf_file}. Either the verifier's hf_upload "
            "hook didn't run (older trial?) or it skipped (missing HF_TOKEN in the "
            "verifier env). Pass --repo to provide the repo id directly."
        )
    line = hf_file.read_text().strip()
    m = re.search(r"huggingface\.co/([\w.\-]+/[\w.\-]+)", line)
    if not m:
        raise SystemExit(f"Couldn't parse HF repo from: {line!r}")
    return m.group(1)


def chat_loop(chat: "Chat", system: str | None, max_tokens: int, temperature: float) -> None:
    history: list[dict] = []
    if system:
        history.append({"role": "system", "content": system})
    print("Ready. /exit to quit, /reset to clear history, /sys <msg> to set system.\n")
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print("(history cleared)\n")
            continue
        if user.startswith("/sys "):
            history = [{"role": "system", "content": user[len("/sys ") :]}]
            print("(system message set)\n")
            continue
        history.append({"role": "user", "content": user})
        reply = chat.generate.remote(history, max_tokens, temperature)
        print(reply + "\n")
        history.append({"role": "assistant", "content": reply})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trial", type=Path, nargs="?", help="Trial dir (reads verifier/final_model_hf.txt)")
    p.add_argument("--repo", type=str, default=None, help="Override: HF repo id directly")
    p.add_argument("--system", type=str, default=None, help="System message")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    if args.repo:
        repo = args.repo
    elif args.trial:
        repo = _resolve_repo_from_trial(args.trial)
    else:
        p.print_help()
        raise SystemExit("Pass a trial dir or --repo.")
    print(f"Connecting to Modal, repo={repo}")
    with modal.enable_output(), app.run():
        chat = Chat(repo_id=repo)
        chat_loop(chat, args.system, args.max_tokens, args.temperature)
    return 0


if __name__ == "__main__":
    sys.exit(main())
