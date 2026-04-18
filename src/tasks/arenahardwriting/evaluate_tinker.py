#!/usr/bin/env python3
"""Evaluate a Tinker checkpoint (or base model) on Arena-Hard-v2.0 (Writing).

Usage:
    python evaluate.py --checkpoint "tinker://<run_id>/sampler_weights/final"
    python evaluate.py --base-model "Qwen/Qwen3-1.7B-Base"
"""

# IMPORTANT: You are NOT allowed to use the OpenAI API for anything but this evaluation script.
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, sys.path[0])
sys.path.insert(0, sys.path[0] + "/../..")
import shortuuid
import tiktoken
from evaluation_code.show_result import print_leaderboard
from evaluation_code.utils.add_markdown_info import count_markdown_elements, remove_pattern
from evaluation_code.utils.completion import (
    load_model_answers,
    load_questions,
    make_config,
)
from evaluation_code.utils.judge_utils import JUDGE_SETTINGS
from tinker import types
from tqdm import tqdm

from tinker_util import setup_tinker

API_MAX_RETRY = 3
API_RETRY_SLEEP = 5
DEFAULT_JUDGE_WORKERS = 64
MAX_REPETITIONS = 5

BENCHMARK = "arena-hard-v2.0"
JUDGE_MODEL = "gpt-5-mini"
REASONING_EFFORT = "medium"
JUDGE_CONFIG = "evaluation_code/config/arena-hard-v2.0.yaml"
JUDGE_MAX_COMPLETION = 49152
DATA_PATH = Path("evaluation_code/data/" + BENCHMARK)
MAX_TOKENS = 16384


# --- Text processing (copied from GPU version) ---


def limit_repetitions(text: str, max_reps: int = MAX_REPETITIONS) -> str:
    def _limit_consecutive_lines(txt):
        lines = txt.split("\n")
        result, i = [], 0
        modified = False
        while i < len(lines):
            line = lines[i]
            count, j = 1, i + 1
            while j < len(lines) and lines[j] == line:
                count += 1
                j += 1
            if count > max_reps:
                result.extend([line] * max_reps)
                modified = True
            else:
                result.extend([line] * count)
            i = j
        return "\n".join(result), modified

    def _limit_regex_patterns(txt):
        modified = False
        while True:
            changed = False
            for min_len, max_len in [(3, 30), (30, 60), (60, 120), (120, 250), (250, 500)]:
                pattern = rf"(.{{{min_len},{max_len}}}?)\1{{{max_reps},}}"

                def replace_func(match):
                    nonlocal changed, modified
                    unit = match.group(1)
                    full = match.group(0)
                    if len(full) // len(unit) > max_reps:
                        modified = True
                        changed = True
                        return unit * max_reps
                    return full

                txt = re.sub(pattern, replace_func, txt, flags=re.DOTALL)
            if not changed:
                break
        return txt, modified

    for _ in range(10):
        text, m1 = _limit_consecutive_lines(text)
        text, m2 = _limit_regex_patterns(text)
        if not m1 and not m2:
            break
    return text


# --- Metadata ---


def _make_metadata(answer: str) -> dict:
    encoding = tiktoken.encoding_for_model("gpt-4o")
    token_len = len(encoding.encode(answer, disallowed_special=()))
    metadata = {"token_len": token_len}
    markdown_info = count_markdown_elements(
        remove_pattern(answer, re.compile("```([^`]*)```")),
        suffix="",
    )
    metadata.update(markdown_info)
    return metadata


# --- Tinker generation ---


def generate_answers_tinker(ctx, questions, model_alias) -> dict[str, dict]:
    """Generate answers using Tinker sampling."""
    sampling_params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stop=ctx.renderer.get_stop_sequences(),
    )

    print(f"[generate] Generating answers for {len(questions)} questions via Tinker")

    # Fire all requests concurrently
    futures = []
    for question in questions:
        messages = [{"role": "user", "content": question["prompt"]}]
        prompt = ctx.renderer.build_generation_prompt(messages)
        futures.append(ctx.sampling_client.sample(prompt=prompt, sampling_params=sampling_params, num_samples=1))

    print(f"[generate] Fired {len(futures)} requests...")

    answers_dict: dict[str, dict] = {}
    for i, (future, question) in enumerate(zip(futures, questions)):
        try:
            result = future.result()
            parsed, _ = ctx.renderer.parse_response(result.sequences[0].tokens)
            answer_text = parsed["content"]

            # Strip thinking tags
            if answer_text.startswith("<think>") and "</think>" in answer_text:
                answer_text = answer_text.split("</think>", maxsplit=1)[1].strip()

            answer_text = limit_repetitions(answer_text)

            messages = [
                {"role": "user", "content": question["prompt"]},
                {"role": "assistant", "content": {"answer": answer_text}},
            ]

            record = {
                "uid": question["uid"],
                "ans_id": shortuuid.uuid(),
                "model": model_alias,
                "messages": messages,
                "tstamp": time.time(),
                "metadata": _make_metadata(answer_text),
            }
            answers_dict[question["uid"]] = record
        except Exception as e:
            print(f"[generate] Question {question['uid']}: ERROR - {e}", file=sys.stderr)

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            print(f"[generate] Progress: {i + 1}/{len(questions)}", flush=True)

    return answers_dict


# --- Judge (same as GPU version) ---


def call_openai(messages: list[dict]):
    import openai

    client = openai.OpenAI()
    request_kwargs = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "max_completion_tokens": JUDGE_MAX_COMPLETION,
    }
    if REASONING_EFFORT is not None:
        request_kwargs["reasoning_effort"] = REASONING_EFFORT

    for attempt in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(**request_kwargs)
            return {"answer": completion.choices[0].message.content}
        except openai.BadRequestError as err:
            if "reasoning" in str(err).lower() and "reasoning_effort" in request_kwargs:
                print("[judge] reasoning_effort not supported; retrying without it.")
                request_kwargs.pop("reasoning_effort", None)
                continue
            wait_time = API_RETRY_SLEEP * (2**attempt)
            print(f"[judge] OpenAI API error ({type(err).__name__}): {err}. Retry in {wait_time}s.")
            time.sleep(wait_time)
        except Exception as err:
            wait_time = API_RETRY_SLEEP * (2**attempt)
            print(f"[judge] OpenAI API error ({type(err).__name__}): {err}. Retry in {wait_time}s.")
            time.sleep(wait_time)
    print("[judge] Exhausted retries; returning None.")
    return None


def get_score(judgment: str, patterns: Iterable[str]) -> str | None:
    for pattern in patterns:
        compiled = re.compile(pattern)
        matches = [m for m in compiled.findall(judgment.upper()) if isinstance(m, str) and m]
        if matches:
            return matches[-1].strip("\n")
        if matches and isinstance(matches[-1], tuple):
            for item in matches[-1]:
                if item:
                    return item.strip("\n")
    return None


def _judge_single_question(question, model_alias, model_answers, prompt_template, regex_patterns):
    uid = question["uid"]
    category = question["category"]
    baseline_model = JUDGE_SETTINGS[category]["baseline"]

    if baseline_model not in model_answers:
        raise FileNotFoundError(f"Baseline model '{baseline_model}' answers not found")

    candidate_answer = model_answers[model_alias].get(uid)
    baseline_answer = model_answers[baseline_model].get(uid)

    if candidate_answer is None or baseline_answer is None:
        return None

    prompt_args = {
        "QUESTION": question["prompt"],
        "ANSWER_A": baseline_answer["messages"][-1]["content"]["answer"],
        "ANSWER_B": candidate_answer["messages"][-1]["content"]["answer"],
    }
    user_prompt = prompt_template.format(**prompt_args)
    messages = [
        {"role": "system", "content": JUDGE_SETTINGS[category]["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]

    result_ab = call_openai(messages)
    score_ab = get_score(result_ab["answer"], regex_patterns) if result_ab else None

    prompt_args_swap = {
        "QUESTION": question["prompt"],
        "ANSWER_A": candidate_answer["messages"][-1]["content"]["answer"],
        "ANSWER_B": baseline_answer["messages"][-1]["content"]["answer"],
    }
    user_prompt_swap = prompt_template.format(**prompt_args_swap)
    messages_swap = [
        {"role": "system", "content": JUDGE_SETTINGS[category]["system_prompt"]},
        {"role": "user", "content": user_prompt_swap},
    ]

    result_ba = call_openai(messages_swap)
    score_ba = get_score(result_ba["answer"], regex_patterns) if result_ba else None

    return {
        "uid": uid,
        "category": category,
        "judge": JUDGE_MODEL,
        "model": candidate_answer["model"],
        "baseline": baseline_answer["model"],
        "games": [
            {"score": score_ab, "judgment": result_ab, "prompt": messages},
            {"score": score_ba, "judgment": result_ba, "prompt": messages_swap},
        ],
    }


def judge_answers(questions, model_alias, candidate_answers, judge_workers=DEFAULT_JUDGE_WORKERS):
    judge_config = make_config(JUDGE_CONFIG)
    prompt_template = judge_config["prompt_template"]
    regex_patterns = judge_config["regex_patterns"]

    answer_dir = DATA_PATH / "model_answer"
    model_answers = load_model_answers(str(answer_dir))
    model_answers[model_alias] = candidate_answers

    if "OPENAI_API_KEY" not in os.environ:
        raise OSError("OPENAI_API_KEY is not set.")

    results: list[dict | None] = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        futures = {
            executor.submit(
                _judge_single_question,
                question,
                model_alias,
                model_answers,
                prompt_template,
                regex_patterns,
            ): idx
            for idx, question in enumerate(questions)
        }
        with tqdm(total=len(futures), desc="Judging answers") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pbar.update(1)
                    raise
                pbar.update(1)

    return results


# --- Scoring (same as GPU version) ---


def _judgments_to_battles(judgments, weight=3):
    import pandas as pd

    score_map = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }

    battles_data = []
    for record in judgments:
        if record is None:
            continue
        games = record.get("games", [])
        if len(games) < 2:
            continue
        score_ab = games[0].get("score")
        score_ba = games[1].get("score")
        if score_ab is None or score_ba is None:
            continue
        scores_ab = score_map[score_ab.upper()]
        scores_ba = score_map[score_ba.upper()]
        scores = [1 - s for s in scores_ab] + scores_ba
        battles_data.append(
            {
                "uid": record["uid"],
                "category": record["category"],
                "model": record["model"],
                "baseline": record["baseline"],
                "scores": scores,
            }
        )

    battles = pd.DataFrame(battles_data)
    if battles.empty:
        return battles
    return battles.explode("scores").reset_index(drop=True)


def _compute_metrics(battles):
    scores = battles["scores"].astype(float)
    num_samples = len(scores)
    if num_samples == 0:
        return {"accuracy": 0.0, "stderr": 0.0}
    accuracy = float(scores.mean())
    if num_samples == 1:
        stderr = 0.0
    else:
        std_dev = float(scores.std(ddof=1))
        stderr = std_dev / math.sqrt(num_samples) if not math.isnan(std_dev) else 0.0
    return {"accuracy": accuracy, "stderr": stderr}


def summarize_results(model_alias, judgments):
    battles = _judgments_to_battles(judgments)
    battles = battles[battles.model == model_alias].reset_index(drop=True)
    if battles.empty:
        return {"accuracy": 0.0, "stderr": 0.0}
    categories = battles.category.unique().tolist()
    for category in categories:
        print_leaderboard(battles[battles.category == category].reset_index(drop=True), category)
    return _compute_metrics(battles)


# --- Main ---


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run Arena-Hard evaluation with Tinker.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--json-output-file", type=str, default=None)
    parser.add_argument("--limit", type=int, default=-1, help="Sample cap for dev iteration. -1 = full set (default).")
    parser.add_argument("--judge-workers", type=int, default=DEFAULT_JUDGE_WORKERS)
    return parser.parse_args()


def main():
    args = parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise OSError("OPENAI_API_KEY is not set.")

    ctx = setup_tinker(args)
    model_alias = ctx.model_name.split("/")[-1]

    # Load questions
    questions = load_questions(str(DATA_PATH / "question.jsonl"))
    if args.limit is not None and args.limit != -1:
        random.Random(42).shuffle(questions)
        questions = questions[: args.limit]
    elif args.limit == -1:
        random.Random(42).shuffle(questions)

    print(f"[data] Loaded {len(questions)} questions")

    # Generate answers via Tinker
    candidate_answers = generate_answers_tinker(ctx, questions, model_alias)
    print(f"[generate] Generated {len(candidate_answers)} answers")

    # Judge
    judgments = judge_answers(questions, model_alias, candidate_answers, args.judge_workers)
    print("[done] Judgments complete")

    # Score
    metrics = summarize_results(model_alias, judgments)
    print(f"\nScore (winrate) is: {metrics['accuracy']}")

    if args.json_output_file is not None:
        with open(args.json_output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[done] Metrics saved to {args.json_output_file}")


if __name__ == "__main__":
    main()
