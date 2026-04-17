"""IFEval scoring (adapted from tinkering2/ifbench/simple_eval.py, re-pointed
at Google's original IFEval registry).

Exposes `evaluate_output(response, instruction_id_list, kwargs, prompt)` which
returns prompt-level and instruction-level strict/loose scores per IFEval's
standard methodology.
"""

from __future__ import annotations

from dataclasses import dataclass

from .instructions_registry import INSTRUCTION_DICT


def get_loose_transformations(response: str) -> list[str]:
    """Response variants for loose eval (strip formatting artifacts + markdown)."""
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    no_asterisk = response.replace("*", "")
    return [
        response,
        no_asterisk,
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def strip_thinking(content: str) -> str:
    s = content.strip()
    if not s:
        return s
    if "</think>" in s:
        return s.split("</think>", 1)[-1].strip()
    if s.startswith("<think>"):
        return ""
    return s


@dataclass
class InstructionResult:
    instruction_id: str
    strict_pass: bool
    loose_pass: bool


def evaluate_output(
    response: str,
    instruction_id_list: list[str],
    kwargs: list[dict],
    prompt: str,
) -> tuple[list[InstructionResult], dict]:
    results: list[InstructionResult] = []
    variants = get_loose_transformations(response)

    for idx, iid in enumerate(instruction_id_list):
        if iid not in INSTRUCTION_DICT:
            results.append(InstructionResult(iid, False, False))
            continue
        checker = INSTRUCTION_DICT[iid](iid)
        inst_kwargs = kwargs[idx] if idx < len(kwargs) else {}
        filtered = {k: v for k, v in inst_kwargs.items() if v is not None}
        checker.build_description(**filtered)

        args = checker.get_instruction_args()
        if args and "prompt" in args:
            checker.build_description(prompt=prompt)

        strict = bool(response.strip() and checker.check_following(response))
        loose = any(v.strip() and checker.check_following(v) for v in variants)
        results.append(InstructionResult(iid, strict, loose))

    n = len(results) or 1
    strict_correct = sum(r.strict_pass for r in results)
    loose_correct = sum(r.loose_pass for r in results)
    return results, {
        "prompt_strict": int(strict_correct == len(results)),
        "prompt_loose": int(loose_correct == len(results)),
        "instruction_strict": strict_correct / n,
        "instruction_loose": loose_correct / n,
    }
