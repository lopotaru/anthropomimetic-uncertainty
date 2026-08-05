from typing import Iterable

import ollama
from ollama import Client

"""
def print_logprobs(logprobs: Iterable[dict], label: str) -> None:
    print(f"\n{label}:")
    for entry in logprobs:
        token = entry.get("token", "")
        logprob = entry.get("logprob")
        print(f"  token={token!r:<12} logprob={logprob:.3f}")
        for alt in entry.get("top_logprobs", []):
            if alt["token"] != token:
                print(f"    alt -> {alt['token']!r:<12} ({alt['logprob']:.3f})")
"""


def generate_logprobs(answer: str, question_id: str) -> dict[list]:

    client = Client()

    response = ollama.generate(
        model="gemma3",
        prompt=answer,
        logprobs=True,
        top_logprobs=3,
    )

    logprobs = response.get("logprobs", [])

    if logprobs:
        avg_logprob = sum(entry.get("logprob", 0) for entry in logprobs) / len(logprobs)
        total_logprob = sum(entry.get("logprob", 0) for entry in logprobs)
    else:
        avg_logprob = 0.0
        total_logprob = 0.0

    return {
        "logprobs_data": logprobs,
        "avg_logprob": avg_logprob,
        "total_logprob": total_logprob,
        "num_tokens": len(logprobs),
    }
