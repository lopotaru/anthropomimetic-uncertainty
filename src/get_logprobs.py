from typing import Iterable

import numpy as np
import ollama
import pandas as pd
from ollama import Client

USED_MODEL = "smollm:135m"
# "gemma3:1b"
MAX_TOKENS = 50


def calculate_average_logprobs(response):
    logprobs = response.get("logprobs", [])
    if logprobs:
        return np.mean([logprob.get("logprob", -np.inf) for logprob in logprobs])
    else:
        return -np.inf


def calculate_total_logprobs(response):
    logprobs = response.get("logprobs", [])
    if logprobs:
        return np.sum([logprob.get("logprob", 0) for logprob in logprobs])
    else:
        return 0.0


def generate_logprobs(answer: str) -> dict:

    client = Client()

    response = ollama.generate(
        model=USED_MODEL,
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

    logprobs_results = {
        "logprobs_data": logprobs,
        "avg_logprob": avg_logprob,
        "total_logprob": total_logprob,
        "num_tokens": len(logprobs),
    }

    return logprobs_results


def get_logprobs_in_batches(batch: list, first_batch: bool, output_file: str):

    batch_results = []

    for question_line in batch:
        question_id = question_line["question_id"]

        initial_answer_logprobs = generate_logprobs(question_line["initial_answer"])

        rephrased_answer_logprobs = generate_logprobs(question_line["rephrased_answer"])

        question_results = dict(question_line)
        question_results.update(
            {
                "question_id": question_id,
                "initial_avg_logprob": initial_answer_logprobs["avg_logprob"],
                "initial_total_logprob": initial_answer_logprobs["total_logprob"],
                "initial_num_tokens": initial_answer_logprobs["num_tokens"],
                "rephrased_avg_logprob": rephrased_answer_logprobs["avg_logprob"],
                "rephrased_logprob": rephrased_answer_logprobs["total_logprob"],
                "rephrased_num_tokens": rephrased_answer_logprobs["num_tokens"],
            }
        )

        batch_results.append(question_results)

    if batch_results:
        data = pd.DataFrame(batch_results)
    if first_batch:
        data.to_csv(output_file, mode="w", index=False)
    else:
        data.to_csv(output_file, mode="a", header=False, index=False)


def get_answer_and_logprobs(question: str, session_message: list[dict]) -> dict:
    client = Client()

    if session_message:
        messages = session_message
    else:
        messages = [
            {"role": "user", "content": f"Answer the following question: {question}"}
        ]

    response = client.chat(
        USED_MODEL,
        messages=messages,
        stream=False,
        logprobs=True,
        top_logprobs=3,
        options={"num_predict": MAX_TOKENS},
    )

    return {
        "answer": response.get("response", ""),
        "logprobs": response.get("logprobs", []),
        "avg_logprobs": calculate_average_logprobs(response),
        "total_logprobs": calculate_total_logprobs(response),
        "num_tokens": len(response.get("logprobs", [])),
    }


def save_logprobs(input_file: str, output_file: str, batch_size: int):
    input_data = pd.read_csv(input_file)
    batch = []
    current_batch_number = 0

    input_data_dictionary = input_data.to_dict("records")

    for question_row in input_data_dictionary:
        batch.append(question_row)

        if len(batch) == batch_size:
            if current_batch_number == 0:
                get_logprobs_in_batches(batch, True, output_file)
            else:
                get_logprobs_in_batches(batch, False, output_file)
            current_batch_number += 1

    if batch:
        if current_batch_number == 0:
            get_logprobs_in_batches(batch, True, output_file)
        else:
            get_logprobs_in_batches(batch, False, output_file)


def get_confidence_from_logprobs(logprobs: list[dict]) -> float:
    logprobs_array = np.array([logprob.get("logprob", -np.inf) for logprob in logprobs])
    scaled_logporbs = np.clip(logprobs_array, -100, 0)
    probabilities = np.exp(scaled_logporbs)
    averaged_probabilities = np.mean(probabilities)
    confidence = averaged_probabilities * 100

    return min(confidence, 100.00)


if __name__ == "__main__":
    save_logprobs("results.csv", "logprobs_results.csv", 10)
