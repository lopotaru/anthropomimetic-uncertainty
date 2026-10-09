import pandas as pd
from datasets import load_dataset
from ollama import Client

from prompts.system_prompt import SYSTEM_PROMPT
from src.get_logprobs import (
    get_answer_and_logprobs,
    get_confidence_from_logprobs,
)
from src.structured_output import RephrasedOutput

MAX_TOKENS = 512
USED_MODEL = "gemma3:27b"
# "smollm:135m"
# "gemma3:1b"


def measure_perceived_confidence(question: str, answer: str):
    client = Client()
    prompt = f""" 
    How confident is the phrasing of the utterance below. Give your answer as 
    a percentage (0-100%). Utterance: {answer}
    """

    try:
        response = client.chat(
            USED_MODEL, messages=[{"role": "user", "content": prompt}], stream=False
        )

        return response["message"]["content"].strip()

    except Exception as e:
        print(f"Could not measure perceived confidence for '{question}'. ")
        return None


def get_rephrased_answer(
    initial_answer: str, confidence: float, max_tokens: int = MAX_TOKENS
) -> RephrasedOutput | None:
    client = Client()

    ANTHROPOMIMETIC_PROMPT = f"""Rephrase this utterance {initial_answer} to reflect the confidence level of 
            {confidence} in natural language."""

    try:
        client = Client()
        response = client.chat(
            USED_MODEL,
            messages=ANTHROPOMIMETIC_PROMPT,
            stream=False,
            format=RephrasedOutput.model_json_schema(),
            options={"num_predict": max_tokens},
        )

        rephrased_answer = RephrasedOutput.model_validate_json(
            response["message"]["content"]
        )
        return rephrased_answer

    except Exception as e:
        print(f"Error rephrasing answer for utterance '{initial_answer}'. ")
        return None


def send_prompt(
    batch: list[dict],
    output_file: str,
    first_batch: bool = False,
    max_tokens: int = MAX_TOKENS,
):

    results = []

    for question_line in batch:
        question_id = question_line["question_id"]
        question = question_line["question"]

        # initiate session message
        session_messages = []

        session_messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # ask question
        session_messages.append(
            {"role": "user", "content": f"Answer the following question: {question}"}
        )

        result = get_answer_and_logprobs(question, session_messages)

        initial_answer = result["answer"]

        # get perceived confidence for initial answer
        initial_perceived_confidence = measure_perceived_confidence(
            question, initial_answer
        )

        # get confidence percentage
        confidence_answer = get_confidence_from_logprobs(result["logprobs"])

        rephrased_answer = get_rephrased_answer(
            initial_answer, confidence_answer, MAX_TOKENS
        )

        # get perceived confidence for rephrased answer
        rephrased_perceived_confidence = measure_perceived_confidence(
            question, rephrased_answer.answer
        )

        question_result = {
            "question_id": question_id,
            "question": question,
            "initial_answer": initial_answer,
            "initial_perceived_confidence": initial_perceived_confidence,
            "actual_confidence": confidence_answer,
            "rephrased_answer": rephrased_answer,
            "rephrased_perceived_confidence": rephrased_perceived_confidence,
        }

        results.append(question_result)

    # save batch results
    if output_file:
        df = pd.DataFrame(results)
        if first_batch:
            df.to_csv(output_file, mode="w", index=False)
        else:
            df.to_csv(output_file, mode="a", header=False, index=False)

    return results


def get_and_send_prompts_in_batches(
    name_dataset: str,
    config: str,
    split: str,
    batch_size: int,
    output_file: str,
    max_tokens: int = MAX_TOKENS,
) -> list[dict]:
    question_dictionary = {}
    dataset = load_dataset(name_dataset, config)
    split_data = dataset[split]

    batch = []
    batch_number = 0

    for _, question in enumerate(split_data):
        question_dictionary = {
            "question_id": question["question_id"],
            "question": question["question"],
        }
        batch.append(question_dictionary)

        if len(batch) == batch_size:
            if batch_number == 0:
                send_prompt(batch, output_file, True, max_tokens)
            else:
                send_prompt(batch, output_file, False, max_tokens)

            batch = []
            batch_number += 1

    if batch:
        if batch_number == 0:
            send_prompt(batch, output_file, True)
        else:
            send_prompt(batch, output_file, False)


if __name__ == "__main__":
    get_and_send_prompts_in_batches(
        "mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", "train", 10, "results.csv"
    )
