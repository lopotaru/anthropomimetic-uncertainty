import pandas as pd
from datasets import load_dataset
from ollama import Client

from get_logprobs import generate_logprobs, get_confidence_from_logprobs
from prompts.system_prompt import SYSTEM_PROMPT

MAX_TOKENS = 50
USED_MODEL = "gemma3:1b"


def prompt_model_same_session(
    session_prompt: list[dict], max_tokens: int = MAX_TOKENS
) -> str:
    client = Client()
    response = client.chat(
        USED_MODEL,
        messages=session_prompt,
        stream=False,
        options={"num_predict": max_tokens},
        think=False,
    )
    return response["message"]["content"]


def measure_perceived_confidence(question: str, answer: str):
    client = Client()
    prompt = f""" Question: {question}
    Answer: {answer}

    Based on the used language, what confidence level does this answer express (0 - 100)?
    Respond with only a number."""

    response = client.chat(
        USED_MODEL, messages=[{"role": "user", "content": prompt}], stream=False
    )

    try:
        return float(response.get("response", "").strip())
    except:
        return 50.0


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

        initial_answer = prompt_model_same_session(session_messages, max_tokens)

        # get perceived confidence for initial answer
        initial_perceived_confidence = measure_perceived_confidence(
            question, initial_answer
        )

        # get logprobs
        logprobs_dictionary = generate_logprobs(initial_answer)
        logprobs_answer = logprobs_dictionary["logprobs_data"]

        # get confidence percentage
        confidence_answer = get_confidence_from_logprobs(logprobs_answer)

        ANTHROPOMIMETIC_PROMPT = f"""
        The model was {confidence_answer} confident in the answer. 
        Rephrase this answer in 1-2 sentences to reflect this confidence level 
        in natural language. Do not use the actual the actual percentage in 
        your answer."""

        session_messages.append({"role": "assistant", "content": initial_answer})

        # anthropomimetic prompting
        session_messages.append({"role": "user", "content": ANTHROPOMIMETIC_PROMPT})

        # ask to rephrase question based on confidence
        rephrased_answer = prompt_model_same_session(session_messages, max_tokens)

        # get perceived confidence for rephrased answer
        rephrased_perceived_confidence = measure_perceived_confidence(
            question, rephrased_answer
        )

        question_result = {
            "question_id": question_id,
            "question": question,
            "initial_answer": initial_answer,
            "initial_perceived_confidence": initial_perceived_confidence,
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
