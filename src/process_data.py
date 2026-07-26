import pandas as pd
from datasets import load_dataset
from ollama import Client

from prompts.anthropomimetic_prompt import ANTHROPOMIMETIC_PROMPT


def prompt_model_same_session(prompt: str) -> str:
    client = Client()

    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = client.chat("smollm2:135m", messages=messages, stream=False)
    return response["message"]["content"]


def send_prompt(batch: list[dict], output_file: str, first_batch: bool = False):

    results = []

    for question_line in batch:
        question_id = question_line["question_id"]
        question = question_line["question"]

        # initiate session message
        session_messages = []

        # ask question
        session_messages.append(
            {"role": "user", "content": f"Answer the following question: {question}"}
        )

        initial_answer = prompt_model_same_session(session_messages)

        # anthropomimetic prompting
        session_messages.append({"role": "user", "content": ANTHROPOMIMETIC_PROMPT})

        rephrased_answer = prompt_model_same_session(session_messages)

        question_result = {
            "question_id": question_id,
            "question": question,
            "initial_answer": initial_answer,
            "rephrased_answer": rephrased_answer,
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
    name_dataset: str, config: str, split: str, batch_size: int
) -> list[dict]:
    question_dictionary = {}
    dataset = load_dataset(name_dataset, config)
    split_data = dataset[split]

    batch = []

    for _, question in enumerate(split_data):
        question_dictionary = {
            "question_id": question["question_id"],
            "question": question["question"],
        }
        batch.append(question_dictionary)

        if len(batch) == batch_size:
            send_prompt(batch)


def save_prompt(output_file: str, prompts: list[dict]):

    df = pd.DataFrame(prompts)
    df.to_csv(output_file, index=False)

    return df
