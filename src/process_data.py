import pandas as pd
from datasets import load_dataset

dataset = load_dataset("manu/trivia_qa_wiki")


def get_prompt(name_dataset: str, split: str) -> list[dict]:
    question_dictionary = {}
    prompting_questions = []
    dataset = load_dataset(name_dataset)
    split_data = dataset[split]

    for question in split_data:
        question_dictionary = {
            "question_id": question["question_id"],
            "question": question["question"],
        }
        prompting_questions.append(question_dictionary)

    return prompting_questions


def save_prompt(output_file: str, prompts: list[dict]):

    df = pd.DataFrame(prompts)
    df.to_csv(output_file, index=False)

    return df


prompt_questions = get_prompt(dataset, "train")
saved_data = save_prompt("data/questions", prompt_questions)
