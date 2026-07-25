from datasets import load_dataset

dataset = load_dataset("manu/trivia_qa_wiki")


def get_prompt(name_dataset: str, split: str) -> list[str]:
    prompting_questions = []
    dataset = load_dataset(name_dataset)
    split_data = dataset[split]

    for question in split_data:
        prompt = question["question"]
        prompting_questions.append(prompt)

    return prompting_questions
