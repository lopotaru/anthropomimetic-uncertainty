"Work in progress"

USED_MODEL = "gemma3:27b"
import pandas as pd
from datasets import load_dataset
from ollama import Client

from structured_output import PerceivedConfidenceOutput


def perceived_confidence(question_id: str, answer: str) -> float | None:
    new_prompt = f"""How reliable does the utternace seem, based on the phrasing of the utterance: "{answer}"?"""

    try:
        response = client.chat(
            USED_MODEL,
            messages=[{"role": "user", "content": new_prompt}],
            stream=False,
            format=PerceivedConfidenceOutput.model_json_schema(),
        )
        output = PerceivedConfidenceOutput.model_validate_json(
            response["message"]["content"]
        )

        return output.score

    except Exception as e:
        print(f"Could not measure perceived confidence for '{question_id}'. ")
        return None


def save_batch(batch: list[dict], output_file: str, first_batch: bool) -> bool:
    """Append a batch to the CSV. Returns False so the caller knows the header is written."""
    if not batch:
        return first_batch
    df = pd.DataFrame(batch)
    if first_batch:
        df.to_csv(output_file, mode="w", index=False)
    else:
        df.to_csv(output_file, mode="a", header=False, index=False)
    return False


if __name__ == "__main__":
    client = Client()
    dataset = load_dataset("lopotaru/anthropomimetic-uncertainty")
    results = []
    for row in dataset:
        question_id = row["question_id"]
        results.append(
            {
                "question_id": question_id,
                "initial_perceived_confidence": perceived_confidence(
                    question_id, row["initial_answer"]
                ),
                "rephrased_perceived_confidence": perceived_confidence(
                    question_id, row["rephrased_answer"]
                ),
            }
        )
