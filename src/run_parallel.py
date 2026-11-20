"Work in progress"

import asyncio

from datasets import load_dataset
from ollama import AsyncClient

from structured_output import PerceivedConfidenceOutput

dataset = load_dataset("lopotaru/anthropomimetic-uncertainty")

USED_MODEL = "gemma3:27b"
HOSTS = [f"http://127.0.0.1:{11434 + i}" for i in range(4)]
OUTPUT_FILE = "perceived_confidence_v2.csv"

clients = [AsyncClient(host=h) for h in HOSTS]


async def perceived_confidence(client: AsyncClient, answer: str) -> float | None:

    new_prompt = f"""How reliable does the utternace seem, based on the phrasing of the utterance: "{answer}"?"""
    try:
        response = await client.chat(
            model=USED_MODEL,
            messages=[{"role": "user", "content": new_prompt}],
            format=PerceivedConfidenceOutput.model_json_schema(),
            stream=False,
        )
        output = PerceivedConfidenceOutput.model_validate_json(
            response["message"]["content"]
        )
        return output.score
    except Exception as e:
        print(f"Failed for answer '{answer}': {e}")
        return None
