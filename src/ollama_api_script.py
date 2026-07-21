from ollama import Client

client = Client()


def prompt_model(prompt: str):
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = client.chat("smollm2:135m", messages=messages, stream=False)
    return response["message"]["content"]


"""
# This works for stream=True
for part in client.chat("smollm2:135m", messages=messages, stream=True):
    print(part.message.content, end="", flush=True)
"""
