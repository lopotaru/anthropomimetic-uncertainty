from ollama import Client

client = Client()

messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?",
    },
]

response = client.chat("smollm2:135m", messages=messages, stream=False)
print(response["message"]["content"])

"""
for part in client.chat("smollm2:135m", messages=messages, stream=False):
    print(part.message.content, end="", flush=True)
"""
