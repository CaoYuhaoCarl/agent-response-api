from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o-mini",
    input=[{
        "role": "user",
        "content": "Write a story about a Nezha."
    },
    ],
    stream=True,
)

for event in stream:
    if hasattr(event, "delta"):
        print(event.delta, end="")
    elif hasattr(event, "text") and event.type == 'response.output_text.done':
        pass
