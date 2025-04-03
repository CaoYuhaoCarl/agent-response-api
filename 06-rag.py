from openai import OpenAI
from rich import print

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="what's the preposition?",
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["vs_67da6ec7cf488191979f188450ade50b"] # replace with your vector store id from https://platform.openai.com/storage
    }]
)

print(response)

