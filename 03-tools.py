from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Give me 5 GTC 2025 news from today.",
    tools=[{"type": "web_search_preview"}]
    # tool_choice="auto"
)

print(response.output_text)
