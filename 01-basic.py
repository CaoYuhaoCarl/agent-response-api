from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Make a list of the 10 largest cities in the world."
)

print(response.output_text)