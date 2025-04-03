from openai import OpenAI
import gradio as gr

client = OpenAI()

def ai_qa(question):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=question,
        tools=[{"type": "web_search_preview"}]
    )
    return response.output_text

demo = gr.Interface(
    fn=ai_qa,
    inputs="text",
    outputs="text",
    title="Carl's AI Q&A with Web Search",
    description="Ask anything you want"
)

if __name__ == "__main__":
    demo.launch()