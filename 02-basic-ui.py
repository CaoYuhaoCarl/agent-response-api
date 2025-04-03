from openai import OpenAI
import gradio as gr

client = OpenAI()

def ai_qa(question):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=question
    )
    return response.output_text
    
demo = gr.Interface(
    fn=ai_qa,
    inputs="text",
    outputs="text",
    title="Carl's AI Q&A",
    description="Ask anything you want"
)

if __name__ == "__main__":
    demo.launch()