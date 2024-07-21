import os
from dotenv import load_dotenv
import gradio as gr
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

#! 1) LOAD THE MODEL
def load_model(model: str, temp: float, max_tokens: int):
    if model == "Cohere Command":
        return ChatCohere(
            cohere_api_key=COHERE_API_KEY,
            max_tokens=max_tokens,
            temperature=temp
        )
    else:
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            max_tokens=max_tokens,
            temperature=temp
        )

#! 2) CONSTRUCT THE PROMPT
template = """
    You are a sentiment analysis AI tool, you should accurately analyse given description and predict the underlying sentiment
    
    Human: Following is the description: {description}
"""

prompt_template = ChatPromptTemplate.from_template(template)

def submit(model, temperature, max_tokens, input_text):
    llm = load_model(model, temperature, max_tokens)
    
    #! 3) SETUP THE CHAIN
    chain = prompt_template | llm | StrOutputParser()
    
    #! 4) INVOKE THE CHAIN
    result = chain.invoke(
        {
            "description": input_text
        }
    )
    return result

def demo():
    with gr.Blocks() as demo:
        with gr.Tabs() as tabs:
            
            with gr.Tab("Model Selection"):
                with gr.Row():
                    model = gr.Dropdown(choices=["Cohere Command", "Gemini Pro"], label="Select Model")
                with gr.Row():
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature")
                with gr.Row():
                    max_tokens = gr.Slider(minimum=50, maximum=2048, value=150, step=10, label="Max Tokens")
                    
            with gr.Tab("Sentiment Analysis"):
                with gr.Row():
                    input_text = gr.Textbox(lines=5, placeholder="Enter your text here...", label="Input")
                with gr.Row():
                    output_text = gr.Textbox(lines=5, placeholder="Output will appear here...", label="Output")
                with gr.Row():
                    submit_button = gr.Button("Submit")
                    submit_button.click(submit, inputs=[model, temperature, max_tokens, input_text], outputs=output_text)
                    
    demo.launch(share=True)

if __name__ == "__main__":
    demo()
