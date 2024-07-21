import os 
from dotenv import load_dotenv
import gradio as gr
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

#! 1) INTRODUCTION
# def analyze_sentiment(text, description):
#     prompt = prompt_template.format(description=description)
#     result = chain.invoke(prompt)
#     return result

# demo = gr.Interface(
#     fn=analyze_sentiment,
#     inputs=[gr.Textbox(lines=2, placeholder="Enter text here..."), gr.TextArea(placeholder="Enter some description here...")],
#     outputs=[gr.Textbox()],
#     title="Sentiment Analysis",
#     description="Enter text to analyze sentiment (positive / negative)."
# )

# demo.launch(share=True)

#! 2) LANGCHAIN INTEGRATION

#! 2.1) LOAD THE MODEL
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    max_tokens=1000,
    temperature=0.1
)

#! 2.2) CONSTRUCT THE PROMPT
template = """
    You are a sentiment analysis AI tool, you should accurately analyse given description and predict the underlying sentiment
    
    Human: Following is the description: {description}
"""

prompt_template = ChatPromptTemplate.from_template(template)

#! 2.3) SETUP THE CHAIN
chain = prompt_template | llm | StrOutputParser()

#! 2.4) INTEGRATE WITH GRADIO
def analyze_sentiment(description):
    result = chain.invoke(
        {
            "description": f"{description}"
        }
    )
    return result

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=[gr.TextArea(placeholder="Enter some description here...")],
    outputs=[gr.Textbox()],
    title="Sentiment Analysis",
    description="Enter text to analyze sentiment (positive / negative)."
)

demo.launch(share=True)
