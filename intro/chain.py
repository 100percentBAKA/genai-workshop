import os 
from dotenv import load_dotenv 
from langchain_cohere import ChatCohere
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# from langchain_cohere.llms import Cohere
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

#! 1) LOAD THE MODEL
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    max_tokens=1000,
    temperature=0.1
)

#! 2) PREPARE THE PROMPT TEMPLATE
template = """
    You are a {role}
    
    Human: {query}
"""

prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke(
#     {
#         "role": "programming assist",
#         "query": "how do I print hello world in bash"
#     }
# )

#! 3) STRUCTURE THE CHAIN
chain = prompt_template | llm | StrOutputParser()

#! 4) INVOKE THE MODEL ON THE CHAIN
response = chain.invoke(
    {
        "role": "programming assist",
        "query": "how do I print hello world in bash"
    }
)

#! 5) PRINT THE RESPONSE 
print(response)
