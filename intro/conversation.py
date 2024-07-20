import os 
from dotenv import load_dotenv 
from langchain_cohere import ChatCohere
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

#! 1) LOAD THE MODEL
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    max_tokens=1000,
    temperature=0.1
)

#! 2) STRUCTURE THE MESSAGES
messages = [
    SystemMessage(content="You are my personal assistant, help me come up with some adjectives for the names provided by the user"),
    HumanMessage(content="Cherry"),
    AIMessage(content="Charming Cherry"),
    HumanMessage(content="My name is Adarsh")
]

# batch_messages = [
#     [
#         SystemMessage(content="You are my personal assistant, help me come up with some adjectives for the names provided by the user"),
#         HumanMessage(content="My name is Adarsh")
#     ],
#     [
#         SystemMessage(content="You are my personal assistant, help me come up with some adjectives for the names provided by the user"),
#         HumanMessage(content="My name is Rahul")
#     ]
# ]

response = llm.invoke(messages)
# response = llm.generate(batch_messages)

#! 3) PRINT THE RESPONSE 
print(response)