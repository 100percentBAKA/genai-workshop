import os 
import uvicorn
from dotenv import load_dotenv 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_cohere import ChatCohere
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler

# from langchain_cohere.llms import Cohere
# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

#! CREATE AN INSTANCE OF FASTAPI AND CONFIG MIDDLEWARES
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#! FOLLOWING CODE DEFINES THE REQUEST MODEL
class QueryRequest(BaseModel):
    query: str


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

#! 3) STRUCTURE THE CHAIN
chain = prompt_template | llm | StrOutputParser()

config = {
    'callbacks': [StdOutCallbackHandler()]
}

#! 4) POST ENDPOINT
@app.post("/query")
async def handle_query(request: QueryRequest):
    
    try:
        #! 5) STREAM THE MODEL RESPONSE ON THE CHAIN
        response = chain.invoke(
            {
                "role": "programming assist",
                "query": request.query ## query is str
            },
            config=config
        )
        
        #! 6) INSTEAD OF PRINTING, WE ARE RETURNING THE RESPONSE HERE
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, details=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)