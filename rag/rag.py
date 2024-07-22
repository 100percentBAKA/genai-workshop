import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain.schema import embeddings
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ! PRIVATE KEYS
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# ! CONSTANTS
PDF_PATH = "documents/data.pdf"
CHROMA_DB_PATH = "rag/db"

# ! FUNCTIONS
def create_llm():
    return ChatCohere(cohere_api_key=COHERE_API_KEY)


def create_embeddings():
    return CohereEmbeddings(model="embed-english-light-v3.0")


def chroma_vector_store(docx: list[Document], embedx, path: str):
    return Chroma.from_documents(docx, embedx, persist_directory=path)


if __name__ == "__main__":

    # ! CREATING AN LLM
    llm = create_llm()

    # ! CHOOSING EMBEDDINGS MODEL
    embeddings = create_embeddings()

    # ! LOADING THE DOCUMENTS
    loader = PyPDFLoader(PDF_PATH)

    # ! LOAD THE SPLITTER
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    # ? chunk_size=512

    # ! SPLIT THE DOCUMENT INTO CHUNKS
    docs_from_pdf = loader.load_and_split(text_splitter=splitter)

    # ! CHOOSING VECTOR STORE
    vector_store = chroma_vector_store(docs_from_pdf, embeddings, CHROMA_DB_PATH)

    # ! CREATING A RETRIEVER
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2})  # ? search_kwargs specifies the number of similar chunks/docs to return

    # ! CREATING A TEMPLATE
    prompt_template = """You are a knowledgeable assistant with access to information from various newspapers. 
    Your task is to provide accurate and relevant answers based on the news context provided. If any detail is 
    missing in the context, it's your responsibility to provide a suitable and informative response. When asked for 
    any web links, always ensure to provide the answers in markdown code format. Aim to be as descriptive as possible, 
    and provide insights that help the user understand the latest news trends and events.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:"""

    prompt_template = ChatPromptTemplate.from_template(prompt_template)

    # ! CREATING A CHAIN
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    # ! TAKING THE QUERY FROM THE USER
    query = input("Enter the query: ")

    # ! IMPLEMENTING A STREAM
    for chunk in chain.stream(query):
        print(chunk, end="", flush=True)