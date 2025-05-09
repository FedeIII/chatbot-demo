import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from vector_store import init_vector_store
import warnings
from langsmith import Client

# Load environment variables
load_dotenv()

# Configure LangSmith (if API key is available)
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY_BOE')
langsmith_project = os.getenv('LANGCHAIN_PROJECT_BOE', 'lawyer-ai-boe')
langsmith_tracing = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'

if langsmith_api_key and langsmith_tracing:
    os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
    os.environ['LANGCHAIN_PROJECT'] = langsmith_project
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    print(f"LangSmith tracing enabled for project: {langsmith_project}")
    
    # Test LangSmith connection
    try:
        client = Client()
        print("Successfully connected to LangSmith")
    except Exception as e:
        print(f"Warning: Could not connect to LangSmith: {e}")
        os.environ['LANGCHAIN_TRACING_V2'] = 'false'
else:
    # Disable tracing if no API key
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    print("LangSmith tracing is disabled. Set LANGCHAIN_API_KEY_BOE and LANGCHAIN_TRACING_V2=true to enable.")

# Suppress warnings
warnings.filterwarnings("ignore")

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.

Context: {context}

Question: {question}
"""

def create_rag_chain():
    """
    Create a RAG chain that combines a retriever with a language model
    
    Returns:
        chain: A LangChain chain that combines retrieval and generation
    """
    # Get the XAI API key from environment variables
    xai_api_key = os.getenv('XAI_API_KEY')
    if not xai_api_key:
        raise ValueError("XAI API key must be set")
    
    # Initialize the retriever from Pinecone
    retriever = init_vector_store()
    
    # Initialize the ChatXAI model
    model = ChatXAI(xai_api_key=xai_api_key, model="grok-3-mini-beta")
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    # Create the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    # Test the RAG chain
    try:
        chain = create_rag_chain()
        print("RAG chain created successfully.\n")
        
        # Test the chain with a sample query
        query = "What are some legal considerations for starting a business?"
        print(f"Query: {query}\n")
        
        response = chain.invoke(query)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error: {e}") 