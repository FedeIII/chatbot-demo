import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI

# Load environment variables from .env file
load_dotenv()

# Get the API keys and environment variables
xai_api_key = os.getenv('XAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_index = os.getenv('PINECONE_INDEX_BOE')
langchain_project = os.getenv('LANGCHAIN_PROJECT_BOE')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY_BOE')
langchain_tracing = os.getenv('LANGCHAIN_TRACING_V2')

# Validate environment variables
if not xai_api_key:
    raise ValueError("Please set the XAI_API_KEY environment variable in your .env file.")
if not pinecone_api_key:
    raise ValueError("Please set the PINECONE_API_KEY environment variable in your .env file.")
if not pinecone_index:
    raise ValueError("Please set the PINECONE_INDEX_BOE environment variable in your .env file.")
if not langchain_api_key:
    raise ValueError("Please set the LANGCHAIN_API_KEY_BOE environment variable in your .env file.")
if not langchain_project:
    raise ValueError("Please set the LANGCHAIN_PROJECT_BOE environment variable in your .env file.")

# Set up basic model (will be integrated with RAG in next steps)
chat = ChatXAI(xai_api_key=xai_api_key, model="grok-3-mini-beta")

# Temporary basic example to test environment setup
if __name__ == "__main__":
    print("Environment variables loaded successfully.")
    print(f"Pinecone Index: {pinecone_index}")
    print(f"LangChain Project: {langchain_project}")
    print(f"LangChain Tracing: {langchain_tracing}")
    
    try:
        response = chat.invoke("Tell me fun things to do in NYC")
        print("\nTest response from Grok:")
        print(response.content)
    except Exception as e:
        print(f"Error: {e}") 