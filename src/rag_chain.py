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

# Configure LangSmith (ensure it's properly enabled)
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY_BOE')
langsmith_project = os.getenv('LANGCHAIN_PROJECT_BOE', 'lawyer-ai-boe')
langsmith_tracing = os.getenv('LANGCHAIN_TRACING_V2', 'false').lower() == 'true'

if langsmith_api_key:
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
        print("Please check your LANGCHAIN_API_KEY_BOE in the .env file")
else:
    print("LangSmith API key not found. Please set LANGCHAIN_API_KEY_BOE in your .env file.")

# Suppress warnings
warnings.filterwarnings("ignore")

# System prompt for the chatbot
SYSTEM_PROMPT = """Eres un servicial asesor legal. El siguiente contexto incluye extractos de artículos del BOE para ayudar a aconsejar al usuario sobre su consulta legal.
Úsalos para responder la pregunta lo más fielmente posible en base a su contenido.
Si no aplican directamente, usa el sentido común y tus conocimientos legales para responder la pregunta, usando el contexto como guía.

Se conciso. Asume que el usuario sabe que eres un asesor legal que tiene todos los conocimientos necesarios para ayudarle.

Ve directamente al grano, sin decir en qué te basas para responder, pero menciona los artículos del BOE que apliquen durante la explicación.

Contexto: {context}

Pregunta: {question}
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
