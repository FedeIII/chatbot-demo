import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
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

# Custom output parser for LangServe compatibility
class LegifAIOutputParser(BaseOutputParser[dict]):
    """Custom output parser that returns the response in LangServe-compatible format."""
    
    def parse(self, text):
        """Parse the output from the language model."""
        if hasattr(text, 'content'):
            response_text = text.content
        else:
            response_text = str(text)
        
        # Return in the format expected by LangServe
        return {"response": response_text}
    
    @property
    def _type(self):
        return "legifai_output_parser"

# System prompt for the chatbot - updated to work with message history
SYSTEM_PROMPT = """Eres un servicial asesor legal especializado en legislación española. El siguiente contexto incluye extractos de artículos del BOE para ayudar a aconsejar al usuario sobre su consulta legal.

INSTRUCCIONES SEGÚN EL PASO DE LA CONVERSACIÓN:

1. **PRIMERA INTERACCIÓN** (si no hay historial previo):
   - Proporciona una respuesta breve y concisa mencionando los artículos del BOE relevantes
   - Formula 2-3 preguntas específicas para entender mejor el caso del usuario
   - Sé directo y profesional

2. **SEGUNDA INTERACCIÓN** (cuando el usuario responde a tus preguntas):
   - SOLO presenta una conclusión breve indicando que te harás cargo del caso
   - Seguido de un resumen técnico muy conciso dirigido a un abogado
   - NO hagas más preguntas en esta etapa
   - Formato: "Me haré cargo de su caso. [RESUMEN TÉCNICO PARA ABOGADO: ...]"

3. **INTERACCIONES POSTERIORES**:
   - Simplemente agradece al usuario
   - Reitera que su caso será atendido
   - Mantén la respuesta muy breve

DIRECTRICES GENERALES:
- Sé conciso y ve directamente al grano
- Menciona los artículos del BOE que apliquen durante la explicación
- Asume que el usuario sabe que eres un asesor legal competente
- Usa un tono profesional pero accesible

Contexto de documentos BOE:
{context}

Historial de conversación:
{history}

Consulta actual: {human_input}"""


def create_rag_chain_with_history(get_session_history):
    """
    Create a RAG chain with message history persistence

    Args:
        get_session_history: Function to get chat history for a session

    Returns:
        chain: A LangChain chain with message history that combines retrieval and generation
    """
    # Get the XAI API key from environment variables
    xai_api_key = os.getenv('XAI_API_KEY')
    if not xai_api_key:
        raise ValueError("XAI API key must be set")

    # Initialize the retriever from Pinecone
    retriever = init_vector_store()

    # Initialize the ChatXAI model
    model = ChatXAI(xai_api_key=xai_api_key, model="grok-3-mini")

    # Create the prompt template with message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{human_input}")
    ])

    # Function to retrieve context based on the human input
    def get_context(inputs):
        docs = retriever.invoke(inputs["human_input"])
        return "\n\n".join([doc.page_content for doc in docs])

    # Create the RAG chain with custom output parser
    rag_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(get_context))
        | prompt
        | model
        | LegifAIOutputParser()
    )

    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="human_input",
        history_messages_key="history",
    )

    return chain_with_history
