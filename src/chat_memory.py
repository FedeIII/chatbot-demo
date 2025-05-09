from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Dict, Any, Optional
import warnings
import os
from dotenv import load_dotenv
from langsmith import Client

# Load environment variables
load_dotenv()

# Configure LangSmith
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY_BOE')
if langsmith_api_key:
    os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
    os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT_BOE', 'lawyer-ai-boe')
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Suppress warnings
warnings.filterwarnings("ignore")

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """In-memory implementation of chat message history."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List = []
    
    def add_message(self, message):
        self.messages.append(message)
    
    def clear(self):
        self.messages = []
        
    @property
    def messages(self) -> List:
        return self._messages
    
    @messages.setter
    def messages(self, messages: List):
        self._messages = messages


class ChatbotWithMemory:
    """A chatbot with memory to maintain conversation context and documents."""
    
    def __init__(self, retriever, model, prompt_template: str):
        """
        Initialize the chatbot with retriever, model and prompt template.
        
        Args:
            retriever: The vector store retriever for RAG
            model: The LLM to use for responses
            prompt_template: The template for the chatbot prompts
        """
        self.retriever = retriever
        self.model = model
        self.prompt_template = prompt_template
        self.session_histories = {}
        self.session_documents = {}
        
    def chat(self, message: str, session_id: str) -> str:
        """
        Process a user message and return a response, maintaining conversation history
        and retrieving documents only on first message.
        
        Args:
            message: The user's message
            session_id: The unique identifier for the conversation session
            
        Returns:
            The chatbot's response
        """
        # Initialize session if it doesn't exist
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
            # Retrieve documents only on first message
            docs = self.retriever.invoke(message)
            self.session_documents[session_id] = docs
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            # Use previously retrieved documents for follow-up messages
            if session_id in self.session_documents:
                context = "\n\n".join([doc.page_content for doc in self.session_documents[session_id]])
            else:
                context = ""
        
        # Get history and documents for this session
        history = self.session_histories[session_id]
        
        # Prepare conversation history context
        history_context = ""
        if history:
            history_context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
            history_context = f"Previous conversation:\n{history_context}\n\n"
            
        # Process response using the model and retrieved documents
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        inputs = {
            "context": context,
            "question": message
        }
        
        response = self.model.invoke(prompt.format(**inputs))
        response_text = response.content
        
        # Store message and response in history
        history_entry = {"user": message, "assistant": response_text}
        self.session_histories[session_id].append(history_entry)
        
        return response_text
        
    def clear_history(self, session_id: str) -> None:
        """Clear the history and documents for a given session."""
        if session_id in self.session_histories:
            self.session_histories[session_id] = []
        if session_id in self.session_documents:
            del self.session_documents[session_id] 