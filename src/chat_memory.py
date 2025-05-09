from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Dict, Any, Optional
import warnings

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
    """A chatbot with memory to maintain conversation context."""
    
    def __init__(self, rag_chain, system_prompt: str):
        """
        Initialize the chatbot with a RAG chain and system prompt.
        
        Args:
            rag_chain: The RAG chain for answering questions
            system_prompt: The system prompt for the chatbot
        """
        self.rag_chain = rag_chain
        self.system_prompt = system_prompt
        self.session_histories: Dict[str, List] = {}
        
    def chat(self, message: str, session_id: str) -> str:
        """
        Process a user message and return a response, maintaining conversation history.
        
        Args:
            message: The user's message
            session_id: The unique identifier for the conversation session
            
        Returns:
            The chatbot's response
        """
        # Initialize session history if not exists
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
            
        # Get history for this session
        history = self.session_histories[session_id]
        
        # Prepare context from history
        context = ""
        if history:
            context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
            context = f"Previous conversation:\n{context}\n\n"
        
        # Add current message to input
        full_message = f"{message}"
        
        # Get response from the chain
        response = self.rag_chain.invoke(full_message)
        
        # Store message and response in history
        history_entry = {"user": message, "assistant": response}
        self.session_histories[session_id].append(history_entry)
        
        return response
        
    def clear_history(self, session_id: str) -> None:
        """Clear the history for a given session."""
        if session_id in self.session_histories:
            self.session_histories[session_id] = []


if __name__ == "__main__":
    # Test the chatbot with memory
    from rag_chain import create_rag_chain, SYSTEM_PROMPT
    
    try:
        # Create the RAG chain
        rag_chain = create_rag_chain()
        
        # Create the chatbot with memory
        chatbot = ChatbotWithMemory(rag_chain, SYSTEM_PROMPT)
        
        # Test a conversation
        session_id = "test-session"
        
        # First message
        first_query = "What are some legal considerations for starting a business?"
        first_response = chatbot.chat(first_query, session_id)
        print(f"User: {first_query}")
        print(f"Bot: {first_response}\n")
        
        # Follow-up questions 
        follow_up_1 = "Can you explain more about business structures?"
        follow_up_response_1 = chatbot.chat(follow_up_1, session_id)
        print(f"User: {follow_up_1}")
        print(f"Bot: {follow_up_response_1}\n")
        
        follow_up_2 = "What about tax obligations?"
        follow_up_response_2 = chatbot.chat(follow_up_2, session_id)
        print(f"User: {follow_up_2}")
        print(f"Bot: {follow_up_response_2}")
        
    except Exception as e:
        print(f"Error: {e}") 