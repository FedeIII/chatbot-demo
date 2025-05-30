import os
import gradio as gr
from rag_chain import create_rag_chain, SYSTEM_PROMPT
from chat_memory import ChatbotWithMemory
import uuid
import warnings
from dotenv import load_dotenv
from vector_store import init_vector_store
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import Client

# Load environment variables
load_dotenv()

# Configure LangSmith (mandatory)
langsmith_api_key = os.getenv('LANGCHAIN_API_KEY_BOE')
langsmith_project = os.getenv('LANGCHAIN_PROJECT_BOE', 'lawyer-ai-boe')

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

# Get the XAI API key from environment variables
xai_api_key = os.getenv('XAI_API_KEY')
if not xai_api_key:
    raise ValueError("XAI API key must be set in the .env file")

# Create the chatbot with memory
class RAGChatbot:
    def __init__(self):
        self.session_docs = {}
        self.retriever = init_vector_store()
        self.model = ChatXAI(xai_api_key=xai_api_key, model="grok-3-mini")
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        self.session_histories = {}
        self.conversation_steps = {}  # Track the step in the conversation flow
        
    def chat(self, message, session_id):
        # Initialize session if it doesn't exist
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
            self.conversation_steps[session_id] = 1  # Initial step
            # Retrieve documents only on first message of the session
            docs = self.retriever.invoke(message)
            self.session_docs[session_id] = docs
            context = "\n\n".join([doc.page_content for doc in docs])
        else:
            # Use previously retrieved documents for follow-up messages
            if session_id in self.session_docs:
                context = "\n\n".join([doc.page_content for doc in self.session_docs[session_id]])
            else:
                context = ""
            
            # Advance to the next conversation step    
            current_step = self.conversation_steps.get(session_id, 1)
            self.conversation_steps[session_id] = current_step + 1
                
        # Get history for this session
        history = self.session_histories[session_id]
        current_step = self.conversation_steps[session_id]
        
        # Prepare conversation history context
        history_context = ""
        if history:
            history_context = "\n".join([f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history])
            history_context = f"Previous conversation:\n{history_context}\n\n"
            
        # Prepare input for the model
        inputs = {
            "context": context,
            "question": message
        }
        
        # Add conversation step hint to the question
        if current_step == 1:
            inputs["question"] = f"{message} [PRIMERA INTERACCIÓN]"
        elif current_step == 2:
            inputs["question"] = f"{message} [SEGUNDA INTERACCIÓN - RESPUESTA DEL USUARIO A LAS PREGUNTAS]"
        else:
            inputs["question"] = f"{message} [INTERACCIÓN POSTERIOR]"
        
        # Get response from the model - fixed chain construction
        response = self.model.invoke(self.prompt.format(**inputs))
        response_text = response.content
        
        # Store message and response in history
        history_entry = {"user": message, "assistant": response_text}
        self.session_histories[session_id].append(history_entry)
        
        return response_text
        
    def clear_history(self, session_id):
        if session_id in self.session_histories:
            self.session_histories[session_id] = []
        if session_id in self.session_docs:
            del self.session_docs[session_id]
        if session_id in self.conversation_steps:
            self.conversation_steps[session_id] = 1

# Initialize chatbot
chatbot_agent = RAGChatbot()

def generate_session_id():
    """Generate a unique session ID for a new conversation."""
    return str(uuid.uuid4())

def respond(message, chat_history, session_id):
    """Process the user message and return the chatbot's response."""
    # Create a new session if needed
    if not session_id:
        session_id = generate_session_id()
        
    # Get the bot's response
    response = chatbot_agent.chat(message, session_id)
    
    # Update the chat history
    chat_history.append((message, response))
    
    return "", chat_history, session_id

def clear_conversation():
    """Clear the conversation and start a new session."""
    session_id = generate_session_id()
    return [], session_id

# Create the Gradio interface
with gr.Blocks(css="""
    .gradio-container {max-width: 800px; margin: auto;}
    .message-bot {background-color: #f0f0f0; padding: 10px; border-radius: 10px;}
    .message-user {background-color: #e6f7ff; padding: 10px; border-radius: 10px;}
""") as demo:
    gr.Markdown("# LegifAI")
    gr.Markdown("Ask legal questions and the bot will retrieve relevant BOE (Boletín Oficial del Estado) documents for reference.")
    
    with gr.Row():
        with gr.Column():
            # Hidden session ID
            session_id = gr.State(generate_session_id())
            
            # Chat interface
            chat_interface = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False,
                show_copy_button=True,
                avatar_images=("🧑", "🤖"),
                elem_id="chatbot"
            )
            
            # Message input
            msg = gr.Textbox(
                placeholder="Type your message here...",
                label="Your Message",
                scale=4,
                container=False,
                elem_id="message-input"
            )
            
            # Buttons
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Conversation", variant="secondary", scale=1)
    
    # Set up event handlers
    submit_btn.click(respond, [msg, chat_interface, session_id], [msg, chat_interface, session_id])
    msg.submit(respond, [msg, chat_interface, session_id], [msg, chat_interface, session_id])
    clear_btn.click(clear_conversation, [], [chat_interface, session_id])
    
    # Examples
    gr.Examples(
        examples=[
            "¿Qué documentos necesito para crear una sociedad limitada?",
            "¿Cuáles son los derechos laborales básicos en España?",
            "¿Cómo puedo registrar una marca comercial?",
        ],
        inputs=msg
    )

# Launch the app
if __name__ == "__main__":
    print("Starting the Gradio app. Access it in your browser at http://127.0.0.1:7860")
    server_name = os.getenv("RENDER_INTERNAL_HOSTNAME", "0.0.0.0")
    server_port = int(os.getenv("PORT", 7860))
    demo.launch(server_name=server_name, server_port=server_port, share=False) 