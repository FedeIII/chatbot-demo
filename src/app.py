import os
import gradio as gr
from rag_chain import create_rag_chain, SYSTEM_PROMPT
from chat_memory import ChatbotWithMemory
import uuid
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Get the XAI API key from environment variables
xai_api_key = os.getenv('XAI_API_KEY')
if not xai_api_key:
    raise ValueError("XAI API key must be set in the .env file")

# Create the RAG chain using the Pinecone retriever
rag_chain = create_rag_chain()

# Create the chatbot with memory
chatbot_agent = ChatbotWithMemory(rag_chain, SYSTEM_PROMPT)

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
    return [], generate_session_id()

# Create the Gradio interface
with gr.Blocks(css="""
    .gradio-container {max-width: 800px; margin: auto;}
    .message-bot {background-color: #f0f0f0; padding: 10px; border-radius: 10px;}
    .message-user {background-color: #e6f7ff; padding: 10px; border-radius: 10px;}
""") as demo:
    gr.Markdown("# RAG-Powered Chatbot with XAI Grok-3-mini-beta")
    gr.Markdown("Ask questions and the bot will use the XAI Grok-3-mini-beta model with RAG to answer.")
    
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
            "What are some legal considerations for starting a business?",
            "Tell me about the benefits of forming an LLC.",
            "How do I register a business name?",
        ],
        inputs=msg
    )

# Launch the app
if __name__ == "__main__":
    print("Starting the Gradio app. Access it in your browser at http://127.0.0.1:7860")
    demo.launch(share=False) 