#!/usr/bin/env python
"""LegifAI - Legal consultation chatbot with persistence.

A FastAPI server that provides legal consultation services using RAG (Retrieval-Augmented Generation)
with BOE (Boletín Oficial del Estado) documents. The server maintains conversation history
and follows a structured consultation flow.

This version includes both the API endpoints and a Gradio web interface.
"""
import re
import os
from pathlib import Path
from typing import Callable, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langserve import add_routes
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import gradio as gr

from rag_chain import create_rag_chain_with_history
from legifai_gradio import create_gradio_app

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
        from langsmith import Client
        client = Client()
        print("Successfully connected to LangSmith")
    except Exception as e:
        print(f"Warning: Could not connect to LangSmith: {e}")
        print("Please check your LANGCHAIN_API_KEY_BOE in the .env file")
else:
    print("LangSmith API key not found. Please set LANGCHAIN_API_KEY_BOE in your .env file.")

# Verify XAI API key
xai_api_key = os.getenv('XAI_API_KEY')
if not xai_api_key:
    raise ValueError("XAI API key must be set in the .env file")


def _is_valid_identifier(value: str) -> bool:
    """Check if the session ID is in a valid format."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    base_dir: Union[str, Path],
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a session ID factory that creates session IDs from a base dir.

    Args:
        base_dir: Base directory to use for storing the chat histories.

    Returns:
        A session ID factory that creates session IDs from a base path.
    """
    base_dir_ = Path(base_dir) if isinstance(base_dir, str) else base_dir
    if not base_dir_.exists():
        base_dir_.mkdir(parents=True)

    def get_chat_history(session_id: str) -> FileChatMessageHistory:
        """Get a chat history from a session ID."""
        if not _is_valid_identifier(session_id):
            raise HTTPException(
                status_code=400,
                detail=f"Session ID `{session_id}` is not in a valid format. "
                "Session ID must only contain alphanumeric characters, "
                "hyphens, and underscores.",
            )
        file_path = base_dir_ / f"{session_id}.json"
        return FileChatMessageHistory(str(file_path))

    return get_chat_history


# Create FastAPI app
app = FastAPI(
    title="LegifAI - Legal Consultation API",
    version="1.0",
    description="A legal consultation chatbot that provides advice based on BOE (Boletín Oficial del Estado) documents. "
                "The chatbot follows a structured consultation flow: initial consultation, follow-up questions, "
                "and final legal summary. Includes both API endpoints and web interface.",
)


class InputChat(BaseModel):
    """Input for the chat endpoint."""
    
    human_input: str = Field(
        ...,
        description="The human input to the legal consultation system.",
        examples=["¿Qué documentos necesito para crear una sociedad limitada?"]
    )


# Create the RAG chain with history
chain_with_history = create_rag_chain_with_history(
    create_session_factory("chat_histories")
).with_types(input_type=InputChat)


# Add the chat route
add_routes(
    app,
    chain_with_history,
    path="/chat",
)

# Create and mount Gradio app
print("Creating Gradio interface...")
gradio_app = create_gradio_app(api_base_url="")  # Empty string means same host
gradio_app.queue()  # Enable queuing for better performance

# Mount Gradio app
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

@app.get("/")
async def root():
    """Root endpoint - redirect to Gradio interface."""
    return RedirectResponse(url="/ui")

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Welcome to LegifAI - Legal Consultation API",
        "description": "A legal consultation chatbot powered by RAG with BOE documents",
        "interfaces": {
            "web": "/ui - Web interface (Gradio)",
            "api": "/chat - REST API endpoints",
            "docs": "/docs - API documentation",
            "playground": "/chat/playground - Interactive API playground"
        },
        "usage": {
            "web": "Visit /ui for the web interface",
            "api": "Send POST requests to /chat/invoke with session configuration"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "LegifAI", "interfaces": ["web", "api"]}


if __name__ == "__main__":
    import uvicorn
    
    # For Render deployment, use 0.0.0.0 and PORT from environment
    server_name = "0.0.0.0"
    
    # Render provides the PORT environment variable
    port_env = os.getenv("PORT")
    if port_env:
        server_port = int(port_env)
        print(f"Launching LegifAI (API + Web UI) on {server_name}:{server_port} (PORT from environment)")
    else:
        # Fallback for local development
        server_port = 8000
        print(f"Launching LegifAI (API + Web UI) on {server_name}:{server_port} (local development)")
        
    print(f"🌐 Web Interface: http://{server_name}:{server_port}/ui")
    print(f"🔗 API Docs: http://{server_name}:{server_port}/docs")
    print(f"💬 Chat API: http://{server_name}:{server_port}/chat")

    uvicorn.run(app, host=server_name, port=server_port)