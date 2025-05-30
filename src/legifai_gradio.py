#!/usr/bin/env python
"""Gradio client for LegifAI that communicates with the FastAPI backend."""

import gradio as gr
import requests
import uuid
import json
import os
from typing import List, Tuple

class LegifAIGradioClient:
    def __init__(self, api_base_url: str = None):
        """Initialize the Gradio client."""
        # For deployment, the API will be on the same host
        self.api_base_url = api_base_url or "http://localhost:8000"
        
    def send_message_to_api(self, message: str, session_id: str) -> str:
        """Send a message to the FastAPI backend."""
        try:
            url = f"{self.api_base_url}/chat/invoke"
            
            payload = {
                "input": {
                    "human_input": message
                },
                "config": {
                    "configurable": {
                        "session_id": session_id
                    }
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                # Extract the response text from the dictionary format
                if isinstance(result.get("output"), dict) and "response" in result["output"]:
                    return result["output"]["response"]
                else:
                    return result.get("output", "Error: Unexpected response format")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "❌ No se pudo conectar con el servidor. Por favor, inténtalo más tarde."
        except requests.exceptions.Timeout:
            return "⏱️ El servidor tardó demasiado en responder. Por favor, inténtalo de nuevo."
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}"

    def chat_response(self, message: str, history: List[Tuple[str, str]], session_id: str) -> Tuple[str, List[Tuple[str, str]], str]:
        """Process chat message and return response."""
        if not message.strip():
            return "", history, session_id
            
        # Generate session ID if not provided
        if not session_id:
            session_id = f"gradio-session-{uuid.uuid4()}"
            
        # Get response from API
        bot_response = self.send_message_to_api(message, session_id)
        
        # Update history
        history = history or []
        history.append((message, bot_response))
        
        return "", history, session_id

    def clear_conversation(self) -> Tuple[List, str]:
        """Clear the conversation and generate a new session ID."""
        new_session_id = f"gradio-session-{uuid.uuid4()}"
        return [], new_session_id

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        
        with gr.Blocks(
            title="LegifAI - Consultor Legal",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {
                    max-width: 900px !important;
                    margin: auto !important;
                }
                .chat-message {
                    padding: 10px !important;
                    margin: 5px 0 !important;
                    border-radius: 10px !important;
                }
                .user-message {
                    background-color: #e3f2fd !important;
                    margin-left: 20% !important;
                }
                .bot-message {
                    background-color: #f5f5f5 !important;
                    margin-right: 20% !important;
                }
                #session-info {
                    font-size: 0.8em;
                    color: #666;
                    text-align: center;
                }
            """
        ) as interface:
            
            gr.Markdown("""
            # ⚖️ LegifAI - Consultor Legal Inteligente
            
            Bienvenido a LegifAI, tu asistente legal especializado en legislación española. 
            Haz tu consulta legal y recibirás asesoramiento basado en documentos del BOE.
            
            **Proceso de consulta:**
            1. 📋 **Consulta inicial**: Describe tu situación legal
            2. ❓ **Preguntas de seguimiento**: Responde a las preguntas específicas
            3. ⚖️ **Conclusión legal**: Recibe tu asesoramiento y resumen técnico
            """)
            
            with gr.Row():
                with gr.Column():
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Conversación con LegifAI",
                        height=500,
                        show_copy_button=True,
                        avatar_images=("👤", "⚖️"),
                        bubble_full_width=False,
                        show_share_button=False
                    )
                    
                    # Message input
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Escribe tu consulta legal aquí...",
                            label="Tu mensaje",
                            scale=4,
                            container=False,
                            autofocus=True
                        )
                        send_btn = gr.Button("Enviar", variant="primary", scale=1)
                    
                    # Action buttons
                    with gr.Row():
                        clear_btn = gr.Button("Nueva Consulta", variant="secondary")
                        
                    # Session ID (hidden)
                    session_id = gr.State(f"gradio-session-{uuid.uuid4()}")
                    
                    # Session info
                    gr.Markdown(
                        "💡 **Tip**: Cada consulta mantiene el contexto durante toda la conversación.",
                        elem_id="session-info"
                    )
            
            # Example queries
            with gr.Row():
                gr.Examples(
                    examples=[
                        "¿Qué documentos necesito para crear una sociedad limitada?",
                        "¿Cuáles son los derechos laborales básicos en España?",
                        "¿Cómo puedo registrar una marca comercial?",
                        "¿Qué pasos debo seguir para divorciarse por mutuo acuerdo?",
                        "¿Cuáles son las obligaciones fiscales de un autónomo?",
                    ],
                    inputs=msg,
                    label="Ejemplos de consultas legales"
                )
            
            # Footer
            gr.Markdown("""
            ---
            <div style="text-align: center; color: #666; font-size: 0.9em;">
                🔒 Tus consultas se mantienen privadas y seguras<br>
                ⚖️ LegifAI utiliza documentos oficiales del BOE para proporcionar asesoramiento preciso<br>
                📊 Powered by XAI Grok, Pinecone & LangChain
            </div>
            """)
            
            # Event handlers
            def respond(message, history, session_id):
                return self.chat_response(message, history, session_id)
            
            def clear():
                return self.clear_conversation()
            
            # Set up interactions
            send_btn.click(
                respond,
                inputs=[msg, chatbot, session_id],
                outputs=[msg, chatbot, session_id]
            )
            
            msg.submit(
                respond,
                inputs=[msg, chatbot, session_id],
                outputs=[msg, chatbot, session_id]
            )
            
            clear_btn.click(
                clear,
                outputs=[chatbot, session_id]
            )
        
        return interface

def create_gradio_app(api_base_url: str = None) -> gr.Blocks:
    """Create and return the Gradio application."""
    client = LegifAIGradioClient(api_base_url)
    return client.create_interface()

if __name__ == "__main__":
    # For standalone testing
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 