# RAG-Powered Chatbot with XAI Grok-3-mini-beta

This project implements a chatbot that uses Retrieval-Augmented Generation (RAG) with XAI's Grok-3-mini-beta model to provide context-aware responses. The chatbot retrieves relevant information from a Pinecone vector store and maintains conversation context.

## Features

- Uses XAI's Grok-3-mini-beta model
- Implements RAG with Pinecone vector store
- Maintains conversation context across multiple interactions
- Provides a simple Gradio UI for interaction

## Prerequisites

- Python 3.8+
- API keys for:
  - XAI (for Grok-3-mini-beta)
  - Pinecone (for vector store)
  - OpenAI (for embeddings)
  - LangChain (optional, for tracing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/chatbot-demo.git
   cd chatbot-demo
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   XAI_API_KEY=your_xai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_INDEX_BOE=your_pinecone_index_name_here
   LANGCHAIN_API_KEY_BOE=your_langchain_api_key_here
   LANGCHAIN_PROJECT_BOE=your_langchain_project_name_here
   LANGCHAIN_TRACING_V2=true
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the UI

1. Start the Gradio interface:
   ```
   python app.py
   ```

2. Open your browser and go to `http://127.0.0.1:7860` to interact with the chatbot.

### Testing Components Individually

- Test Vector Store: `python vector_store.py`
- Test RAG Chain: `python rag_chain.py`
- Test Chat Memory: `python chat_memory.py`

## Project Structure

- `main.py`: Loads environment variables and verifies setup
- `vector_store.py`: Sets up the Pinecone vector store and retriever
- `rag_chain.py`: Implements the RAG chain with the Grok model
- `chat_memory.py`: Manages conversation history and context
- `app.py`: Provides the Gradio UI for the chatbot

## Notes

- The chatbot uses a dummy retriever by default in `app.py` for demonstration purposes. 
- Set `use_real_retriever=True` in `app.py` once you have your Pinecone credentials set up.
- For production use, consider implementing proper error handling and monitoring.

## License

[Your License]