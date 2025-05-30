# LegifAI - Legal Consultation API

A FastAPI-based legal consultation chatbot that provides advice based on BOE (Boletín Oficial del Estado) documents using RAG (Retrieval-Augmented Generation). The chatbot follows a structured consultation flow with persistent conversation history.

## Features

- 🤖 **Intelligent Legal Consultation**: Uses XAI's Grok model for legal advice
- 📚 **BOE Document Retrieval**: Retrieves relevant Spanish legal documents
- 💬 **Persistent Conversations**: Maintains chat history across sessions
- 🔄 **Structured Flow**: Three-step consultation process
- 📊 **LangSmith Tracing**: Complete observability of the conversation chain
- 🚀 **Production Ready**: FastAPI server ready for deployment

## Architecture

- **Backend**: FastAPI with LangServe for API endpoints
- **LLM**: XAI Grok-3-mini for legal reasoning
- **Vector Store**: Pinecone for document retrieval
- **Embeddings**: OpenAI text-embedding-3-large
- **Persistence**: File-based chat history using LangChain
- **Observability**: LangSmith for tracing and monitoring

## Consultation Flow

1. **Initial Consultation**: User asks a legal question, bot provides brief answer with relevant BOE articles and asks follow-up questions
2. **Detailed Response**: User answers the questions, bot provides conclusion and technical summary for lawyers
3. **Case Closure**: Further interactions receive acknowledgment that the case is being handled

## Setup

### Prerequisites

- Python 3.8+
- XAI API key
- Pinecone account with populated index
- OpenAI API key (for embeddings)
- LangSmith account (optional, for tracing)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chatbot-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp env.example .env
# Edit .env with your API keys
```

4. Test the setup:
```bash
python src/test_server.py
```

5. Start the server:
```bash
./start_server.sh
# Or manually: cd src && python app.py
```

## API Endpoints

### Main Chat Endpoint
- **POST** `/chat/invoke` - Send a message to the chatbot
- **POST** `/chat/batch` - Send multiple messages
- **GET** `/chat/playground` - Interactive web interface

### Management Endpoints
- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/docs` - OpenAPI documentation

### Example Usage

```bash
# Start a conversation
curl -X POST "http://localhost:8000/chat/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "human_input": "¿Qué documentos necesito para crear una sociedad limitada?"
    },
    "config": {
      "configurable": {
        "session_id": "user123-session1"
      }
    }
  }'
```

## Deployment

### Render Deployment

1. Connect your repository to Render
2. Set environment variables in Render dashboard:
   - `XAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_BOE`
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY_BOE` (optional)
   - `LANGCHAIN_PROJECT_BOE` (optional)

3. Set build command: `pip install -r requirements.txt`
4. Set start command: `cd src && python app.py`

The server will automatically use the `PORT` environment variable provided by Render.

### Local Development

```bash
# Start with default port 8000
cd src && python app.py

# Or use uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
```

## LangSmith Integration

The application is configured to send detailed traces to LangSmith, showing:
- User input processing
- Document retrieval from Pinecone  
- LLM reasoning and response generation
- Complete conversation chain

Set `LANGCHAIN_API_KEY_BOE` and `LANGCHAIN_PROJECT_BOE` in your environment to enable tracing.

## File Structure

```
chatbot-demo/
├── src/
│   ├── app.py              # FastAPI server with LangServe
│   ├── rag_chain.py        # RAG chain with message history
│   ├── vector_store.py     # Pinecone vector store setup
│   ├── test_server.py      # Server test suite
│   └── ...                 # Other utilities
├── requirements.txt        # Python dependencies
├── env.example            # Environment variables template
├── start_server.sh        # Startup script
└── README.md              # This file
```

## Environment Variables

### Required
- `XAI_API_KEY`: Your XAI API key for Grok model
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_INDEX_BOE`: Name of your Pinecone index
- `OPENAI_API_KEY`: OpenAI API key for embeddings

### Optional
- `LANGCHAIN_API_KEY_BOE`: LangSmith API key for tracing
- `LANGCHAIN_PROJECT_BOE`: LangSmith project name
- `PORT`: Server port (auto-set by Render)

## License

[Your License Here]