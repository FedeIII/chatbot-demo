# LegifAI - Legal Consultation API & Web Interface

A complete legal consultation solution that provides advice based on BOE (Boletín Oficial del Estado) documents using RAG (Retrieval-Augmented Generation). Includes both a FastAPI backend with REST endpoints and a beautiful Gradio web interface, all deployed as a single service.

## Features

- 🤖 **Intelligent Legal Consultation**: Uses XAI's Grok model for legal advice
- 📚 **BOE Document Retrieval**: Retrieves relevant Spanish legal documents
- 💬 **Persistent Conversations**: Maintains chat history across sessions
- 🔄 **Structured Flow**: Three-step consultation process
- 📊 **LangSmith Tracing**: Complete observability of the conversation chain
- 🌐 **Dual Interface**: Web UI (Gradio) + REST API (FastAPI)
- 🚀 **Production Ready**: Single service deployment on Render

## Architecture

- **Backend**: FastAPI with LangServe for API endpoints
- **Frontend**: Gradio web interface for user-friendly interaction
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

## Available Interfaces

### 🌐 Web Interface (Gradio)
- **URL**: `/ui` or root `/`
- **Description**: Beautiful, user-friendly chat interface
- **Features**: 
  - Spanish language interface
  - Legal consultation examples
  - Session persistence
  - Professional styling
  - Mobile-responsive design

### 🔗 REST API (FastAPI)
- **Chat Endpoint**: `POST /chat/invoke`
- **API Docs**: `/docs`
- **Playground**: `/chat/playground`
- **Health Check**: `/health`

### Example API Usage

```bash
# Start a legal consultation
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

## Deployment on Render

### Option 1: Using render.yaml (Recommended for new services)

1. **Create New Service**: In Render dashboard, click "New" → "Blueprint"
2. **Connect Repository**: Link your GitHub repository
3. **Auto-Configuration**: Render will detect `render.yaml` and pre-fill settings
4. **Set Secret Environment Variables**: Add your API keys in the dashboard:
   - `XAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_BOE`
   - `OPENAI_API_KEY`
   - `LANGCHAIN_API_KEY_BOE` (optional)

### Option 2: Manual Configuration (For existing services)

1. **Push Changes**: Commit and push all code to your repository
2. **Update Service Settings** in Render dashboard:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `cd src && python app.py`
3. **Set Environment Variables**:
   - Add all required API keys
   - Set `LANGCHAIN_PROJECT_BOE=lawyer-ai-boe`
   - Set `LANGCHAIN_TRACING_V2=true`
4. **Deploy**: Trigger manual deploy or push to auto-deploy

After deployment, your service will be available at:
- **Web Interface**: `https://your-service.onrender.com/ui`
- **API Documentation**: `https://your-service.onrender.com/docs`
- **API Endpoint**: `https://your-service.onrender.com/chat/invoke`

### Local Development

```bash
# Start combined service (API + Web UI)
cd src && python app.py

# Available at:
# Web Interface: http://localhost:8000/ui
# API Docs: http://localhost:8000/docs
# Chat API: http://localhost:8000/chat
```

## LangSmith Integration

The application sends detailed traces to LangSmith, showing:
- User input processing (both web and API)
- Document retrieval from Pinecone  
- LLM reasoning and response generation
- Complete conversation chain across interfaces

Set `LANGCHAIN_API_KEY_BOE` and `LANGCHAIN_PROJECT_BOE` in your environment to enable tracing.

## File Structure

```
chatbot-demo/
├── src/
│   ├── app.py                 # Combined FastAPI + Gradio server
│   ├── legifai_gradio.py      # Gradio web interface
│   ├── rag_chain.py           # RAG chain with message history
│   ├── vector_store.py        # Pinecone vector store setup
│   ├── test_server.py         # Server test suite
│   └── chat_histories/        # Session storage (auto-created)
├── requirements.txt           # Python dependencies
├── env.example               # Environment variables template
├── render.yaml               # Render deployment configuration
├── start_server.sh           # Startup script
├── client_example.py         # API client example
└── README.md                 # This file
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

## Testing

Run the comprehensive test suite:
```bash
python src/test_server.py
```

Test the API client:
```bash
python client_example.py
```

## License

[Your License Here]