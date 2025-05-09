import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Print environment variables
print("Environment Variables:")
print(f"XAI_API_KEY: {'✓' if os.getenv('XAI_API_KEY') else '✗'}")
print(f"PINECONE_API_KEY: {'✓' if os.getenv('PINECONE_API_KEY') else '✗'}")
print(f"PINECONE_INDEX_BOE: {'✓' if os.getenv('PINECONE_INDEX_BOE') else '✗'}")
print(f"OPENAI_API_KEY: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
print(f"LANGCHAIN_API_KEY_BOE: {'✓' if os.getenv('LANGCHAIN_API_KEY_BOE') else '✗'}")
print(f"LANGCHAIN_PROJECT_BOE: {'✓' if os.getenv('LANGCHAIN_PROJECT_BOE') else '✗'}")
print(f"LANGCHAIN_TRACING_V2: {'✓' if os.getenv('LANGCHAIN_TRACING_V2') else '✗'}") 