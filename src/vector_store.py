import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_xai import ChatXAI
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

def init_vector_store():
    """
    Initialize the Pinecone vector store and create a retriever
    
    Returns:
        VectorStoreRetriever: A retriever for the Pinecone vector store
    """
    # Get Pinecone credentials from environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_index_name = os.getenv('PINECONE_INDEX_BOE')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not pinecone_api_key or not pinecone_index_name:
        raise ValueError("Pinecone API key and index name must be set")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key must be set for embeddings")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Initialize embeddings with text-embedding-3-large model and 3072 dimensions
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        dimensions=3072,
        api_key=openai_api_key
    )
    
    # Get the Pinecone index
    try:
        index = pc.Index(pinecone_index_name)
        print(f"Successfully connected to Pinecone index: {pinecone_index_name}")
    except Exception as e:
        raise ValueError(f"Error connecting to Pinecone index: {e}")
    
    # Create the vector store
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    
    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve top 5 most similar documents
    )
    
    return retriever

if __name__ == "__main__":
    # Test the vector store setup
    try:
        retriever = init_vector_store()
        print("Vector store retriever initialized successfully.")
        
        # Test the retriever with a simple query
        docs = retriever.invoke("test query")
        print(f"Retrieved {len(docs)} documents")
        
        # Print the first document content (if any)
        if len(docs) > 0:
            print("\nSample document content:")
            print(docs[0].page_content[:200] + "..." if len(docs[0].page_content) > 200 else docs[0].page_content)
    except Exception as e:
        print(f"Error: {e}") 