#!/usr/bin/env python
"""Test script to verify the LegifAI server setup."""

import os
import sys
from dotenv import load_dotenv

def test_environment():
    """Test that all required environment variables are set."""
    load_dotenv()
    
    required_vars = [
        'XAI_API_KEY',
        'PINECONE_API_KEY', 
        'PINECONE_INDEX_BOE',
        'OPENAI_API_KEY'
    ]
    
    optional_vars = [
        'LANGCHAIN_API_KEY_BOE',
        'LANGCHAIN_PROJECT_BOE'
    ]
    
    print("Testing environment variables...")
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
        else:
            print(f"✓ {var} is set")
    
    if missing_required:
        print(f"\n❌ Missing required environment variables: {missing_required}")
        print("Please check your .env file")
        return False
    
    print(f"\n✓ All required environment variables are set")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✓ {var} is set (optional)")
        else:
            print(f"! {var} is not set (optional)")
    
    return True

def test_imports():
    """Test that all required imports work."""
    print("\nTesting imports...")
    
    try:
        from app import app
        print("✓ Combined FastAPI + Gradio app imported successfully")
        
        from rag_chain import create_rag_chain_with_history
        print("✓ RAG chain creation function imported")
        
        from vector_store import init_vector_store
        print("✓ Vector store initialization function imported")
        
        from legifai_gradio import create_gradio_app
        print("✓ Gradio interface creation function imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_vector_store():
    """Test vector store connection."""
    print("\nTesting vector store connection...")
    
    try:
        from vector_store import init_vector_store
        retriever = init_vector_store()
        print("✓ Vector store connected successfully")
        
        # Test retrieval
        test_results = retriever.invoke("sociedad limitada")
        print(f"✓ Retrieved {len(test_results)} documents for test query")
        return True
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

def main():
    """Run all tests."""
    print("LegifAI Server Test Suite")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 3
    
    if test_environment():
        tests_passed += 1
    
    if test_imports():
        tests_passed += 1
    
    if test_vector_store():
        tests_passed += 1
    
    print(f"\n{'=' * 40}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The server should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 