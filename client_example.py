#!/usr/bin/env python
"""Example client for interacting with the LegifAI API."""

import requests
import json
import uuid

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change to your deployed URL
SESSION_ID = f"example-session-{uuid.uuid4()}"

def send_message(message, session_id=SESSION_ID):
    """Send a message to the LegifAI chatbot."""
    url = f"{API_BASE_URL}/chat/invoke"
    
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
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result["output"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def example_consultation():
    """Example consultation flow."""
    print("🤖 LegifAI - Example Consultation")
    print("=" * 50)
    
    # First interaction - initial legal question
    print("\n📋 Step 1: Initial Consultation")
    first_message = "¿Qué documentos necesito para crear una sociedad limitada?"
    print(f"User: {first_message}")
    
    response1 = send_message(first_message)
    if response1:
        print(f"LegifAI: {response1}")
    
    # Second interaction - answering the bot's questions
    print("\n📋 Step 2: Detailed Information")
    second_message = "Queremos crear una SL con 2 socios, capital inicial de 10.000€, y actividad de consultoría tecnológica."
    print(f"User: {second_message}")
    
    response2 = send_message(second_message)
    if response2:
        print(f"LegifAI: {response2}")
    
    # Third interaction - case closure
    print("\n📋 Step 3: Follow-up")
    third_message = "Gracias por la información."
    print(f"User: {third_message}")
    
    response3 = send_message(third_message)
    if response3:
        print(f"LegifAI: {response3}")

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print("❌ API is not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

if __name__ == "__main__":
    print("LegifAI API Client Example")
    print("=" * 40)
    
    # Check API health
    if check_api_health():
        print(f"Session ID: {SESSION_ID}")
        example_consultation()
    else:
        print("\nPlease start the LegifAI server:")
        print("cd src && python app.py") 