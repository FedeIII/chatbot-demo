#!/bin/bash

# LegifAI Server Startup Script

echo "Starting LegifAI Server..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy env.example to .env and fill in your API keys"
    exit 1
fi

# Navigate to src directory
cd src

# Run the test script first
echo "Running pre-deployment tests..."
python test_server.py

if [ $? -ne 0 ]; then
    echo "Tests failed! Please fix the issues before starting the server."
    exit 1
fi

# Start the server
echo "Tests passed! Starting the FastAPI server..."
python app.py 