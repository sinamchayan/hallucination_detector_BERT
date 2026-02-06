#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

# Function to cleanup background processes on exit
cleanup() {
    echo "Shutting down..."
    kill $(jobs -p)
}
trap cleanup EXIT

# Start the API in the background
echo "Starting API server..."
python run.py api &

# Wait a few seconds for API to be ready
sleep 5

# Start the UI
echo "Starting Streamlit UI..."
python run.py ui
