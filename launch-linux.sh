#!/bin/bash

# Activate the virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate

# Start the Python application
echo "Starting Python application..."
python3 Home.py

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Done."
