#!/bin/bash

# Create a new virtual environment
echo "Creating virtual environment..."
python3 -m venv myenv

# Activate the virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Done."
