#!/bin/bash

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if pex is installed
if ! command -v pex &> /dev/null
then
    echo "pex is not installed. Installing now..."
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null
    then
        echo "pip3 is not installed. Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py
        rm get-pip.py
    fi
    
    # Install pex using pip
    pip3 install pex
    
    echo "pex has been installed successfully!"
fi

# Check if test.pex exists in the current directory
if [ -f "$SCRIPT_DIR/aux.pex" ]; then
    echo "Starting Fitting Tool..."
    # Run the test.pex file
    "$SCRIPT_DIR/aux.pex"
else
    echo "Error: aux.pex not found in the current directory. Contact Ahmad"
    exit 1
fi