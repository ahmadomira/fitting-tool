"""
Main entry point to launch the Fitting App GUI.
"""

import os
import sys

import openpyxl

# prevent removing openpyxl by the linter
if not hasattr(openpyxl, "__version__"):
    raise ImportError("openpyxl is not installed or is not accessible.")

# Ensure the gui/ directory is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gui"))

from gui.main_interface import main


def run():
    main()


if __name__ == "__main__":
    run()
