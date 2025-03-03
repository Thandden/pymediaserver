"""
Test configuration for pytest.
This file ensures that the src directory is in the Python path for tests.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to Python's path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root)) 