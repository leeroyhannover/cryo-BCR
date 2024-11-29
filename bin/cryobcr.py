#!/usr/bin/env python3

import os
import sys

# Dynamically add the parent directory to PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from cryobcr.main import main

if __name__ == "__main__":
    main()