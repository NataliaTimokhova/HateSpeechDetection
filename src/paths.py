import os
from pathlib import Path

# Get the root of the project (2 levels up from current file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Common data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"