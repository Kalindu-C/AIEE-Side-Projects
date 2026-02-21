import os
from pathlib import Path
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Paths
path = Path(__file__).parent
ROOT = path.parent
DATA_DIR = ROOT / "data"

# API Keys
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY is not set in the .env file")