import os
from dotenv import load_dotenv
from pathlib import Path

# Correct: only go up one level
dotenv_path = Path(__file__).resolve().parent.parent / ".env"

# or simply:
# dotenv_path = Path(__file__).resolve().parent.parent / ".env"

print("DOTENV PATH:", dotenv_path)
print("EXISTS?", dotenv_path.exists())

load_dotenv(dotenv_path)

print("SERPER_API_KEY:", os.getenv("SERPER_API_KEY"))
