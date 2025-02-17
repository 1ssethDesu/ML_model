import gdown
import os

def download(model_path: str):
    """Download the latest model from Google Drive."""

    url = os.getenv("MODEL_URL")
    if not url:
        raise ValueError("MODEL_URL environment variable is not set.")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("getting latest model from Google Drive...")
    gdown.download(url=url, output=model_path, quiet=False, fuzzy=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found at path: {model_path}")
    
    print("Model downloaded successfully!")