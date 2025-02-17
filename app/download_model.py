try:
    import gdown
    from dotenv import load_dotenv
except ImportError:
    print("Downloading required libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "gdown", "python-dotenv"])
    import gdown
    from dotenv import load_dotenv

import os

def download():
    """Download the latest model from Google Drive."""
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(dotenv_path)

    url = os.getenv("MODEL_URL")
    model_path = os.path.join("app", "models", "model.pt")

    if not os.path.exists("app/models"):
        os.makedirs("app/models")

    print("getting latest model from Google Drive...")
    gdown.download(url=url, output=model_path, quiet=False, fuzzy=True)