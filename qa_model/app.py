import yaml
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qa_model.pipeline import get_answer
import uvicorn

# Load the configuration from the config.yaml file
def load_config(file_path: str):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise Exception(f"Config file not found: {file_path}")
    except yaml.YAMLError as e:
        raise Exception(f"Error loading YAML config: {e}")

# Load the configuration values
config = load_config(r"D:/Downloads/Website-QA-Model/config/config.yaml")  # Absolute path

# Create FastAPI app instance
app = FastAPI()

# ✅ Add CORS middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# You can access configuration values like this:
api_host = config['api']['host']
api_port = config['api']['port']

# Example usage of configuration values (like bot_token)
bot_token = os.getenv("BOT_TOKEN", config['telegram']['bot_token'])  # Use environment variable if available

# Request model for QA
class QARequest(BaseModel):
    question: str

@app.post("/answer")
async def answer(request: QARequest):
    result = get_answer(request.question)  # remove 'await'
    return result

# Start the app with the following:
if __name__ == "__main__":
    uvicorn.run(app, host=api_host, port=api_port)
