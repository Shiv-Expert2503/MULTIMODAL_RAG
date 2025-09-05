import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles 
from src.logger import logging
from src.exception import CustomException
from src.utils import load_resources, get_rag_response, extract_image_paths

app = FastAPI()

app.mount("/images", StaticFiles(directory="data/images"), name="images")

try:
    # Load all AI resources once when the server starts
    text_model, llm, text_collection = load_resources()
    logging.info("--- AI Resources loaded successfully on startup ---")
except Exception as e:
    # If models fail to load, log the detailed error and stop the app
    logging.error(CustomException(e, sys))
    # This prevents the server from starting in a broken state
    raise RuntimeError("Could not load AI models or database. Check logs.") from e

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in allowed_origins_str.split(',')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_agent(query: Query):
    try:
        question = query.question
        logging.info(f"API received query: {question}")
        response_text = get_rag_response(question, text_collection, text_model, llm)
        image_paths = extract_image_paths(response_text)
        return {"answer": response_text, "images": image_paths}  

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        return {"error": str(error)}, 500