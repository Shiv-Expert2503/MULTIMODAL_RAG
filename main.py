import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.logger import logging
from src.exception import CustomException
from src.utils import load_resources, get_rag_response

app = FastAPI()

try:
    # Load all AI resources once when the server starts
    text_model, llm, text_collection = load_resources()
    logging.info("--- AI Resources loaded successfully on startup ---")
except Exception as e:
    # If models fail to load, log the detailed error and stop the app
    logging.error(CustomException(e, sys))
    # This prevents the server from starting in a broken state
    raise RuntimeError("Could not load AI models or database. Check logs.") from e


origins = [
    "http://localhost:5173", # Your local React dev server
    "https://shiv2503portfolio.netlify.app", # Your deployed portfolio URL
]

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

        return {"answer": response_text}

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        return {"error": str(error)}, 500