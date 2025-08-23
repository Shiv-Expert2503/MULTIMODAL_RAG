# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Crucial for React integration
from pydantic import BaseModel
import os

# --- THIS IS WHERE YOU WILL ADD YOUR RAG LOGIC ---
# from rag_pipeline import initialize_models, get_rag_response
# text_collection, image_collection, text_embedding_model, image_embedding_model, llm = initialize_models()
# ---------------------------------------------------

# Initialize the FastAPI app
app = FastAPI()

# --- CRITICAL: CORS Configuration ---
# This allows your React app (running on netlify.app) to talk to this backend.
origins = [
    "http://localhost:5173",          # Your local React dev server
    "https://shiv2503portfolio.netlify.app", # Your deployed portfolio
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------------------

# Define the request model: what the React app will send
class Query(BaseModel):
    question: str

# Define a simple root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"status": "AI Portfolio Agent API is running"}

# This is your main RAG endpoint
@app.post("/ask")
def ask_agent(query: Query):
    question = query.question
    print(f"Received question: {question}")

    # --- REPLACE THIS DUMMY LOGIC WITH YOUR ACTUAL RAG CALL ---
    # In the final version, this will be:
    # response_text, image_path = get_rag_response(question, text_collection, ...)
    if "siren" in question.lower():
        response_text = "SIRENs are amazing! They represent images as continuous functions. I built a full Streamlit demo for it."
        image_path = "data/images/siren_model_architecture_diagram.png" # Example path
    else:
        response_text = "That's a great question. I'm an AI assistant trained on Shivansh's projects. Ask me about PINNs, MLOps, or SIRENs."
        image_path = None
    # -----------------------------------------------------------

    return {"answer": response_text, "image": image_path}