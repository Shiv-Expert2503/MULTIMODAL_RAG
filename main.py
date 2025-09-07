import sys
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.logger import logging
from src.exception import CustomException
from src.utils import load_resources, get_rag_response_as_text, extract_image_paths, route_query, get_rag_response_as_tree

app = FastAPI()
app.mount("/images", StaticFiles(directory="data/images"), name="images")

try:
    text_model, llm, text_collection = load_resources()
    logging.info("--- AI Resources loaded successfully on startup ---")
except Exception as e:
    logging.error(CustomException(e, sys))
    raise RuntimeError("Could not load AI models or database. Check logs.") from e

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in allowed_origins_str.split(',')]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def read_root():
    return {"status": "AI Portfolio Agent is running"}

class Query(BaseModel):
    question: str
    chat_history: list = [] 

@app.post("/ask")
async def ask_agent(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        chat_history = data.get("chat_history", [])

        if not question:
            return {"error": "No question provided"}, 400

        topic = route_query(question, text_model)

        is_first_topic_mention = True
        if "Project" in topic: # Check if the topic is a project
            for message in reversed(chat_history): # Look through past messages
                if message.get("topic") == topic:
                    is_first_topic_mention = False
                    break
        else:
            is_first_topic_mention = False

        if is_first_topic_mention:
            # --- THIS BLOCK IS NOW UPGRADED WITH THE FALLBACK LOGIC ---
            logging.info(f"First mention of topic '{topic}'. Attempting to generate JSON tree response.")
            
            # 1. First, try to get the intelligent tree view
            tree_data = get_rag_response_as_tree(question, text_collection, text_model, llm)
            
            if tree_data:
                # 2. If we get a valid tree, send it to the frontend
                response = {"type": "tree", "data": tree_data, "topic": topic}
            else:
                # 3. If tree generation fails, FALLBACK to the standard text response
                logging.warning("Tree generation failed. Falling back to standard text response.")
                response_text = get_rag_response_as_text(question, chat_history, text_collection, text_model, llm)
                image_urls = extract_image_paths(response_text)
                response = {"type": "text", "answer": response_text, "images": image_urls, "topic": topic}
            # ----------------------------------------------------------------

        else:
            # It's a follow-up or a general query, so generate a standard text response
            logging.info(f"Follow-up or general query for topic '{topic}'. Generating text response.")
            response_text = get_rag_response_as_text(question, chat_history, text_collection, text_model, llm)
            image_urls = extract_image_paths(response_text)
            response = {
                "type": "text",
                "answer": response_text,
                "images": image_urls,
                "topic": topic
            }
            
        return response

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        return {"error": str(error)}, 500