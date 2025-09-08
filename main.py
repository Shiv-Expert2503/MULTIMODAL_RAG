# main.py
import sys
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.logger import logging
from src.exception import CustomException
from src.utils import load_resources, get_rag_response_as_text, extract_image_paths, get_rag_response_as_tree

app = FastAPI()
app.mount("/images", StaticFiles(directory="data/images"), name="images")

try:
    text_model, llm, text_collection, router = load_resources()
    logging.info("--- AI Resources & Router loaded successfully on startup ---")
except Exception as e:
    logging.error(CustomException(e, sys))
    raise RuntimeError("Could not load AI models or database. Check logs.") from e

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins or ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
        chat_history = data.get("chat_history") or []

        logging.info(f"[DEBUG] question={question!r}, chat_history={(chat_history)}")


        if not question:
            return {"error": "No question provided"}, 400

        # Use the router to decide topic (includes memory-based rewrite & top-3 logging)
        decision = router.route_query(question, chat_history)
        topic = decision.get("matched_topic", "general_query")

        # Determine if it's first mention of a project in this conversation
        is_first_topic_mention = True
        if "project" in topic.lower():
            # scan chat_history for previous mention of same topic
            for message in reversed(chat_history):
                # we expect message may include 'topic' field if your frontend sets it
                if message.get("topic") == topic:
                    is_first_topic_mention = False
                    break
            # if not found by explicit field, do a content scan to be safe
            if is_first_topic_mention:
                for message in reversed(chat_history):
                    if isinstance(message.get("content"), str) and topic.lower() in message.get("content", "").lower():
                        is_first_topic_mention = False
                        break
        else:
            is_first_topic_mention = False

        logging.info(f"Router decision metadata: {decision}")

        if is_first_topic_mention:
            logging.info(f"First mention of topic '{topic}'. Attempting JSON tree response.")
            tree_data = get_rag_response_as_tree(question, text_collection, text_model, llm)
            if tree_data:
                response = {"type": "tree", "data": tree_data, "topic": topic}
            else:
                logging.warning("Tree generation failed. Falling back to standard text response.")
                response_text = get_rag_response_as_text(question, chat_history, text_collection, text_model, llm)
                image_urls = extract_image_paths(response_text)
                response = {"type": "text", "answer": response_text, "images": image_urls, "topic": topic}
        else:
            logging.info(f"Follow-up or general query for topic '{topic}'. Generating text response.")
            response_text = get_rag_response_as_text(question, chat_history, text_collection, text_model, llm)
            image_urls = extract_image_paths(response_text)
            response = {
                "type": "text",
                "answer": response_text,
                "images": image_urls,
                "topic": topic
            }

        # Add routing metadata to response for debugging (optional)
        response["_routing_debug"] = {
            "matched_topic": decision.get("matched_topic"),
            "top3": decision.get("top3"),
            "highest_similarity": decision.get("highest_similarity"),
            "second_similarity": decision.get("second_similarity"),
            "gap": decision.get("gap"),
            "rewritten_query": decision.get("rewritten_query"),
            "rewrite_reason": decision.get("rewrite_reason"),
            "decision_reason": decision.get("decision_reason"),
        }

        return response

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        return {"error": str(error)}, 500
