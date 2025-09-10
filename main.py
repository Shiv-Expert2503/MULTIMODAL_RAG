import asyncio
import json
import sys
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.logger import logging
from src.exception import CustomException
from src.utils import (
    load_resources,
    Router,
    TopicRegistry,
    get_rag_response_as_text,
    get_rag_response_as_tree,
    extract_image_paths,
    enrich_text_with_markdown_images
)

# ----------------------------
# App setup
# ----------------------------
app = FastAPI()
app.mount("/images", StaticFiles(directory="data/images"), name="images")

# ----------------------------
# Load AI resources
# ----------------------------
try:
    text_model, llm, text_collection, router = load_resources()
    logging.info("--- AI Resources & Router loaded successfully on startup ---")
except Exception as e:
    logging.error(CustomException(e, sys))
    raise RuntimeError("Could not load AI models or database. Check logs.") from e

# ----------------------------
# Global components
# ----------------------------
topic_registry = TopicRegistry(maxlen=6)
query_queue = asyncio.Queue()

# ----------------------------
# CORS setup
# ----------------------------
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "AI Portfolio Agent is running"}


# ----------------------------
# Request model
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []
    user_id: str = "default"  # extendable for multi-user


# ----------------------------
# Worker
# ----------------------------
async def worker():
    while True:
        user_id, question, chat_history, future = await query_queue.get()

        try:
            # Step 1: Route the query
            decision = router.route_query(question, chat_history)
            topic = decision.get("matched_topic", "general_query")
            state = topic_registry.get_state(topic)

            logging.info(f"User: {user_id} | Topic: {topic} | State: {state} | Decision: {json.dumps(decision, indent=2)}")
            image_urls = [] 
            # Step 2: Apply topic registry logic
            if decision["matched_topic"] == "general_query":
                # General query → bypass cache
                response_type = "text"
                final_answer = get_rag_response_as_text(
                    question, chat_history, text_collection, text_model, llm
                )
                image_urls = extract_image_paths(final_answer)
                final_answer = enrich_text_with_markdown_images(final_answer, image_urls) 

            else:
                if state == "NEW":
                    # First mention → JSON tree
                    response_type = "tree"
                    tree_data = get_rag_response_as_tree(
                        question, text_collection, text_model, llm
                    )
                    if tree_data:
                        final_answer = tree_data
                    else:
                        logging.warning("Tree generation failed. Falling back to text.")
                        response_type = "text"
                        final_answer = get_rag_response_as_text(
                            question, chat_history, text_collection, text_model, llm
                        )
                        image_urls = extract_image_paths(final_answer)
                        final_answer = enrich_text_with_markdown_images(final_answer, image_urls) 
                    topic_registry.set_state(topic, "FOLLOWUP")
                else:
                    # Follow-up → normal text
                    response_type = "text"
                    final_answer = get_rag_response_as_text(
                        question, chat_history, text_collection, text_model, llm
                    )

                    image_urls = extract_image_paths(final_answer)
                    final_answer = enrich_text_with_markdown_images(final_answer, image_urls) 

                # Store in topic registry
                topic_registry.push_message(
                    topic,
                    {
                        "question": question,
                        "answer": final_answer,
                        "type": response_type,
                    },
                )

            # Step 3: Build response
            result = {
                "type": response_type,
                "answer": final_answer if response_type == "text" else None,
                "data": final_answer if response_type == "tree" else None,
                "images": image_urls,
                "topic": topic,
                "_routing_debug": decision
            }
            future.set_result(result)

        except Exception as e:
            future.set_exception(CustomException(e, sys))
        finally:
            query_queue.task_done()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())


# ----------------------------
# API endpoint
# ----------------------------
@app.post("/ask")
async def ask_agent(req: QueryRequest):
    future = asyncio.get_event_loop().create_future()
    await query_queue.put((req.user_id, req.question, req.chat_history, future))
    result = await future
    safe_result = {
        "type": result.get("type", "text"),            # default: text
        "answer": result.get("answer", ""),            # default: empty string
        "images": result.get("images", []),            # default: []
        "topic": result.get("topic", None),            # default: None
        "data": result.get("data", None),              # used only for tree JSON
        "_routing_debug": result.get("_routing_debug", {})  # debug info
    }

    return safe_result

