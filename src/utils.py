# import sys
# import os
# import re
# import chromadb
# import shutil
# import json
# from dotenv import load_dotenv
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from src.exception import CustomException
# from src.logger import logging

# TOPICS = [
#     "Physics-Informed Neural Networks (PINNs) Project",
#     "pinns project", 
#     "siren project",
#     "SIRENs (Sinusoidal Representation Networks) Project",
#     "Real-Time License Plate Anonymizer Project",
#     "Real-Time License Plate Blur Project",
#     "Sensor Fault Detection MLOps Project",
#     "General questions about Shivansh Project"
# ]

# def load_resources():
#     logging.info("--- Loading AI resources ---")
#     load_dotenv()

#     source_db_path = "./chroma_db"
#     writable_db_path = "./chroma_db" 

#     # If the writable path doesn't exist, copy the pre-built DB from the read-only location
#     if not os.path.exists(writable_db_path):
#         logging.info(f"Copying database from {source_db_path} to {writable_db_path}")
#         shutil.copytree(source_db_path, writable_db_path)

#     text_embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

#     client = chromadb.PersistentClient(path=writable_db_path)

#     text_collection = client.get_collection("portfolio_text")
#     return text_embedding_model, llm, text_collection

# def get_rag_response_as_text(query, chat_history, text_collection, text_model, llm):
#     try:
#         logging.info(f"Starting RAG response generation for query: '{query}' with history.")
        
#         formatted_history = "\n".join([f"{msg.get('type', 'user')}: {msg.get('content', '')}" for msg in chat_history])

#         text_embedding = text_model.embed_query(query)
#         text_results = text_collection.query(query_embeddings=[text_embedding], n_results=5)
#         context_for_llm = "\n\n".join(text_results['documents'][0])
#         logging.info(f"Retrieved Context (first 200 chars): '{context_for_llm[:200]}...'")

#         prompt = f"""
#         You are Shivansh's personal AI Career Agent, “AI Expert”.
#         Your goal: answer crisply, professionally, and help recruiters/investors skim fast.

#         STYLE & FORMAT (must follow):
#         - Default to brevity. Prefer a 1-2 line intro, then bullet points. Use tables when helpful.
#         - Never reveal sources or doc structure. Do NOT say: “in the context”, “under the heading”, “the document says”, or anything that leaks the knowledge base.
#         - Links: when asked for resources or code, return ONLY a short bulleted list of Markdown links (with full https:// URLs) and nothing else.
#         - Images: NEVER say “I cannot display images”. Only include images when (a) the user explicitly asks for images/snapshots, OR (b) the query is a broad “explain the project” where one primary visual improves clarity. Otherwise, omit images to reduce cost.
#         - If the question is broad, end with ONE short follow-up offer (e.g., “Want the tech stack or the pipeline summary?”). For specific questions, don't add follow-ups.

#         INTENT → OUTPUT RULES:
#         Classify the user's query into one of:
#         1) OVERVIEW: broad “explain/tell me about …”
#         - 1-2 line intro + 3-6 bullets max. No walls of text.
#         - Optional single image if truly helpful.
#         - Optional one follow-up offer.
#         2) DETAIL_TECHSTACK / DETAIL_PIPELINE / DETAIL_RESULTS:
#         - Use a heading and bullets or a table. No intro paragraph beyond 1 line.
#         - No images unless explicitly requested.
#         - No follow-up unless natural.
#         3) RESOURCES_ONLY (e.g., “give me the resources/source code/docs”):
#         - Output ONLY a short bulleted list of Markdown links. No images. No extra prose.
#         4) SHOW_IMAGE_ONLY (e.g., “show me only the image of …”):
#         - Output ONLY the Markdown image tag `![alt](filename.ext)` for the exact file(s) referenced by the context. No extra text.

#         ADDITIONAL GUARDRAILS:
#         - If the user asks for “code/resources/docs”, don't summarize—just give the links.
#         - If a table is available or beneficial, render it in Markdown.
#         - Never invent links; only use those present in the context. If none exist, say briefly: “No public link provided.”

#         Use the following CHAT HISTORY to understand the context of the current question:

#         CHAT HISTORY:
#         {formatted_history}

#         CONTEXT:
#         {context_for_llm}

#         USER'S QUESTION:
#         {query}
        
#         YOUR PROFESSIONAL RESPONSE:
#         """
#         response_text = llm.invoke(prompt).content
#         logging.info(f"Generated Response: '{response_text}'")
#         return response_text
    
#     except Exception as e:
#         error = CustomException(e, sys)
#         logging.error(error)
#         raise error
     
# def extract_image_paths(response_text):
#     """
#     Finds all image filenames in the text, checks if they exist locally,
#     and returns a list of full, web-accessible URLs for the frontend.
#     """
#     image_pattern = r'([a-zA-Z0-9_\-]+\.(?:png|jpg|jpeg))\b'
#     image_urls = []
    
#     # The local path to the image directory
#     local_image_dir = "data/images"
#     # If not found than use local host as base url for backend
#     base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
#     for match in re.finditer(image_pattern, response_text):
#         image_file = match.group(1).strip()
        

#         local_image_path = os.path.join(local_image_dir, image_file)
#         if os.path.exists(local_image_path):
#             image_url = f"{base_url}/images/{image_file}"
#             image_urls.append(image_url)
#             logging.info(f"Found and created URL for image: {image_url}")
#         else:
#             # If the file doesn't exist, just log it. Don't send a broken link.
#             logging.warning(f"Image '{image_file}' mentioned in text but not found locally.")
            
#     return image_urls


# def route_query(query, text_embedding_model):
#     """
#     Uses semantic similarity to classify a user's query into one of the predefined TOPICS.
#     """
#     logging.info(f"Routing query: '{query}'")
    
#     query_embedding = text_embedding_model.embed_query(query)
    
#     topic_embeddings = text_embedding_model.embed_documents(TOPICS)

#     similarities = cosine_similarity(
#         [query_embedding],
#         topic_embeddings
#     )[0] # Get the first row of the similarity matrix
    
#     # Find the topic with the highest similarity score
#     most_similar_index = np.argmax(similarities)
#     highest_similarity = similarities[most_similar_index]
    
#     # a threshold for a confident match
#     threshold = 0.3
    
#     if highest_similarity > threshold:
#         matched_topic = TOPICS[most_similar_index]
#         logging.info(f"Query routed to topic: '{matched_topic}' with score {highest_similarity:.2f}")
#         return matched_topic
#     else:
#         logging.info(f"Query did not meet similarity threshold. Highest score: {highest_similarity:.2f}")
#         return "general_query" # default category for unmatched queries
    

# def get_rag_response_as_tree(query, text_collection, text_model, llm):
#     """
#     Performs a RAG query and asks the LLM to generate a JSON tree structure.
#     Includes a retry mechanism and returns None if JSON parsing fails.
#     """
#     logging.info("Attempting to generate JSON tree response.")
    
#     # Step 1: Retrieve context, same as the text function
#     text_embedding = text_model.embed_query(query)
#     text_results = text_collection.query(query_embeddings=[text_embedding], n_results=5)
#     context_for_llm = "\n\n".join(text_results['documents'][0])

#     # Step 2: The new JSON-specific prompt
#     prompt = f"""
#     You are a knowledge architect. Your job is to analyze the provided context about one of Shivansh's projects and break it down into its core components.
#     You must respond ONLY with a single, valid JSON object and nothing else.

#     The JSON object must have this exact structure:
#     {{
#       "root_node": "Name of the Project",
#       "child_nodes": [
#         {{ "title": "Definition", "summary": "A concise, one-sentence summary of the project." }},
#         {{ "title": "Problem Solved", "summary": "A brief description of the problem the project addresses." }},
#         {{ "title": "Tech Stack", "summary": "A list of the key technologies used." }},
#         {{ "title": "Key Visuals", "summary": "A reference to the available demo images or videos." }}
#       ],
#       "follow_up_question": "Which of these topics would you like me to explain in more detail?"
#     }}
    
#     Do not include any text, greetings, or markdown formatting before or after the JSON object.

#     CONTEXT:
#     {context_for_llm}

#     USER'S QUESTION:
#     {query}

#     VALID JSON RESPONSE:
#     """

#     for i in range(2): # Try up to 2 times tocheck is json format or not
#         try:
#             response_str = llm.invoke(prompt).content
#             logging.info(f"LLM attempt {i+1} raw output: {response_str}")
            
#             json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
#             if json_match:
#                 json_str = json_match.group(0)

#                 parsed_json = json.loads(json_str)
#                 logging.info(f"Successfully parsed JSON from LLM response {parsed_json}.")
#                 return parsed_json 
#             else:
#                 logging.warning(f"Attempt {i+1}: No JSON object found in the response.")

#         except json.JSONDecodeError as e:
#             logging.warning(f"Attempt {i+1}: Failed to parse JSON. Error: {e}. Retrying...")
#         except Exception as e:
#             raise CustomException(e, sys)

#     logging.error("Failed to generate valid JSON after multiple attempts.")
#     return None # Return None if all attempts fail


# src/utils.py
import sys
import os
import re
import chromadb
import shutil
import json
import pickle
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.exception import CustomException
from src.logger import logging
from collections import deque
from typing import List, Dict, Any, Tuple

# ----------------------------
# Topics (canonical) - keep as before
# ----------------------------
TOPICS = [
    "Physics-Informed Neural Networks (PINNs) Project",
    "pinns project",
    "siren project",
    "SIRENs (Sinusoidal Representation Networks) Project",
    "Real-Time License Plate Anonymizer Project",
    "Real-Time License Plate Blur Project",
    "Sensor Fault Detection MLOps Project",
    "General questions about Shivansh Project"
]

# ----------------------------
# Router + Memory Classes & Config
# ----------------------------
EMBED_CACHE_PATH = "./.topic_embeddings_cache.pkl"

# Tunable thresholds
MIN_SIM = 0.50         # absolute minimum similarity to accept
MIN_SIM_LOW = 0.36     # lower baseline that can be accepted via gap rule
GAP_THRESHOLD = 0.12   # required gap between top1 and top2 to accept borderline
MEMORY_WINDOW = 6      # last N messages kept for rewrite heuristics

PRONOUN_PATTERNS = re.compile(
    r"\b(his|her|their|him|them|this|that|these|those)\b", flags=re.IGNORECASE
)

def _build_topic_keyword_map(topics: List[str]) -> Dict[str, List[str]]:
    kw_map = {}
    for t in topics:
        words = re.findall(r"[A-Za-z0-9]+", t.lower())
        words = [w for w in words if len(w) >= 3]
        kw_map[t] = list(dict.fromkeys(words))
    return kw_map

TOPIC_KEYWORD_MAP = _build_topic_keyword_map(TOPICS)


class ConversationMemory:
    """
    Lightweight conversation memory built from chat_history (list of message dicts).
    Stores last N messages and provides utility functions for coref-like rewriting.
    """
    def __init__(self, window_size: int = MEMORY_WINDOW):
        self.window_size = window_size
        self._deque = deque(maxlen=window_size)

    def add_message(self, role: str, content: str):
        self._deque.append({"role": role, "content": content})

    def add_from_chat_history(self, chat_history: List[Dict[str, Any]]):
        """
        chat_history expected format: list of dicts with 'type'/'role' and 'content' keys.
        We'll accept variety but try to use message.get('content').
        """
        # only keep the last `window_size` messages
        recent = chat_history[-self.window_size:] if chat_history else []
        for m in recent:
            content = m.get("content") if isinstance(m, dict) else str(m)
            role = m.get("type", m.get("role", "user")) if isinstance(m, dict) else "user"
            self.add_message(role, content)

    def get_recent_text(self) -> str:
        return "\n".join([m["content"] for m in self._deque])

    def debug_print(self):
        logging.info("ConversationMemory dump (most recent last):")
        for i, m in enumerate(self._deque):
            logging.info("  [%d] %s: %s", i, m["role"], (m["content"][:200] + "...") if len(m["content"])>200 else m["content"])


class Router:
    """
    Router object encapsulates topic embeddings, similarity routing logic,
    rewrite heuristics using ConversationMemory, and debug logging.
    """
    def __init__(self, text_embedding_model, topics: List[str] = TOPICS, cache_path: str = EMBED_CACHE_PATH):
        self.topics = topics
        self.text_embedding_model = text_embedding_model
        self.cache_path = cache_path
        self.topic_keyword_map = TOPIC_KEYWORD_MAP

        # Precompute or load cached topic embeddings (normalized)
        self.topic_embeddings = self._load_or_compute_topic_embeddings()

    def _load_or_compute_topic_embeddings(self):
        # Attempt to load cache
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("topics") == self.topics and "embeddings" in cache:
                    logging.info("Loaded cached topic embeddings.")
                    return cache["embeddings"]
                else:
                    logging.info("Topic list changed or cache invalid - recomputing embeddings.")
        except Exception as e:
            logging.warning("Failed to load cache: %s", e)

        # Compute embeddings live
        logging.info("Computing topic embeddings...")
        try:
            # Use the provided text_embedding_model, keep normalized vectors
            embeddings = np.array(self.text_embedding_model.embed_documents(self.topics), dtype=float)
            # normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms
            # cache
            try:
                with open(self.cache_path, "wb") as f:
                    pickle.dump({"topics": self.topics, "embeddings": embeddings}, f)
                logging.info("Topic embeddings cached.")
            except Exception as e:
                logging.warning("Unable to write topic embedding cache: %s", e)
            return embeddings
        except Exception as e:
            logging.error("Failed to compute topic embeddings: %s", e)
            raise CustomException(e, sys)

    def _embed_query(self, text: str) -> np.ndarray:
        vec = np.array(self.text_embedding_model.embed_query(text), dtype=float)
        # normalize
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec

    def _find_last_mentioned_topic(self, memory: ConversationMemory) -> Tuple[str, float]:
        """Simple heuristic: look for topic keyword presence in recent messages."""
        recent_msgs = list(memory._deque)[::-1]  # last-first
        for idx, m in enumerate(recent_msgs):
            txt = m["content"].lower()
            for topic, keywords in self.topic_keyword_map.items():
                for kw in keywords:
                    if kw in txt:
                        score = 1.0 / (1 + idx)  # nearer messages => higher score
                        return topic, score
        return None, 0.0

    def _find_person_in_memory(self, memory: ConversationMemory) -> Tuple[str, float]:
        """Heuristic to find explicit mention of 'Shivansh' or capitalized tokens."""
        recent = list(memory._deque)[::-1]
        for idx, m in enumerate(recent):
            txt = m["content"]
            if "shivansh" in txt.lower():
                return "Shivansh", 1.0 / (1 + idx)
            caps = re.findall(r"\b[A-Z][a-z]{2,}\b", txt)
            if caps:
                return caps[0], 1.0 / (1 + idx)
        return None, 0.0

    def rewrite_query_with_memory(self, query: str, memory: ConversationMemory) -> Tuple[str, bool, str]:
        """
        If the query contains pronouns (his/their/that/this/etc.), attempt to rewrite using memory:
        - prefer last-mentioned topic if found
        - else prefer person mention (e.g., 'Shivansh')
        Returns (rewritten_query, changed_flag, reason)
        """
        if not PRONOUN_PATTERNS.search(query):
            return query, False, "no_pronoun"

        # prefer topic rewrite
        topic, tscore = self._find_last_mentioned_topic(memory)
        if topic:
            # replace first pronoun token with topic short name
            short = topic
            rewritten = re.sub(PRONOUN_PATTERNS, short, query, count=1)
            return rewritten, True, f"rewritten_topic:{topic}"

        # fallback to person replacement
        person, pscore = self._find_person_in_memory(memory)
        if person:
            replacement = person + "'s"
            rewritten = re.sub(PRONOUN_PATTERNS, replacement, query, count=1)
            return rewritten, True, f"rewritten_person:{person}"

        # nothing found
        return query, False, "no_candidate_found"

    def route_query(self, query: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main routing method. Returns a dict with decision metadata:
        {
          "matched_topic": str,
          "top3": [(topic, score), ...],
          "highest_similarity": float,
          "second_similarity": float,
          "gap": float,
          "rewritten_query": str,
          "rewrite_reason": str,
          "decision_reason": str
        }
        """
        logging.info("Routing query: '%s'", query)

        # Build memory from chat_history
        memory = ConversationMemory(window_size=MEMORY_WINDOW)
        memory.add_from_chat_history(chat_history)
        memory.debug_print()

        # Try rewrite
        rewritten_query, changed, rewrite_reason = self.rewrite_query_with_memory(query, memory)
        if changed:
            logging.info("Query rewritten for coref: '%s' -> '%s' (reason=%s)", query, rewritten_query, rewrite_reason)

        # Embed
        try:
            qvec = self._embed_query(rewritten_query)
        except Exception as e:
            logging.error("Embedding query failed: %s", e)
            raise CustomException(e, sys)

        # Compute similarities against precomputed topic embeddings
        sims = cosine_similarity([qvec], self.topic_embeddings)[0]
        sims = np.array(sims, dtype=float)

        # Get top-3 candidates
        top_indices = np.argsort(sims)[::-1][:3]
        top3 = [(self.topics[i], float(sims[i])) for i in top_indices]

        logging.info("Top-3 candidates for query '%s':", query)
        for rank, (t, s) in enumerate(top3, start=1):
            logging.info("  %d) %s (score=%.4f)", rank, t, s)

        highest_idx = top_indices[0]
        second_idx = top_indices[1] if len(top_indices) > 1 else None
        highest_sim = float(sims[highest_idx])
        second_sim = float(sims[second_idx]) if second_idx is not None else 0.0
        gap = highest_sim - second_sim

        # Decision logic: gap-based acceptance
        decision_reason = ""
        matched_topic = "general_query"

        if highest_sim >= MIN_SIM:
            matched_topic = self.topics[highest_idx]
            decision_reason = f"accepted_by_min_sim ({highest_sim:.3f})"
        elif highest_sim >= MIN_SIM_LOW and gap >= GAP_THRESHOLD:
            matched_topic = self.topics[highest_idx]
            decision_reason = f"accepted_by_gap (top={highest_sim:.3f}, gap={gap:.3f})"
        else:
            matched_topic = "general_query"
            decision_reason = f"rejected (top={highest_sim:.3f}, gap={gap:.3f})"

        logging.info("Routing decision: %s -> %s", decision_reason, matched_topic)

        return {
            "matched_topic": matched_topic,
            "top3": top3,
            "highest_similarity": highest_sim,
            "second_similarity": second_sim,
            "gap": gap,
            "rewritten_query": rewritten_query,
            "rewrite_reason": rewrite_reason,
            "decision_reason": decision_reason,
        }

# ----------------------------
# Existing functions retained/updated (RAG / helpers)
# ----------------------------
def load_resources():
    """
    Loads embeddings, LLM, chroma DB collection and returns them plus a Router object.
    """
    try:
        logging.info("--- Loading AI resources ---")
        load_dotenv()

        source_db_path = "./chroma_db"
        writable_db_path = "./chroma_db"

        if not os.path.exists(writable_db_path):
            logging.info(f"Copying database from {source_db_path} to {writable_db_path}")
            shutil.copytree(source_db_path, writable_db_path)

        text_embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

        client = chromadb.PersistentClient(path=writable_db_path)
        text_collection = client.get_collection("portfolio_text")

        # create router
        router = Router(text_embedding_model, topics=TOPICS)

        logging.info("Resources loaded successfully.")
        return text_embedding_model, llm, text_collection, router

    except Exception as e:
        raise CustomException(e, sys)


def get_rag_response_as_text(query, chat_history, text_collection, text_model, llm):
    try:
        logging.info(f"Starting RAG response generation for query: '{query}' with history.")
        formatted_history = "\n".join([f"{msg.get('type', 'user')}: {msg.get('content', '')}" for msg in chat_history])

        text_embedding = text_model.embed_query(query)
        text_results = text_collection.query(query_embeddings=[text_embedding], n_results=5)
        context_for_llm = "\n\n".join(text_results['documents'][0])
        logging.info(f"Retrieved Context (first 200 chars): '{context_for_llm[:200]}...'")

        prompt = f"""
        You are Shivansh's personal AI Career Agent, “AI Expert”.
        Your goal: answer crisply, professionally, and help recruiters/investors skim fast.

        STYLE & FORMAT (must follow):
        - Default to brevity. Prefer a 1-2 line intro, then bullet points. Use tables when helpful.
        - Never reveal sources or doc structure. Do NOT say: “in the context”, “under the heading”, “the document says”, or anything that leaks the knowledge base.
        - Links: when asked for resources or code, return ONLY a short bulleted list of Markdown links (with full https:// URLs) and nothing else.
        - Images: NEVER say “I cannot display images”. Only include images when (a) the user explicitly asks for images/snapshots, OR (b) the query is a broad “explain the project” where one primary visual improves clarity. Otherwise, omit images to reduce cost.
        - If the question is broad, end with ONE short follow-up offer (e.g., “Want the tech stack or the pipeline summary?”). For specific questions, don't add follow-ups.

        INTENT → OUTPUT RULES:
        Classify the user's query into one of:
        1) OVERVIEW: broad “explain/tell me about …”
        - 1-2 line intro + 3-6 bullets max. No walls of text.
        - Optional single image if truly helpful.
        - Optional one follow-up offer.
        2) DETAIL_TECHSTACK / DETAIL_PIPELINE / DETAIL_RESULTS:
        - Use a heading and bullets or a table. No intro paragraph beyond 1 line.
        - No images unless explicitly requested.
        - No follow-up unless natural.
        3) RESOURCES_ONLY (e.g., “give me the resources/source code/docs”):
        - Output ONLY a short bulleted list of Markdown links. No images. No extra prose.
        4) SHOW_IMAGE_ONLY (e.g., “show me only the image of …”):
        - Output ONLY the Markdown image tag `![alt](filename.ext)` for the exact file(s) referenced by the context. No extra text.

        ADDITIONAL GUARDRAILS:
        - If the user asks for “code/resources/docs”, don't summarize—just give the links.
        - If a table is available or beneficial, render it in Markdown.
        - Never invent links; only use those present in the context. If none exist, say briefly: “No public link provided.”

        Use the following CHAT HISTORY to understand the context of the current question:

        CHAT HISTORY:
        {formatted_history}

        CONTEXT:
        {context_for_llm}

        USER'S QUESTION:
        {query}
        
        YOUR PROFESSIONAL RESPONSE:
        """
        # call LLM
        response_text = llm.invoke(prompt).content
        logging.info(f"Generated Response: '{response_text}'")
        return response_text

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error


def extract_image_paths(response_text):
    image_pattern = r'([a-zA-Z0-9_\-]+\.(?:png|jpg|jpeg))\b'
    image_urls = []
    local_image_dir = "data/images"
    base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    for match in re.finditer(image_pattern, response_text):
        image_file = match.group(1).strip()
        local_image_path = os.path.join(local_image_dir, image_file)
        if os.path.exists(local_image_path):
            image_url = f"{base_url}/images/{image_file}"
            image_urls.append(image_url)
            logging.info(f"Found and created URL for image: {image_url}")
        else:
            logging.warning(f"Image '{image_file}' mentioned in text but not found locally.")
    return image_urls


def get_rag_response_as_tree(query, text_collection, text_model, llm):
    """
    Retained exactly as before - attempts to get JSON tree.
    """
    logging.info("Attempting to generate JSON tree response.")
    try:
        text_embedding = text_model.embed_query(query)
        text_results = text_collection.query(query_embeddings=[text_embedding], n_results=5)
        context_for_llm = "\n\n".join(text_results['documents'][0])

        prompt = f"""
        You are a knowledge architect. Your job is to analyze the provided context about one of Shivansh's projects and break it down into its core components.
        You must respond ONLY with a single, valid JSON object and nothing else.

        The JSON object must have this exact structure:
        {{
        "root_node": "Name of the Project",
        "child_nodes": [
            {{ "title": "Definition", "summary": "A concise, one-sentence summary of the project." }},
            {{ "title": "Problem Solved", "summary": "A brief description of the problem the project addresses." }},
            {{ "title": "Tech Stack", "summary": "A list of the key technologies used." }},
            {{ "title": "Key Visuals", "summary": "A reference to the available demo images or videos." }}
        ],
        "follow_up_question": "Which of these topics would you like me to explain in more detail?"
        }}
        
        Do not include any text, greetings, or markdown formatting before or after the JSON object.

        CONTEXT:
        {context_for_llm}

        USER'S QUESTION:
        {query}

        VALID JSON RESPONSE:
        """

        for i in range(2):
            try:
                response_str = llm.invoke(prompt).content
                logging.info(f"LLM attempt {i+1} raw output: {response_str}")
                json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    logging.info(f"Successfully parsed JSON from LLM response {parsed_json}.")
                    return parsed_json
                else:
                    logging.warning(f"Attempt {i+1}: No JSON object found in the response.")
            except json.JSONDecodeError as e:
                logging.warning(f"Attempt {i+1}: Failed to parse JSON. Error: {e}. Retrying...")
            except Exception as e:
                raise CustomException(e, sys)

        logging.error("Failed to generate valid JSON after multiple attempts.")
        return None

    except Exception as e:
        raise CustomException(e, sys)
