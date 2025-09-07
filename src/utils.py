import sys
import os
import re
import logging
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from src.exception import CustomException


def load_resources():
    logging.info("--- (Re)loading all cached resources ---")
    load_dotenv()
    text_embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    client = chromadb.PersistentClient(path="./chroma_db")
    text_collection = client.get_collection("portfolio_text")
    return text_embedding_model, llm, text_collection


def get_rag_response(query, text_collection, text_model, llm):
    try:
        logging.info(f"Starting RAG response generation for query: '{query}'")
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

        Use the following context to answer:

        CONTEXT:
        {context_for_llm}

        USER QUESTION:
        {query}

        YOUR RESPONSE:

        """
        response_text = llm.invoke(prompt).content
        logging.info(f"Generated Response: '{response_text}'")
        return response_text
    
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error
    
def extract_image_paths(response_text):
    """
    Finds all image filenames in the text, checks if they exist locally,
    and returns a list of full, web-accessible URLs for the frontend.
    """
    image_pattern = r'([a-zA-Z0-9_\-]+\.(?:png|jpg|jpeg))\b'
    image_urls = []
    
    # The local path to the image directory
    local_image_dir = "data/images"
    # If not found than use local host as base url for backend
    base_url = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")
    for match in re.finditer(image_pattern, response_text):
        image_file = match.group(1).strip()
        

        local_image_path = os.path.join(local_image_dir, image_file)
        if os.path.exists(local_image_path):
            image_url = f"{base_url}/images/{image_file}"
            image_urls.append(image_url)
            logging.info(f"Found and created URL for image: {image_url}")
        else:
            # If the file doesn't exist, just log it. Don't send a broken link.
            logging.warning(f"Image '{image_file}' mentioned in text but not found locally.")
            
    return image_urls