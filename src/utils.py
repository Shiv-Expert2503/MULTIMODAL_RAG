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
        You are Shivansh's personal AI Career Agent, named AI Expert. 
        Your primary goal is to showcase Shivansh's skills and project experience to recruiters in a positive and professional light.
        Adopt a confident and proactive persona. Synthesize information from the provided context to form compelling arguments.
        When discussing a project that has visual aids (like plots or diagrams), you MUST refer to them by their full filenames as mentioned in the context (e.g., "pinns_comparative_total_loss.png").

        CONTEXT:
        {context_for_llm}

        USER'S QUESTION:
        {query}
        
        YOUR PROFESSIONAL RESPONSE:
        """
        response_text = llm.invoke(prompt).content
        logging.info(f"Generated Response: '{response_text}'")
        return response_text
    
    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error
    
def extract_image_paths(response_text, base_url="http://localhost:8000"):
    """
    Finds all image filenames in the text, checks if they exist locally,
    and returns a list of full, web-accessible URLs for the frontend.
    """
    image_pattern = r'([a-zA-Z0-9_\-]+\.(?:png|jpg|jpeg))\b'
    image_urls = []
    
    # The local path to the image directory
    local_image_dir = "data/images"

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