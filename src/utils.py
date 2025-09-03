import streamlit as st
import os
import re
import sys
import logging
import chromadb
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from src.exception import CustomException

@st.cache_resource
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

def render_text_with_images(response_text, image_base_path="data/images"):
    try:
        image_pattern = r'([a-zA-Z0-9_\-]+\.(?:png|jpg|jpeg))\b'
        last_index = 0
        for match in re.finditer(image_pattern, response_text):
            text_part = response_text[last_index:match.start()]
            if text_part:
                st.markdown(text_part, unsafe_allow_html=True)
            
            logging.info(f"Texts Part for images: {text_part}")
            image_file = match.group(1)
            image_path = os.path.join(image_base_path, image_file)
            
            logging.info(f"Rendering image: {image_path}")

            if os.path.exists(image_path):
                st.image(image_path)
            else:
                logging.warning(f"Image mentioned in text but not found: {image_file}")
                st.markdown(f"`(Image not found: {image_file})`")
            last_index = match.end()
        
        remaining_text = response_text[last_index:]
        if remaining_text:
            st.markdown(remaining_text, unsafe_allow_html=True)

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        raise error
