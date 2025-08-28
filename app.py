# app.py

import streamlit as st
import os
import re
import logging
from logger_config import setup_logger # Import our new function
import chromadb
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

# --- SETUP THE LOGGER ---
# This will run once when the app starts
setup_logger()

# Use Streamlit's caching to load models and DB only once
@st.cache_resource
def load_resources():
    logging.info("--- (Re)loading all cached resources ---")
    load_dotenv()
    
    text_embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    image_embedding_model = SentenceTransformer('clip-ViT-B-32')
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    
    client = chromadb.PersistentClient(path="./chroma_db")
    text_collection = client.get_collection("portfolio_text")
    image_collection = client.get_collection("portfolio_images")
    
    return text_embedding_model, image_embedding_model, llm, text_collection, image_collection

# --- Load all resources at the start ---
text_model, image_model, llm, text_collection, image_collection = load_resources()

def get_rag_response(query, text_collection, image_collection, text_model, llm):
    logging.info(f"Starting RAG response generation for query: '{query}'")

    # Step 1: Text-First Retrieval
    text_embedding = text_model.embed_query(query)
    text_results = text_collection.query(query_embeddings=[text_embedding], n_results=4)
    retrieved_texts = text_results['documents'][0]
    
    # Step 2: Find Image References within the Retrieved Text
    final_image_path = None
    context_for_llm = ""
    image_pattern = r'\b([a-zA-Z0-9_-]+\.(?:png|jpg|jpeg))\b'
    
    for text_chunk in retrieved_texts:
        context_for_llm += text_chunk + "\n\n"
        match = re.search(image_pattern, text_chunk)
        if match and not final_image_path:
            image_filename = match.group(1)
            potential_path = os.path.join('data/images', image_filename)
            if os.path.exists(potential_path):
                final_image_path = potential_path

    logging.info(f"Retrieved Text Context (first 100 chars): '{context_for_llm[:100]}...'")
    if final_image_path:
        logging.info(f"Identified Image Path: {final_image_path}")
    else:
        logging.info("No relevant image path identified in the text context.")

    # Step 3: Prompt Engineering
    prompt = f"""
    You are Shivansh's personal AI Career Agent. Your name is Gemini. 
    Your primary goal is to showcase Shivansh's skills and project experience to recruiters in the most positive and professional light.
    You MUST adopt a confident and proactive persona. Do not be passive or say "I don't have enough information" unless the query is completely unrelated to Shivansh's career.
    Synthesize information from the provided context to form compelling arguments for why he is a good fit for AI engineering roles. You are allowed to make reasonable inferences based on the projects.

    CONTEXT:
    {context_for_llm}

    QUESTION:
    {query}

    COMPELLING AND CONFIDENT ANSWER:
    """

    # --- NEW: TOKEN COUNTING ---
    prompt_tokens = llm.get_num_tokens(prompt)
    logging.info(f"Prompt Token Count: {prompt_tokens}")
    # ---------------------------

    # Step 4: Generation
    response_text = llm.invoke(prompt).content
    logging.info(f"Generated Response: '{response_text}'")
    
    return response_text, final_image_path

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide")
st.title("🤖 Shivansh's AI Portfolio Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm an AI assistant trained on Shivansh's projects. How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"])

if prompt := st.chat_input("Ask me about Shivansh's projects..."):
    # --- NEW: LOGGING USER QUERY ---
    logging.info(f"User Query: '{prompt}'")
    # -------------------------------
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    is_download_request = any(keyword in prompt.lower() for keyword in ["get cv", "send resume", "pdf format", "download resume"])

    if is_download_request:
        # Handle the command directly
        with st.chat_message("assistant"):
            st.write("Of course. Here is Shivansh's resume in PDF format:")
            with open("data/pdfs/Resume.pdf", "rb") as pdf_file:
                st.download_button(
                    label="Download Resume (PDF)",
                    data=pdf_file,
                    file_name="Shivansh_Resume.pdf",
                    mime='application/octet-stream'
                )
        # Add the command handling to session state
        st.session_state.messages.append({"role": "assistant", "content": "Here is Shivansh's resume."})
    
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text, image_path = get_rag_response(prompt, text_collection, image_collection, text_model, llm)
                st.markdown(response_text)
                
                bot_message = {"role": "assistant", "content": response_text}
                
                if image_path and os.path.exists(image_path):
                    st.image(image_path)
                    bot_message["image"] = image_path
                
                st.session_state.messages.append(bot_message)