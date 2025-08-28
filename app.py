# app.py

import streamlit as st
import os
import chromadb
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer


# Use Streamlit's caching to load models and DB only once
@st.cache_resource
def load_resources():
    """
    Loads all the necessary models and the ChromaDB client.
    """
    print("--- (Re)loading all resources ---")
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


def get_rag_response(query, text_collection, image_collection, text_model, image_model, llm):
    """
    Takes a user query and returns the RAG-generated response and any relevant image.
    """
    # 1. Multi-modal Retrieval
    # Embed the query for both text and image search
    text_embedding = text_model.embed_query(query)
    image_embedding = image_model.encode(query).tolist()

    # Query the collections
    text_results = text_collection.query(query_embeddings=[text_embedding], n_results=3)
    image_results = image_collection.query(query_embeddings=[image_embedding], n_results=1)

    # 2. Context Formulation
    retrieved_texts = "\n\n".join(text_results['documents'][0])
    retrieved_image_path = image_results['documents'][0][0]

    context = f"""
    Text Context:
    {retrieved_texts}

    Image Context:
    The user's query might be related to the image located at the following path: {retrieved_image_path}
    """

    # 3. Prompt Engineering
    prompt = f"""
    You are Shivansh's personal AI assistant. Your name is Gemini. 
    Your goal is to answer questions about his skills, projects, and background based ONLY on the context provided.
    Be professional, helpful, and concise. If the answer is not in the context, say "I don't have enough information to answer that question."

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """

    # 4. Generation
    response_text = llm.invoke(prompt).content
    
    # Check if the image seems relevant to the answer
    # (A simple check for now, can be improved later)
    final_image_path = None
    if any(keyword in response_text.lower() for keyword in ["image", "plot", "diagram", "figure", "demo", "snapshot", "ui", "interface"]):
        final_image_path = retrieved_image_path

    return response_text, final_image_path



# --- Streamlit UI Setup ---

st.set_page_config(layout="wide")
st.title("🤖 Shivansh's AI Portfolio Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm an AI assistant trained on Shivansh's projects. How can I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"])

# Accept user input
if prompt := st.chat_input("Ask me about Shivansh's projects..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, image_path = get_rag_response(prompt, text_collection, image_collection, text_model, image_model, llm)
            st.markdown(response_text)
            
            bot_message = {"role": "assistant", "content": response_text}
            
            if image_path and os.path.exists(image_path):
                st.image(image_path)
                bot_message["image"] = image_path
            
            st.session_state.messages.append(bot_message)