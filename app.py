import streamlit as st
import sys 
from src.utils import load_resources, get_rag_response, render_text_with_images
from src.logger import logging
from src.exception import CustomException


def main():
    """
    Main function to run the Streamlit app.
    We wrap the entire logic in a try-except block for robust error handling.
    """
    try:

        text_model, llm, text_collection = load_resources()

        st.set_page_config(layout="wide")
        st.title("🤖 Shivansh's AI Portfolio Assistant")
        st.info("Ask me a question about Shivansh's projects to get started.")

        if prompt := st.chat_input("Ask me about Shivansh's projects..."):
            logging.info(f"User Query: '{prompt}'")
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = get_rag_response(prompt, text_collection, text_model, llm)
                    render_text_with_images(response_text)

    except Exception as e:
        error = CustomException(e, sys)
        logging.error(error)
        st.error("An error occurred. Please check the logs for details.")


if __name__ == "__main__":
    main()