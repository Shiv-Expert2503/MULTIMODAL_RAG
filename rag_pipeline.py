import os
import chromadb
from PIL import Image
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

print("--- All libraries imported successfully ---")

def load_documents(directory_path='data'):
    """
    Loads all .md and .pdf files from the specified directory and its subdirectories.
    """
    print(f"--- Loading documents from {directory_path} ---")
    
    # Loader for Markdown files
    text_loader = DirectoryLoader(os.path.join(directory_path, 'text'), glob="**/*.md", show_progress=True)
    
    # Loader for PDF files
    pdf_loader = DirectoryLoader(os.path.join(directory_path, 'pdfs'), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    
    loaded_documents = text_loader.load() + pdf_loader.load()
    
    print(f"--- Loaded {len(loaded_documents)} documents successfully ---")
    return loaded_documents


def chunk_documents(documents):
    """
    Splits the loaded documents into smaller chunks.
    """
    print("--- Chunking documents ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"--- Created {len(chunked_documents)} chunks ---")
    return chunked_documents



def get_models_and_db_client():
    """
    Initializes and returns the embedding models and the ChromaDB client.
    """
    print("--- Initializing models and ChromaDB client ---")
    load_dotenv() # Load GOOGLE_API_KEY from .env file

    # Initialize Google Generative AI Embeddings model for text
    text_embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # Initialize Sentence Transformer model (CLIP) for images
    image_embedding_model = SentenceTransformer('clip-ViT-B-32')

    # Initialize ChromaDB client (saves to a local directory)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    print("--- Models and client initialized ---")
    return text_embedding_model, image_embedding_model, client


def embed_text_and_store(chunks, collection, embedding_model):
    """
    Generates embeddings for text chunks and stores them in the ChromaDB collection.
    """
    print(f"--- Embedding and storing {len(chunks)} text chunks ---")
    
    documents_to_store = [chunk.page_content for chunk in chunks]
    meatadatas_to_store = [chunk.metadata for chunk in chunks]
    ids_to_store = [f"text_{i}" for i in range(len(chunks))]

    # Generate embeddings for all text chunks in a single API call
    embeddings = embedding_model.embed_documents(documents_to_store)
    
    collection.add(
        embeddings=embeddings,
        documents=documents_to_store,
        metadatas=meatadatas_to_store,
        ids=ids_to_store
    )
    print("--- Text chunks embedded and stored successfully ---")

def embed_images_and_store(image_directory, collection, embedding_model):
    """
    Generates embeddings for images and stores them in the ChromaDB collection.
    """
    print(f"--- Embedding and storing images from {image_directory} ---")
    image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    image_paths_to_store = [os.path.join(image_directory, file) for file in image_files]
    ids_to_store = [f"image_{i}" for i in range(len(image_files))]
    
    # Generate embeddings locally using the CLIP model
    pil_images = [Image.open(path) for path in image_paths_to_store]
    embeddings = embedding_model.encode(pil_images)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=image_paths_to_store, # Store the file path as the "document"
        ids=ids_to_store
    )
    print("--- Images embedded and stored successfully ---")
# # --- GLOBAL INITIALIZATIONS ---
# print("--- Initializing models and ChromaDB client ---")
# load_dotenv() # Load environment variables from .env file

# # Initialize Google Generative AI Embeddings model for text
# text_embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# # Initialize Sentence Transformer model (CLIP) for images
# image_embedding_model = SentenceTransformer('clip-ViT-B-32')

# # Initialize ChromaDB client
# # This will create a local directory "./chroma_db" to store the database
# client = chromadb.PersistentClient(path="./chroma_db")

# # Get or create the collections (like tables in a database)
# text_collection = client.get_or_create_collection("portfolio_text")
# image_collection = client.get_or_create_collection("portfolio_images")

# print("--- Models and client initialized ---")


# def embed_text_and_store(chunks, collection):
#     """
#     Generates embeddings for text chunks and stores them in the ChromaDB collection.
#     """
#     print(f"--- Embedding and storing {len(chunks)} text chunks ---")
    
#     # ChromaDB's add method is optimized for batches. We'll prepare our data in lists.
#     documents_to_store = [chunk.page_content for chunk in chunks]
#     metadatas_to_store = [chunk.metadata for chunk in chunks]
#     ids_to_store = [f"text_{i}" for i in range(len(chunks))]

#     # Embed and add to the collection in batches for efficiency
#     # Note: ChromaDB's LangChain integration can also handle this, but doing it directly
#     # gives us more control and clarity.
#     embeddings = text_embedding_model.embed_documents(documents_to_store)
#     collection.add(
#         embeddings=embeddings,
#         documents=documents_to_store,
#         metadatas=metadatas_to_store,
#         ids=ids_to_store
#     )
#     print("--- Text chunks embedded and stored successfully ---")

# def embed_images_and_store(image_directory, collection):
#     """
#     Generates embeddings for images and stores them in the ChromaDB collection.
#     """
#     print(f"--- Embedding and storing images from {image_directory} ---")
#     image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
#     # Prepare lists for batch addition
#     image_paths_to_store = [os.path.join(image_directory, file) for file in image_files]
#     ids_to_store = [f"image_{i}" for i in range(len(image_files))]
    
#     # Generate embeddings for all images
#     # The SentenceTransformer expects a list of PIL Image objects
#     pil_images = [Image.open(path) for path in image_paths_to_store]
#     embeddings = image_embedding_model.encode(pil_images)

#     collection.add(
#         embeddings=embeddings.tolist(), # Convert numpy array to list
#         documents=image_paths_to_store, # Store the file path as the "document"
#         ids=ids_to_store
#     )
#     print("--- Images embedded and stored successfully ---")


# # --- TEMPORARY TEST BLOCK ---
# if __name__ == '__main__':
#     # Test Step 1: Loading
#     documents = load_documents()
#     print(f"\n--- Verification ---")
#     print(f"Number of loaded documents: {len(documents)}")
#     # Print the metadata of the first document to see its source
#     if documents:
#         print(f"Metadata of first document: {documents[0].metadata}")





# # --- TEMPORARY TEST BLOCK ---
# if __name__ == '__main__':
#     # Test Step 1: Loading
#     documents = load_documents()
    
#     # Test Step 2: Chunking
#     chunks = chunk_documents(documents)
#     print(f"\n--- Verification ---")
#     print(f"Number of chunks created: {len(chunks)}")
#     if chunks:
#         print(f"Content of first chunk:\n'{chunks[0].page_content[:200]}...'") # Print first 200 chars
#         print(f"Length of first chunk: {len(chunks[0].page_content)}")




if __name__ == '__main__':
    # Initialize models and database client
    text_model, image_model, db_client = get_models_and_db_client()

    print("Got the models and db client")
    # Get or create the collections
    text_collection = db_client.get_or_create_collection("portfolio_text")
    image_collection = db_client.get_or_create_collection("portfolio_images")

    # Load and chunk the documents
    all_docs = load_documents()
    chunked_docs = chunk_documents(all_docs)
    
    # Embed and store both text and images
    embed_text_and_store(chunked_docs, text_collection, text_model)
    embed_images_and_store('data/images', image_collection, image_model)
    
    print("\n\n--- VECTOR DATABASE CREATION COMPLETE ---")
    print(f"Text collection contains: {text_collection.count()} documents")
    print(f"Image collection contains: {image_collection.count()} documents")




# if __name__ == '__main__':
#     # Step 1: Load the documents
#     all_docs = load_documents()
    
#     # Step 2: Chunk the documents
#     chunked_docs = chunk_documents(all_docs)
    
#     # Step 3: Embed and store text chunks
#     embed_text_and_store(chunked_docs, text_collection)
    
#     # Step 4: Embed and store images
#     embed_images_and_store('data/images', image_collection)
    
#     print("\n\n--- VECTOR DATABASE CREATION COMPLETE ---")
#     print(f"Text collection count: {text_collection.count()}")
#     print(f"Image collection count: {image_collection.count()}")