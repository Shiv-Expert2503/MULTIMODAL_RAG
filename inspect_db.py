# inspect_db.py
import chromadb

def inspect_database():
    # Connect to the existing database
    client = chromadb.PersistentClient(path="./chroma_db")

    print("--- Inspecting ChromaDB ---")
    print(f"Available collections: {[c.name for c in client.list_collections()]}")

    # Get the text collection
    text_collection = client.get_collection("portfolio_text")

    print(f"\n--- Peeking into 'portfolio_text' collection ---")
    # Get the first 5 items from the collection
    peek_data = text_collection.peek(limit=5)

    # A more detailed way to get specific items
    detailed_data = text_collection.get(
        ids=['text_0', 'text_1', 'text_2'],
        include=['metadatas', 'documents']
    )

    print("First 2 documents (using .get()):")
    for doc in detailed_data['documents']:
        print(f"- {doc[:150]}...") # Print first 150 chars

if __name__ == '__main__':
    inspect_database()