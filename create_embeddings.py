import json
import os
from vector_store import VectorStore
import config


def main():
    print("STEP 2: Creating Embeddings\n")

    
    # 1. Check for extracted chunks
    
    if not os.path.exists(config.CHUNKS_PATH):
        print(f"error -> Processed data not found at {config.CHUNKS_PATH}")
        return

    print("✓ Processed data found.")
    print("\nLoading extracted chunks...")

    with open(config.CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"✓ Loaded {len(chunks)} chunks")

    # Count chunk types
    text_count = sum(1 for c in chunks if c['type'] == 'text')
    table_count = sum(1 for c in chunks if c['type'] == 'table')
    image_count = sum(1 for c in chunks if c['type'] == 'image')

    print(f"  - Text chunks: {text_count}")
    print(f"  - Tables: {table_count}")
    print(f"  - Images: {image_count}")

    
    # 2. Create embeddings
    
    print("\nCreating embeddings...\n")

    vector_store = VectorStore(model_name=config.EMBEDDING_MODEL)
    vector_store.create_embeddings(chunks)

    
    # 3. Ensure FAISS folder exists
    
    faiss_folder = os.path.dirname(config.VECTOR_STORE_PATH)
    os.makedirs(faiss_folder, exist_ok=True)

    
    # 4. Save vector store (folder, not file)
    
    vector_store.save(faiss_folder)

    print("\nCOMPLETE")
    print(f"Total vectors stored: {len(chunks)}")


if __name__ == "__main__":
    main()
