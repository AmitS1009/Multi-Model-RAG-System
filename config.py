import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')

# ✅ PDF input file path
PDF_PATH = os.path.join(RAW_DATA_DIR, 'qatar_test_doc.pdf')

# ✅ Output chunks
CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, 'extracted_chunks.json')

# ❗ IMPORTANT → FAISS needs a file, not a folder, so final path must include a filename
VECTOR_STORE_PATH = os.path.join(VECTOR_STORE_DIR, 'faiss_index', 'index.faiss')
VECTOR_STORE_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, 'faiss_index', 'index.pkl')

# Models
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-base'

def create_directories():
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VECTOR_STORE_DIR,
        os.path.join(VECTOR_STORE_DIR, 'faiss_index'),   # ensure FAISS folder exists
        IMAGES_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ All directories created")

if __name__ == "__main__":
    create_directories()
    print(f"\nDirectory structure:")
    print(f"  Raw data: {RAW_DATA_DIR}")
    print(f"  Processed data: {PROCESSED_DATA_DIR}")
    print(f"  Vector store: {VECTOR_STORE_DIR}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"\nPlace your PDF at: {PDF_PATH}")
