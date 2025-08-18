import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Define paths
CHUNKED_DATA_DIR = "data/chunks"
VECTORSTORE_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_chunks_from_file(file_path, file_name):
    """Read a chunk file and return clean documents."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = content.split("\n\n")
    documents = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if chunk:
            documents.append(
                Document(page_content=chunk, metadata={"source": file_name})
            )

    return documents

def process_embeddings():
    """Generate vector embeddings from chunked files and store in FAISS."""
    all_documents = []
    metadata_info = {}

    for file in tqdm(os.listdir(CHUNKED_DATA_DIR), desc="Generating embeddings"):
        file_path = os.path.join(CHUNKED_DATA_DIR, file)

        if not file.endswith(".txt") or file == "chunk_metadata.json":
            continue

        try:
            documents = load_chunks_from_file(file_path, file)
            if documents:
                all_documents.extend(documents)
                metadata_info[file] = len(documents)
            else:
                print(f"⚠️ No valid content in {file}")

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    if not all_documents:
        print("No documents found for embedding. Exiting.")
        return

    # Create FAISS vector store
    vector_store = FAISS.from_documents(all_documents, embedding_model)
    vector_store.save_local(FAISS_INDEX_PATH)

    # Save metadata
    with open(os.path.join(VECTORSTORE_DIR, "embedding_metadata.json"), "w") as f:
        json.dump(metadata_info, f, indent=4)

    print(f"✅ Embedding completed. Saved to: {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    process_embeddings()
