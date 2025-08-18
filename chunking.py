import os
import json
import logging
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO)

# Define paths
PROCESSED_DATA_DIR = "data/processed"
CHUNKED_DATA_DIR = "data/chunks"
os.makedirs(CHUNKED_DATA_DIR, exist_ok=True)

# Chunking configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

def chunk_text_file(input_path):
    """Read and chunk text from a file."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text_splitter.split_text(text)

def save_chunks_to_file(output_path, chunks):
    """Save chunks into a file separated by newlines."""
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")

def process_chunking():
    """Main function to chunk all cleaned files and store metadata."""
    chunk_metadata = {}

    for file in tqdm(os.listdir(PROCESSED_DATA_DIR), desc="Chunking files"):
        if not file.endswith(".txt") or file == "processed_metadata.json":
            continue

        input_path = os.path.join(PROCESSED_DATA_DIR, file)
        output_path = os.path.join(CHUNKED_DATA_DIR, file)

        try:
            chunks = chunk_text_file(input_path)
            save_chunks_to_file(output_path, chunks)
            chunk_metadata[file] = len(chunks)
        except Exception as e:
            logging.error(f"Error processing {file}: {e}")
            continue

    # Save metadata
    metadata_path = os.path.join(CHUNKED_DATA_DIR, "chunk_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=4)

    logging.info("Chunking completed. Chunks saved in 'data/chunks/'.")

if __name__ == "__main__":
    process_chunking()
