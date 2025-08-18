import os
import re
import json
import logging
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Setup logging
logging.basicConfig(level=logging.INFO)

# Directory paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# Ensure output directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# File extension to loader mapping
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8")
}

def clean_text(text):
    """Clean extracted text by removing unwanted characters and formatting."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9.,!? ]+', '', text)).strip()

def save_text_file(path, content):
    """Save cleaned content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def process_documents():
    """Process raw telecom documents using LangChain loaders and save cleaned text."""
    processed_data = {}

    for file in tqdm(os.listdir(RAW_DATA_DIR), desc="Processing files"):
        file_path = os.path.join(RAW_DATA_DIR, file)
        file_name, file_ext = os.path.splitext(file)
        file_ext = file_ext.lower()

        loader_class = LOADERS.get(file_ext)
        if not loader_class:
            logging.warning(f"Skipping unsupported file type: {file}")
            continue

        try:
            loader = loader_class(file_path)
            documents = loader.load()
            text = " ".join(doc.page_content for doc in documents)
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue

        if text:
            cleaned_text = clean_text(text)
            processed_file_path = os.path.join(PROCESSED_DATA_DIR, f"{file_name}.txt")
            save_text_file(processed_file_path, cleaned_text)
            processed_data[file] = processed_file_path

    # Save metadata JSON
    metadata_path = os.path.join(PROCESSED_DATA_DIR, "processed_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    logging.info("Data preprocessing completed. Processed files saved in 'data/processed'.")

if __name__ == "__main__":
    process_documents()
