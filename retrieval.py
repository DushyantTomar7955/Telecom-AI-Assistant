import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORSTORE_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 docs by default

def retrieve_relevant_documents(query, top_k=3, similarity_threshold=0.7):
    """
    Retrieve top-k relevant documents with similarity above threshold.
    
    Args:
        query (str): Query text.
        top_k (int): Number of top documents to retrieve.
        similarity_threshold (float): Similarity threshold (0-1), higher is better.
        
    Returns:
        list of dict: Each dict has 'content' and 'source' keys.
    """
    # Get top_k results with similarity scores (distance)
    results = vector_store.similarity_search_with_score(query, k=top_k)
    
    # Convert FAISS distances to similarity scores (FAISS distance is usually L2 or cosine distance)
    # Here we assume cosine similarity: similarity = 1 - distance (if cosine distance used)
    # Adjust this according to your embedding/vector type!
    
    relevant_docs = []
    for doc, distance in results:
        similarity = 1 - distance  # Adjust if necessary
        if similarity >= similarity_threshold:
            relevant_docs.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "similarity": similarity
            })
    
    if not relevant_docs:
        return [{"content": "No relevant document found.", "source": None, "similarity": 0}]
    
    return relevant_docs

if __name__ == "__main__":
    query = input("Enter a query: ")
    top_docs = retrieve_relevant_documents(query, top_k=3, similarity_threshold=0.7)
    
    for i, doc in enumerate(top_docs, start=1):
        print(f"\nResult {i}:")
        print(f"Source: {doc['source']}")
        print(f"Similarity: {doc['similarity']:.3f}")
        print(f"Content:\n{doc['content']}")
