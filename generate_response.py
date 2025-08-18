import os
import re
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict

# Setup logging
logging.basicConfig(level=logging.INFO)

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not set in environment variables!")

VECTORSTORE_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# You can customize 'k' here or pass it dynamically
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# Initialize Mistral LLM
llm = ChatMistralAI(
    api_key=MISTRAL_API_KEY,
    model="open-mistral-7b",
    temperature=0.3,
    max_tokens=2048
)

TELECOM_KEYWORDS = [
    "5G", "4G", "3G", "network", "tower", "signal", "latency", "bandwidth",
    "telecom", "fiber", "Switch", "VoLTE", "spectrum", "wireless", "antenna",
    "cellular", "coverage", "base station", "call drop", "frequency", "roaming",
    "backhaul", "microwave link", "carrier aggregation", "handovers",
    "power supply", "UPS", "voltage", "current", "battery backup", "inverter",
    "DC rectifier", "generator", "power failure", "electrical outage", "load balancing",
    "fault protection", "circuit breaker", "transformer", "power distribution unit",
    "fault", "error", "diagnostics", "debugging", "alarm", "packet loss",
    "throughput", "interference", "jitter", "latency issues", "outage", "connectivity issue",
    "network congestion", "slow speed", "ping test", "signal drop", "no service",
    "modulation", "demodulation", "retransmission", "bit error rate", "network reset"
]

def is_telecom_query(query):
    """Improved telecom keyword detection using regex word boundaries."""
    query_lower = query.lower()
    for keyword in TELECOM_KEYWORDS:
        # Use regex to avoid partial matches
        pattern = rf'\b{re.escape(keyword.lower())}\b'
        if re.search(pattern, query_lower):
            return True
    return False

PROMPT_TEMPLATES = {
    "report": PromptTemplate(
        template=(
            "You are an AI assistant for telecom field engineers. "
            "Using the context below, generate a structured report (minimum 5000 characters) "
            "with sections: Title, Problem Description, Site Details, Analysis, Issues Identified, "
            "and Recommendations based on the query.\n\nContext: {context}\n\nQuery: {question}"
        ),
        input_variables=["context", "question"]
    ),
    "sop": PromptTemplate(
        template=(
            "You are an AI assistant for telecom field engineers. Using the context below, "
            "generate a Standard Operating Procedure (SOP) (minimum 5000 characters) with sections: "
            "Title, Purpose, Scope, Procedure Steps, Tools & Equipments, Responsibilities, "
            "Safety Guidelines, and References based on the query.\n\nContext: {context}\n\nQuery: {question}"
        ),
        input_variables=["context", "question"]
    ),
    "summary": PromptTemplate(
        template=(
            "You are an AI assistant for telecom field engineers. Using the context below, generate "
            "a concise summary (minimum 5000 characters) with sections: Network Overview, Key Protocols & Standards, "
            "Common Issues & Fixes, Safety Guidelines, Contact Info, and Glossary based on the query.\n\nContext: {context}\n\nQuery: {question}"
        ),
        input_variables=["context", "question"]
    ),
    "default": PromptTemplate(
        template=(
            "You are an AI assistant for telecom field engineers. Provide a response based on the context below "
            "to help field engineers resolve the issue.\n\nContext: {context}\n\nQuery: {question}"
        ),
        input_variables=["context", "question"]
    )
}

class CustomRetrievalQA(RetrievalQA):
    model_config = ConfigDict(arbitrary_types_allowed=True)

def get_prompt(doc_type):
    """Fetch prompt template, fallback to default if unknown type."""
    return PROMPT_TEMPLATES.get(doc_type.lower(), PROMPT_TEMPLATES["default"])

def generate_response(query, doc_type="report", k=2):
    """
    Generate response using RetrievalQA chain with Mistral.
    Parameters:
        query (str): user question
        doc_type (str): type of document response expected
        k (int): number of documents to retrieve
    Returns:
        response (str), relevant_doc (str), doc_type (str)
    """
    try:
        if not is_telecom_query(query):
            return "WARNING: This query is outside the scope of telecom-related topics.", None, doc_type

        # Update retriever with new k dynamically
        retriever.search_kwargs["k"] = k

        prompt = get_prompt(doc_type)
        qa_chain = CustomRetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa_chain.invoke({"query": query})
        response = result.get("result", "No response generated.")

        source_docs = result.get("source_documents") or []
        if source_docs:
            relevant_doc = source_docs[0].metadata.get("source", "No relevant document found.")
        else:
            relevant_doc = "No relevant document found."

        return response, relevant_doc, doc_type

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error: {str(e)}", None, doc_type

if __name__ == "__main__":
    query = input("Enter a query: ").strip()
    doc_type = input("Enter document type (report/sop/summary): ").strip().lower()
    if doc_type not in PROMPT_TEMPLATES:
        print(f"Unknown document type '{doc_type}', defaulting to generic response.")
        doc_type = "default"
    response, relevant_doc, _ = generate_response(query, doc_type)
    print(f"\nGenerated {doc_type}:\n{response}\n\nRelevant document: {relevant_doc}")
