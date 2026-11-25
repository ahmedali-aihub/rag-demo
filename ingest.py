import os
import re
import sys
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Configurable constants
CHROMA_PATH = os.path.join(".", "chroma_db")
COLLECTION_NAME = "pdf_rag_gemini"
EMBED_MODEL = "gemini-embedding-001"

load_dotenv()
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def ingest_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY in the .env file.")
        sys.exit(1)
    gclient = genai.Client(api_key=api_key)

    doc_name = os.path.basename(pdf_path)

    # Initialize persistent ChromaDB collection
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Skip re-ingestion if document already exists
    existing = collection.get(where={"source": doc_name}, limit=1)
    if existing["ids"]:
        print(f"Document '{doc_name}' is already ingested in '{COLLECTION_NAME}'. Skipping.")
        return

    # Extract text page by page
    reader = PdfReader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    print(f"Processing '{doc_name}' ({len(reader.pages)} pages)...")

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        # Clean and normalize whitespace so PDF newline artifacts don't break chunks
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            continue

        # Chunk text per page to preserve page numbers
        chunks = splitter.split_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            response = gclient.models.embed_content(
                model=EMBED_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            documents.append(chunk)
            embeddings.append(response.embeddings[0].values)
            metadatas.append({"source": doc_name, "page": page_num})
            ids.append(f"{doc_name}_p{page_num}_{chunk_idx}")

    # Store chunks into ChromaDB
    if documents:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Ingested {len(documents)} chunks from '{doc_name}'.")
    else:
        print("No text could be extracted from the PDF.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to PDF: ").strip().strip('"')
    ingest_pdf(path)
