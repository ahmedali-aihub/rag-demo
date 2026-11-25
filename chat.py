import logging
import os
import sys
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.getLogger("google_genai").setLevel(logging.ERROR)

# Configurable constants
CHROMA_PATH = os.path.join(".", "chroma_db")
COLLECTION_NAME = "pdf_rag_gemini"
EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a document. "
    "You are given excerpts from it. Treat them as the document's content and "
    "answer strictly from them. Be accurate and specific, copy names, numbers "
    "and lists exactly as written, and never invent facts. For general "
    "questions (e.g. 'tell me about X'), summarize what the excerpts say. "
    "Only if the excerpts contain nothing relevant, reply exactly: "
    "\"I couldn't find that in the document.\" Do not mention page numbers or sources."
)


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY in the .env file.")
        sys.exit(1)
    gclient = genai.Client(api_key=api_key)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("RAG Query Terminal (type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        try:
            # Embed user question
            emb = gclient.models.embed_content(
                model=EMBED_MODEL,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            query_embedding = emb.embeddings[0].values

            results = collection.query(
                query_embeddings=[query_embedding], n_results=TOP_K
            )
            retrieved_docs = results["documents"][0] if results["documents"] else []

            if not retrieved_docs:
                print("No documents found in the database. Run ingest.py first.\n")
                continue

            context = "\n\n".join(retrieved_docs)
            prompt = f"Document excerpts:\n{context}\n\nQuestion: {query}"

            print("\nAnswer: ", end="", flush=True)
            stream = gclient.models.generate_content_stream(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT, temperature=0
                ),
            )
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
