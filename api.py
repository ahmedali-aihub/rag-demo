import io
import logging
import os
import re

import chromadb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()
logging.getLogger("google_genai").setLevel(logging.ERROR)

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(".", "chroma_db"))
COLLECTION_NAME = "pdf_rag_gemini"
EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Optional: if set, clients must send header "X-API-Key: <value>"
APP_API_KEY = os.getenv("APP_API_KEY")
# Comma-separated allowed origins for browsers, e.g. "https://mysite.com"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a document. "
    "You are given excerpts from it. Treat them as the document's content and "
    "answer strictly from them. Be accurate and specific, copy names, numbers "
    "and lists exactly as written, and never invent facts. For general "
    "questions (e.g. 'tell me about X'), summarize what the excerpts say. "
    "Only if the excerpts contain nothing relevant, reply exactly: "
    "\"I couldn't find that in the document.\" Do not mention page numbers or sources."
)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

gclient = genai.Client(api_key=api_key)
collection = chromadb.PersistentClient(path=CHROMA_PATH).get_or_create_collection(
    name=COLLECTION_NAME
)

app = FastAPI(title="TechZone RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_key(x_api_key: str | None = Header(default=None)):
    if APP_API_KEY and x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str


def embed(text: str, task: str) -> list[float]:
    res = gclient.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task),
    )
    return res.embeddings[0].values


def retrieve(question: str) -> str:
    if collection.count() == 0:
        raise HTTPException(status_code=409, detail="No documents ingested yet. POST /ingest first.")
    results = collection.query(
        query_embeddings=[embed(question, "RETRIEVAL_QUERY")], n_results=TOP_K
    )
    return "\n\n".join(results["documents"][0])


def build_prompt(question: str) -> str:
    return f"Document excerpts:\n{retrieve(question)}\n\nQuestion: {question}"


GEN_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": collection.count()}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    prompt = build_prompt(req.question)
    try:
        res = gclient.models.generate_content(model=LLM_MODEL, contents=prompt, config=GEN_CONFIG)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
    return AskResponse(answer=(res.text or "").strip())


@app.post("/ask/stream", dependencies=[Depends(require_key)])
def ask_stream(req: AskRequest):
    prompt = build_prompt(req.question)

    def gen():
        try:
            for chunk in gclient.models.generate_content_stream(
                model=LLM_MODEL, contents=prompt, config=GEN_CONFIG
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[error: {e}]"

    return StreamingResponse(gen(), media_type="text/plain")


@app.post("/ingest", dependencies=[Depends(require_key)])
async def ingest(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    name = os.path.basename(file.filename)
    if collection.get(where={"source": name}, limit=1)["ids"]:
        return {"status": "skipped", "detail": f"'{name}' already ingested"}

    reader = PdfReader(io.BytesIO(await file.read()))
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs, embs, metas, ids = [], [], [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        for i, chunk in enumerate(splitter.split_text(text)):
            docs.append(chunk)
            embs.append(embed(chunk, "RETRIEVAL_DOCUMENT"))
            metas.append({"source": name, "page": page_num})
            ids.append(f"{name}_p{page_num}_{i}")
    if not docs:
        raise HTTPException(status_code=422, detail="No text could be extracted from the PDF")
    collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    return {"status": "ingested", "chunks": len(docs)}
