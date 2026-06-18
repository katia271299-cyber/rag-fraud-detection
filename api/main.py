"""
api/main.py
API REST FastAPI pour exposer le pipeline RAG.

Démarrage : uvicorn api.main:app --reload --port 8000
Docs auto  : http://localhost:8000/docs
"""
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ajouter le répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import create_vector_store
from src.generation.llm import create_llm
from src.generation.rag_chain import RAGChain

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Finance API",
    description="Pipeline RAG pour interroger des documents financiers en langage naturel.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # En production : spécifier les domaines autorisés
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialisation du pipeline (au démarrage) ─────────────────────────────────
embedder     = None
vector_store = None
llm          = None
rag_chain    = None


@app.on_event("startup")
async def startup():
    global embedder, vector_store, llm, rag_chain
    
    logger.info("Initialisation du pipeline RAG...")
    
    embedder     = Embedder(model_name=config.EMBEDDING_MODEL)
    vector_store = create_vector_store(
        backend=config.VECTOR_STORE,
        embedder=embedder,
        collection_name=config.COLLECTION_NAME,
        persist_dir=config.CHROMA_DB_DIR,
    )
    llm = create_llm(
        provider=config.LLM_PROVIDER,
        model=config.LLM_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_NEW_TOKENS,
    )
    rag_chain = RAGChain(
        vector_store=vector_store,
        llm=llm,
        top_k=config.TOP_K,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        system_prompt=config.SYSTEM_PROMPT,
    )
    logger.info(f"✓ Pipeline prêt — {vector_store.count()} documents indexés")


# ── Modèles Pydantic ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Quels sont les patterns de fraude les plus fréquents en 2023 ?",
                "top_k": 5
            }
        }


class SourceModel(BaseModel):
    source:  str
    page:    Optional[str] = None
    chunk:   str
    score:   float


class AskResponse(BaseModel):
    answer:     str
    sources:    List[SourceModel]
    question:   str
    latency_ms: float
    doc_count:  int


class IngestResponse(BaseModel):
    message:      str
    chunks_added: int
    total_docs:   int


class StatsResponse(BaseModel):
    total_documents: int
    embedding_model: str
    llm_model:       str
    vector_backend:  str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Vérifie que l'API et le pipeline sont opérationnels."""
    return {
        "status": "ok",
        "docs_indexed": vector_store.count() if vector_store else 0,
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    """Retourne des statistiques sur le pipeline."""
    return StatsResponse(
        total_documents=vector_store.count(),
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.LLM_MODEL,
        vector_backend=config.VECTOR_STORE,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """
    Pose une question en langage naturel sur le corpus de documents.
    
    Retourne une réponse sourcée avec les extraits de documents utilisés.
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")
    
    # Surcharge optionnelle du top_k
    if request.top_k:
        rag_chain.top_k = request.top_k
    
    try:
        response = rag_chain.ask(request.question)
    except Exception as e:
        logger.error(f"Erreur lors de la génération : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur LLM : {str(e)}")
    
    # Formatage des sources pour la réponse
    sources = [
        SourceModel(
            source=r.document.metadata.get("source", "inconnu"),
            page=str(r.document.metadata.get("page", "")),
            chunk=r.document.content[:300] + "...",  # Preview du chunk
            score=round(r.score, 4),
        )
        for r in response.sources
    ]
    
    return AskResponse(
        answer=response.answer,
        sources=sources,
        question=request.question,
        latency_ms=round(response.latency_ms, 1),
        doc_count=len(response.sources),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingère un nouveau document dans le corpus (PDF, TXT, CSV, JSON).
    Le document est automatiquement découpé et vectorisé.
    """
    if not vector_store:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé")
    
    # Vérification de l'extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension non supportée : {suffix}. Acceptés : {config.SUPPORTED_EXTENSIONS}"
        )
    
    # Sauvegarde temporaire
    temp_path = config.DATA_RAW_DIR / file.filename
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    try:
        loader  = DocumentLoader()
        chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        
        docs   = loader.load_file(temp_path)
        chunks = chunker.split_documents(docs)
        vector_store.add_documents(chunks)
        
        return IngestResponse(
            message=f"Document '{file.filename}' ingéré avec succès.",
            chunks_added=len(chunks),
            total_docs=vector_store.count(),
        )
    except Exception as e:
        logger.error(f"Erreur ingestion : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
