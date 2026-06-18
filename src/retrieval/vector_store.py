"""
src/retrieval/vector_store.py
Interface unifiée pour ChromaDB et FAISS.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from src.ingestion.loader import Document
from src.retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Résultat d'une recherche vectorielle."""
    
    def __init__(self, document: Document, score: float):
        self.document = document
        self.score    = score  # Similarité cosinus [0, 1]
    
    def __repr__(self):
        return f"RetrievalResult(score={self.score:.3f}, source={self.document.metadata.get('source', '?')})"


# ── ChromaDB ─────────────────────────────────────────────────────────────────

class ChromaVectorStore:
    """
    Stockage vectoriel persistant avec ChromaDB.
    
    ✅ Persistant entre les exécutions (données sauvegardées sur disque)
    ✅ Interface haut niveau avec filtres sur les métadonnées
    ✅ Idéal pour la démo et le développement
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | Path,
        embedder: Embedder,
    ):
        # ChromaDB 0.5+ instantiates ONNXMiniLM at class-definition time, which
        # loads onnxruntime DLLs. Stub it out before import since we use our own
        # embedder (Ollama) and never rely on ChromaDB's default function.
        import sys
        import types
        if "onnxruntime" not in sys.modules:
            sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
        import chromadb
        from chromadb.utils.embedding_functions import EmbeddingFunction

        class _NoOpEF(EmbeddingFunction):
            def __call__(self, input):  # noqa: A002
                return []

        self.embedder = embedder
        persist_dir   = str(Path(persist_dir).resolve())

        self.client     = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_NoOpEF(),
        )
        logger.info(f"ChromaDB — collection '{collection_name}' ({self.collection.count()} docs)")

    def add_documents(self, documents: List[Document], batch_size: int = 50) -> None:
        """Ajoute des documents dans la collection par lots pour éviter les crashes HNSW."""
        if not documents:
            return

        texts   = [doc.content for doc in documents]
        vectors = self.embedder.embed_texts(texts)
        ids     = [f"doc_{i}_{hash(doc.content) % 10**8}" for i, doc in enumerate(documents)]
        metas   = [{k: str(v) for k, v in doc.metadata.items()} for doc in documents]

        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self.collection.add(
                ids=ids[start:end],
                embeddings=vectors[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metas[start:end],
            )
            logger.info(f"  Lot ajouté : {end}/{len(documents)} chunks dans ChromaDB")

        logger.info(f"✓ {len(documents)} chunks ajoutés dans ChromaDB")

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Recherche les top_k chunks les plus proches de la question."""
        query_vector = self.embedder.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        
        retrieval_results = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB retourne une distance cosinus [0, 2] → conversion en similarité [0, 1]
            score = 1 - (distance / 2)
            retrieval_results.append(RetrievalResult(
                document=Document(content=doc_text, metadata=metadata),
                score=score,
            ))
        
        return retrieval_results

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)
        logger.warning("Collection supprimée.")


# ── FAISS ────────────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Stockage vectoriel avec FAISS (Facebook AI Similarity Search).
    
    ✅ Extrêmement rapide pour de grands volumes (millions de vecteurs)
    ✅ Optimisé pour la recherche par produit scalaire / cosinus
    ⚠️  Persistance manuelle (pickle des métadonnées)
    ⚠️  Pas de filtres sur les métadonnées natifs
    """

    def __init__(
        self,
        persist_dir: str | Path,
        embedder: Embedder,
    ):
        import faiss
        self.faiss    = faiss
        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.persist_dir / "faiss.index"
        self.meta_path  = self.persist_dir / "faiss_meta.pkl"
        
        # Chargement ou création de l'index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "rb") as f:
                self.metadata_store = pickle.load(f)
            logger.info(f"FAISS chargé — {self.index.ntotal} vecteurs")
        else:
            dim   = embedder.dimension
            self.index = faiss.IndexFlatIP(dim)  # Produit scalaire (= cosinus si vecteurs normalisés)
            self.metadata_store: List[Dict[str, Any]] = []
            logger.info(f"FAISS initialisé — dimension {dim}")

    def add_documents(self, documents: List[Document]) -> None:
        texts   = [doc.content for doc in documents]
        vectors = self.embedder.embed_texts(texts).astype("float32")
        
        self.index.add(vectors)
        self.metadata_store.extend([
            {"content": doc.content, **doc.metadata}
            for doc in documents
        ])
        self._save()
        logger.info(f"✓ {len(documents)} chunks ajoutés dans FAISS")

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_vector = self.embedder.embed_query(query).astype("float32").reshape(1, -1)
        
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta    = self.metadata_store[idx]
            content = meta.pop("content", "")
            results.append(RetrievalResult(
                document=Document(content=content, metadata=meta),
                score=float(score),
            ))
        return results

    def count(self) -> int:
        return self.index.ntotal

    def _save(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata_store, f)


# ── Factory ──────────────────────────────────────────────────────────────────

def create_vector_store(
    backend: str,
    embedder: Embedder,
    collection_name: str = "finance_fraud",
    persist_dir: str | Path = "data/vector_db",
):
    """
    Factory pour créer le bon vector store selon la configuration.
    
    Args:
        backend: "chroma" ou "faiss"
    """
    if backend == "chroma":
        return ChromaVectorStore(collection_name, persist_dir, embedder)
    elif backend == "faiss":
        return FAISSVectorStore(persist_dir, embedder)
    else:
        raise ValueError(f"Backend inconnu : {backend}. Choisir 'chroma' ou 'faiss'.")
