"""
scripts/ingest_data.py
Script d'ingestion standalone — charge et vectorise tous les documents.

Usage :
    python scripts/ingest_data.py                    # Ingère data/raw/
    python scripts/ingest_data.py --data-dir mon/rep # Dossier personnalisé
    python scripts/ingest_data.py --reset            # Réinitialise la base vectorielle
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import create_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingestion de documents dans le pipeline RAG")
    parser.add_argument("--data-dir", default=str(config.DATA_RAW_DIR), help="Répertoire source")
    parser.add_argument("--reset", action="store_true", help="Supprimer et recréer la base vectorielle")
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Répertoire introuvable : {data_dir}")
        logger.info("Conseil : lancez d'abord 'python scripts/generate_samples.py'")
        sys.exit(1)

    # ── Initialisation ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Pipeline RAG Finance — Ingestion")
    logger.info("=" * 60)
    
    embedder     = Embedder(model_name=config.EMBEDDING_MODEL)
    vector_store = create_vector_store(
        backend=config.VECTOR_STORE,
        embedder=embedder,
        collection_name=config.COLLECTION_NAME,
        persist_dir=config.CHROMA_DB_DIR,
    )
    
    if args.reset:
        logger.warning("Réinitialisation de la base vectorielle...")
        try:
            vector_store.delete_collection()
            # Recréer
            vector_store = create_vector_store(
                backend=config.VECTOR_STORE,
                embedder=embedder,
                collection_name=config.COLLECTION_NAME,
                persist_dir=config.CHROMA_DB_DIR,
            )
        except AttributeError:
            logger.warning("Reset non supporté pour FAISS (supprimez manuellement data/vector_db/)")
    
    # ── Chargement ────────────────────────────────────────────────────────────
    logger.info(f"Chargement depuis : {data_dir}")
    loader  = DocumentLoader()
    chunker = TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    
    docs   = loader.load_directory(data_dir)
    if not docs:
        logger.error("Aucun document trouvé.")
        sys.exit(1)
    
    # ── Chunking ──────────────────────────────────────────────────────────────
    logger.info(f"Chunking de {len(docs)} documents...")
    chunks = chunker.split_documents(docs)
    
    # ── Indexation ────────────────────────────────────────────────────────────
    logger.info(f"Indexation de {len(chunks)} chunks...")
    vector_store.add_documents(chunks)
    
    # ── Résumé ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"✅ Ingestion terminée !")
    logger.info(f"   Documents chargés : {len(docs)}")
    logger.info(f"   Chunks créés      : {len(chunks)}")
    logger.info(f"   Total en base     : {vector_store.count()}")
    logger.info(f"   Backend           : {config.VECTOR_STORE}")
    logger.info("=" * 60)
    logger.info("👉 Lancer l'API  : uvicorn api.main:app --reload")
    logger.info("👉 Lancer l'UI   : streamlit run ui/app.py")


if __name__ == "__main__":
    main()
