"""
config.py — Configuration centrale du pipeline RAG Finance
Modifiez ces valeurs pour adapter le projet à votre environnement.
"""
from pathlib import Path

# ── Chemins ─────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_RAW_DIR   = BASE_DIR / "data" / "raw"
DATA_PROC_DIR  = BASE_DIR / "data" / "processed"
CHROMA_DB_DIR  = BASE_DIR / "data" / "chroma_db"

# ── Ingestion & Chunking ─────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # Tokens par chunk
CHUNK_OVERLAP = 64     # Chevauchement entre chunks (contexte préservé)
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".csv", ".json"]

# ── Embeddings ───────────────────────────────────────────────────────────────
# L'embedder (src/retrieval/embedder.py) appelle Ollama : ne mettre ici que des
# modeles d'embedding disponibles sur Ollama (ex: "nomic-embed-text", "mxbai-embed-large").
EMBEDDING_MODEL = "nomic-embed-text"

# ── Vector Store ─────────────────────────────────────────────────────────────
VECTOR_STORE      = "faiss"     # "chroma" | "faiss"
COLLECTION_NAME   = "finance_fraud_docs"
TOP_K             = 5           # Nombre de chunks récupérés par requête
SIMILARITY_THRESHOLD = 0.15     # Score minimum de similarité (repli sur le top-3 brut si rien ne passe ce seuil)

# ── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER  = "ollama"        # "ollama" | "huggingface"
LLM_MODEL     = "phi3"          # Modele Ollama (ollama pull phi3) -- 2.2 Go,
                                 # bon compromis qualite/poids. "mistral" (4.1 Go)
                                 # est un cran au-dessus si l'espace disque le permet.
OLLAMA_BASE_URL = "http://localhost:11434"

# HuggingFace (si LLM_PROVIDER == "huggingface")
HF_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
HF_DEVICE     = "cpu"           # "cuda" si GPU disponible

# ── Génération ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 768
TEMPERATURE     = 0.1           # Bas pour des réponses factuelles

# ── API ──────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Prompt System ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant expert en analyse financière et détection de fraude.
Tu réponds uniquement en te basant sur les documents fournis en contexte.
Si l'information n'est pas dans le contexte, dis-le clairement.
Cite toujours les sources utilisées pour formuler ta réponse.
Réponds en français, de manière précise et structurée."""
