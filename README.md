# RAG Pipeline — Données Financières & Fraude

Un système de Retrieval-Augmented Generation (RAG) permettant d'interroger en **langage naturel** un corpus de documents financiers (rapports de fraude, transactions, notes internes) et d'obtenir des réponses **sourcées et traçables**.

---

## Ce que fait ce projet

```
Question → Embedding → Recherche vectorielle → Contexte → LLM → Réponse + Sources
```

1. **Ingestion** : charge des PDF, CSV et textes financiers
2. **Chunking** : découpe intelligente en segments de 512 tokens
3. **Vectorisation** : embedding avec `sentence-transformers`
4. **Stockage** : ChromaDB (persistant) ou FAISS (rapide)
5. **Retrieval** : top-K chunks par similarité cosinus à la question
6. **Génération** : Mistral 7B via Ollama répond en citant ses sources
7. **Exposition** : API FastAPI + interface Streamlit

---

## Structure du projet

```
rag-finance/
├── data/
│   ├── raw/              # Documents bruts (PDF, CSV, TXT)
│   └── processed/        # Chunks préprocessés (JSON)
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py     # Chargement multi-format
│   │   └── chunker.py    # Découpe en chunks
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py   # Création des embeddings
│   │   └── vector_store.py  # Interface ChromaDB/FAISS
│   └── generation/
│       ├── __init__.py
│       ├── llm.py        # Interface LLM (Ollama/HuggingFace)
│       └── rag_chain.py  # Pipeline RAG complet
├── api/
│   └── main.py           # FastAPI — endpoints REST
├── ui/
│   └── app.py            # Streamlit — interface démo
├── scripts/
│   ├── ingest_data.py    # Script d'ingestion standalone
│   └── generate_samples.py  # Génération de données fictives
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb
├── config.py             # Configuration centrale
├── requirements.txt
└── README.md
```

---

## Démarrage rapide

### 1. Prérequis

```bash
python >= 3.10
ollama (pour le LLM local)
```

### 2. Installation

```bash
git clone https://github.com/katia271299-cyber/rag-fraud-detection
cd rag-finance

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Télécharger le modèle LLM

```bash
# Installer Ollama : https://ollama.com
ollama pull mistral
```

### 4. Générer des données de test

```bash
python scripts/generate_samples.py
```

### 5. Ingérer les documents

```bash
python scripts/ingest_data.py --data-dir data/raw
```

### 6. Lancer l'API

```bash
uvicorn api.main:app --reload --port 8000
# Docs : http://localhost:8000/docs
```

### 7. Lancer l'interface Streamlit

```bash
streamlit run ui/app.py
```

---

## Endpoints API

| Méthode | Route | Description |
|---------|-------|-------------|
| `POST` | `/ask` | Poser une question (retourne réponse + sources) |
| `POST` | `/ingest` | Ingérer un nouveau document |
| `GET` | `/health` | Statut du système |
| `GET` | `/stats` | Nombre de documents indexés |

**Exemple de requête :**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les patterns de fraude les plus fréquents en 2023 ?"}'
```

**Réponse :**
```json
{
  "answer": "Les patterns de fraude les plus fréquents incluent...",
  "sources": [
    {"doc_id": "rapport_q3_2023.pdf", "chunk": "...", "score": 0.92},
    {"doc_id": "analyse_transactions.csv", "chunk": "...", "score": 0.87}
  ],
  "latency_ms": 1240
}
```

---

## Configuration

Voir `config.py` pour ajuster :
- Le modèle d'embedding (`EMBEDDING_MODEL`)
- La taille des chunks (`CHUNK_SIZE`)
- Le nombre de résultats récupérés (`TOP_K`)
- Le backend vectoriel (`VECTOR_STORE`: `"chroma"` ou `"faiss"`)
- Le LLM utilisé (`LLM_PROVIDER`: `"ollama"` ou `"huggingface"`)

---

## Tests

```bash
pytest tests/ -v
```

---

## Angle CV / Portfolio

Ce projet démontre :
- **Ingestion** multi-format (PDF, CSV, JSON, TXT)
- **Chunking** avec recouvrement (overlap) pour préserver le contexte
- **Recherche sémantique** avec embeddings et ChromaDB
- **Prompt engineering** avec contexte injecté dynamiquement
- **API production-ready** avec FastAPI + Pydantic
- **Traçabilité** : chaque réponse cite ses sources avec score de confiance
- **Domain expertise** : appliqué aux données financières / fraude

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Orchestration | LangChain |
| LLM | Mistral 7B (Ollama) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (FAISS en option) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| PDF parsing | PyMuPDF |
| Tests | Pytest |


