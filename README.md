# ðŸ” RAG Pipeline â€” DonnÃ©es FinanciÃ¨res & Fraude

Un systÃ¨me de Retrieval-Augmented Generation (RAG) permettant d'interroger en **langage naturel** un corpus de documents financiers (rapports de fraude, transactions, notes internes) et d'obtenir des rÃ©ponses **sourcÃ©es et traÃ§ables**.

---

## ðŸ§  Ce que fait ce projet

```
Question â†’ Embedding â†’ Recherche vectorielle â†’ Contexte â†’ LLM â†’ RÃ©ponse + Sources
```

1. **Ingestion** : charge des PDF, CSV et textes financiers
2. **Chunking** : dÃ©coupe intelligente en segments de 512 tokens
3. **Vectorisation** : embedding avec `sentence-transformers`
4. **Stockage** : ChromaDB (persistant) ou FAISS (rapide)
5. **Retrieval** : top-K chunks par similaritÃ© cosinus Ã  la question
6. **GÃ©nÃ©ration** : Mistral 7B via Ollama rÃ©pond en citant ses sources
7. **Exposition** : API FastAPI + interface Streamlit

---

## ðŸ—‚ï¸ Structure du projet

```
rag-finance/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/              # Documents bruts (PDF, CSV, TXT)
â”‚   â””â”€â”€ processed/        # Chunks prÃ©processÃ©s (JSON)
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ ingestion/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ loader.py     # Chargement multi-format
â”‚   â”‚   â””â”€â”€ chunker.py    # DÃ©coupe en chunks
â”‚   â”œâ”€â”€ retrieval/
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ embedder.py   # CrÃ©ation des embeddings
â”‚   â”‚   â””â”€â”€ vector_store.py  # Interface ChromaDB/FAISS
â”‚   â””â”€â”€ generation/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â”œâ”€â”€ llm.py        # Interface LLM (Ollama/HuggingFace)
â”‚       â””â”€â”€ rag_chain.py  # Pipeline RAG complet
â”œâ”€â”€ api/
â”‚   â””â”€â”€ main.py           # FastAPI â€” endpoints REST
â”œâ”€â”€ ui/
â”‚   â””â”€â”€ app.py            # Streamlit â€” interface dÃ©mo
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ ingest_data.py    # Script d'ingestion standalone
â”‚   â””â”€â”€ generate_samples.py  # GÃ©nÃ©ration de donnÃ©es fictives
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_pipeline.py
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ exploration.ipynb
â”œâ”€â”€ config.py             # Configuration centrale
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## âš¡ DÃ©marrage rapide

### 1. PrÃ©requis

```bash
python >= 3.10
ollama (pour le LLM local)
```

### 2. Installation

```bash
git clone https://github.com/katia271299-cyber/rag-finance-langchain
cd rag-finance

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. TÃ©lÃ©charger le modÃ¨le LLM

```bash
# Installer Ollama : https://ollama.com
ollama pull mistral
```

### 4. GÃ©nÃ©rer des donnÃ©es de test

```bash
python scripts/generate_samples.py
```

### 5. IngÃ©rer les documents

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

## ðŸ”Œ Endpoints API

| MÃ©thode | Route | Description |
|---------|-------|-------------|
| `POST` | `/ask` | Poser une question (retourne rÃ©ponse + sources) |
| `POST` | `/ingest` | IngÃ©rer un nouveau document |
| `GET` | `/health` | Statut du systÃ¨me |
| `GET` | `/stats` | Nombre de documents indexÃ©s |

**Exemple de requÃªte :**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les patterns de fraude les plus frÃ©quents en 2023 ?"}'
```

**RÃ©ponse :**
```json
{
  "answer": "Les patterns de fraude les plus frÃ©quents incluent...",
  "sources": [
    {"doc_id": "rapport_q3_2023.pdf", "chunk": "...", "score": 0.92},
    {"doc_id": "analyse_transactions.csv", "chunk": "...", "score": 0.87}
  ],
  "latency_ms": 1240
}
```

---

## ðŸ”§ Configuration

Voir `config.py` pour ajuster :
- Le modÃ¨le d'embedding (`EMBEDDING_MODEL`)
- La taille des chunks (`CHUNK_SIZE`)
- Le nombre de rÃ©sultats rÃ©cupÃ©rÃ©s (`TOP_K`)
- Le backend vectoriel (`VECTOR_STORE`: `"chroma"` ou `"faiss"`)
- Le LLM utilisÃ© (`LLM_PROVIDER`: `"ollama"` ou `"huggingface"`)

---

## ðŸ§ª Tests

```bash
pytest tests/ -v
```

---

## ðŸ“ˆ Angle CV / Portfolio

Ce projet dÃ©montre :
- **Ingestion** multi-format (PDF, CSV, JSON, TXT)
- **Chunking** avec recouvrement (overlap) pour prÃ©server le contexte
- **Recherche sÃ©mantique** avec embeddings et ChromaDB
- **Prompt engineering** avec contexte injectÃ© dynamiquement
- **API production-ready** avec FastAPI + Pydantic
- **TraÃ§abilitÃ©** : chaque rÃ©ponse cite ses sources avec score de confiance
- **Domain expertise** : appliquÃ© aux donnÃ©es financiÃ¨res / fraude

---

## ðŸ› ï¸ Stack technique

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

