# RAG Pipeline — Données Financières & Fraude

Un système de Retrieval-Augmented Generation (RAG) permettant d'interroger en **langage naturel** un corpus de documents financiers (rapports de fraude, transactions, notes internes) et d'obtenir des réponses **sourcées et traçables**.

---

## Ce que fait ce projet

```
Question → Embedding → Recherche vectorielle → Contexte → LLM → Réponse + Sources
```

1. **Ingestion** : charge des PDF, CSV et textes financiers
2. **Chunking** : découpe intelligente en segments de 512 tokens
3. **Vectorisation** : embedding avec `nomic-embed-text` via Ollama
4. **Stockage** : ChromaDB (persistant) ou FAISS (rapide)
5. **Retrieval** : top-K chunks par similarité cosinus à la question
6. **Génération** : Phi-3 via Ollama répond en citant ses sources
7. **Exposition** : API FastAPI + interface Streamlit

---

## Jeu de données

Corpus 100% synthétique généré par `scripts/generate_samples.py` (aucune donnée
réelle), pensé pour donner une vraie profondeur de contenu au RAG :

| Source | Volume |
|---|---|
| Rapports annuels narratifs (2022-2024) | 3 documents |
| Cas de fraude anonymisés (`data/raw/cas/`) | 40 fichiers individuels |
| Transactions suspectes (CSV, une ligne = un document indexé) | 1 000 lignes |
| Alertes de conformité (JSON, un item = un document indexé) | 200 alertes |
| Glossaire des termes métier (TRACFIN, KYC, smurfing, etc.) | 30 termes |
| Guide opérationnel de détection | 1 document |
| **Total après ingestion** | **~1 250 documents indexables** |

---

## Structure du projet

```
rag-finance/
├── data/
│   ├── raw/              # Documents bruts (rapports, CSV, JSON, glossaire...)
│   │   └── cas/          # 40 recits de cas de fraude anonymises
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

### 3. Télécharger les modèles Ollama

```bash
# Installer Ollama : https://ollama.com
ollama pull phi3                # LLM de generation (2.2 Go)
ollama pull nomic-embed-text    # Modele d'embedding (274 Mo)
```

Les deux modèles sont nécessaires : `phi3` pour générer les réponses,
`nomic-embed-text` pour vectoriser les documents et les questions.
Alternative plus capable mais plus lourde (4,1 Go) : `ollama pull mistral`,
puis changer `LLM_MODEL` dans `config.py`.

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
| Orchestration | Pipeline RAG fait main (pas de framework type LangChain) |
| LLM | Phi-3 via Ollama (alternative locale plus capable : Mistral 7B ; ou HuggingFace Transformers) |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB (FAISS en option) |
| API | FastAPI + Uvicorn |
| Interface | Streamlit |
| PDF parsing | PyMuPDF |
| Tests | Pytest |


