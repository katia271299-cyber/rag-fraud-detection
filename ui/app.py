"""
ui/app.py - Interface Streamlit simplifiée pour le pipeline RAG Finance.
Lancement : streamlit run ui/app.py
"""
import sys
import logging
import traceback
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Logging dans un fichier ───────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent.parent / "rag_errors.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s — %(levelname)s — %(message)s",
    encoding="utf-8",
    force=True,
)
logging.info("=== Démarrage de l'app ===")

st.set_page_config(
    page_title="RAG Finance",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Finance — Détection de Fraude")
st.caption("Interrogez vos documents financiers en langage naturel.")

# ── Session state init ────────────────────────────────────────────────────────
if "question" not in st.session_state:
    st.session_state.question = ""

# ── Exemples ──────────────────────────────────────────────────────────────────
st.markdown("**Questions d'exemple :**")
col1, col2 = st.columns(2)
with col1:
    if st.button("Patterns de fraude fréquents ?"):
        st.session_state.question = "Quels sont les patterns de fraude les plus fréquents ?"
with col2:
    if st.button("Transactions suspectes élevées ?"):
        st.session_state.question = "Quelle est la transaction suspecte avec le montant le plus élevé ?"

# ── Question ──────────────────────────────────────────────────────────────────
question = st.text_area(
    "Votre question",
    height=80,
    placeholder="Ex : Quels sont les patterns de fraude les plus fréquents ?",
    key="question",
)

submit = st.button("🔍 Rechercher", type="primary")

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du pipeline RAG (première fois : 30-60 s)…")
def get_pipeline():
    logging.info("Chargement du pipeline...")
    import config
    from src.retrieval.embedder import Embedder
    from src.retrieval.vector_store import create_vector_store
    from src.generation.llm import create_llm
    from src.generation.rag_chain import RAGChain

    embedder = Embedder(model_name=config.EMBEDDING_MODEL)
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
    chain = RAGChain(
        vector_store=vector_store,
        llm=llm,
        top_k=config.TOP_K,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
        system_prompt=config.SYSTEM_PROMPT,
    )
    logging.info("Pipeline chargé avec succès.")
    return vector_store, chain

# ── Résultat ──────────────────────────────────────────────────────────────────
if submit and question.strip():
    logging.info(f"Question posée : {question}")
    with st.spinner("Recherche en cours…"):
        try:
            logging.info("Appel get_pipeline()...")
            vector_store, chain = get_pipeline()
            logging.info("Pipeline récupéré.")

            if vector_store.count() == 0:
                st.warning("⚠️ Aucun document indexé. Lancez d'abord : python scripts/ingest_data.py")
                logging.warning("Base vectorielle vide.")
            else:
                logging.info("Appel chain.ask()...")
                response = chain.ask(question)
                logging.info(f"Réponse reçue en {response.latency_ms}ms")

                st.success(f"✅ Réponse en {response.latency_ms:.0f}ms — {len(response.sources)} sources")
                st.subheader("💬 Réponse")
                st.markdown(response.answer)

                if response.sources:
                    st.subheader(f"📚 Sources ({len(response.sources)})")
                    for i, src in enumerate(response.sources, 1):
                        with st.expander(f"[{i}] {src.document.metadata.get('source', '?')} — score {src.score:.2f}"):
                            st.code(src.document.content[:400] + "...", language=None)

        except Exception as e:
            error_details = traceback.format_exc()
            logging.error(f"ERREUR : {e}\n{error_details}")
            st.error(f"Erreur : {e}")
            st.exception(e)

elif submit:
    st.warning("Veuillez entrer une question.")
