"""
ui/app.py - Interface Streamlit simplifiée pour le pipeline RAG Finance.
Lancement : streamlit run ui/app.py
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="RAG Finance",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Finance — Détection de Fraude")
st.caption("Interrogez vos documents financiers en langage naturel.")

# ── Question ──────────────────────────────────────────────────────────────────
question = st.text_area(
    "Votre question",
    height=80,
    placeholder="Ex : Quels sont les patterns de fraude les plus fréquents ?",
)

submit = st.button("🔍 Rechercher", type="primary")

# ── Exemples ─────────────────────────────────────────────────────────────────
st.markdown("**Questions d'exemple :**")
col1, col2 = st.columns(2)
with col1:
    if st.button("Patterns de fraude fréquents ?"):
        question = "Quels sont les patterns de fraude les plus fréquents ?"
with col2:
    if st.button("Transactions suspectes élevées ?"):
        question = "Quelle est la transaction suspecte avec le montant le plus élevé ?"

# ── Pipeline (chargé à la demande) ────────────────────────────────────────────
def get_pipeline():
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
    return vector_store, chain

# ── Résultat ──────────────────────────────────────────────────────────────────
if submit and question.strip():
    with st.spinner("Recherche en cours... (première fois : 30-60 secondes)"):
        try:
            vector_store, chain = get_pipeline()

            if vector_store.count() == 0:
                st.warning("⚠️ Aucun document indexé. Lancez d'abord : python scripts/ingest_data.py")
            else:
                response = chain.ask(question)

                st.success(f"✅ Réponse en {response.latency_ms:.0f}ms — {len(response.sources)} sources")

                st.subheader("💬 Réponse")
                st.markdown(response.answer)

                if response.sources:
                    st.subheader(f"📚 Sources ({len(response.sources)})")
                    for i, src in enumerate(response.sources, 1):
                        with st.expander(f"[{i}] {src.document.metadata.get('source', '?')} — score {src.score:.2f}"):
                            st.write(src.document.content[:400] + "...")

        except Exception as e:
            st.error(f"Erreur : {e}")
            st.exception(e)

elif submit:
    st.warning("Veuillez entrer une question.")
