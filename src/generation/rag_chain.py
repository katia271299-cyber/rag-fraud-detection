"""
src/generation/rag_chain.py
Le cœur du projet : pipeline RAG complet.

Flux :
    Question → Embedding → Retrieval → Prompt enrichi → LLM → Réponse + Sources
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.generation.llm import BaseLLM
from src.retrieval.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Réponse complète du pipeline RAG, avec sources et métriques."""
    answer:     str
    sources:    List[RetrievalResult]
    question:   str
    latency_ms: float
    prompt_used: Optional[str] = None  # Pour le debug

    def format_sources(self) -> str:
        """Formate les sources pour affichage."""
        lines = []
        for i, src in enumerate(self.sources, 1):
            meta = src.document.metadata
            source_name = meta.get("source", "inconnu")
            page = f", p.{meta['page']}" if "page" in meta else ""
            lines.append(f"[{i}] {source_name}{page} (score: {src.score:.2f})")
        return "\n".join(lines)


class RAGChain:
    """
    Pipeline RAG : récupère les chunks pertinents et les injecte dans le prompt.
    
    Design pattern :
        1. Embed la question utilisateur
        2. Retrouve les top_k chunks les plus proches (retrieval)
        3. Construit un prompt avec le contexte récupéré
        4. Génère une réponse avec le LLM
        5. Retourne réponse + sources citées
    """

    def __init__(
        self,
        vector_store,
        llm: BaseLLM,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        system_prompt: Optional[str] = None,
    ):
        self.vector_store         = vector_store
        self.llm                  = llm
        self.top_k                = top_k
        self.similarity_threshold = similarity_threshold
        self.system_prompt        = system_prompt or self._default_system_prompt()

    def ask(self, question: str) -> RAGResponse:
        """
        Pose une question et retourne une réponse avec sources.
        
        Args:
            question: Question en langage naturel
            
        Returns:
            RAGResponse avec réponse, sources, et métriques
        """
        t0 = time.perf_counter()
        
        # ── 1. Retrieval ────────────────────────────────────────────────────
        logger.info(f"Retrieval pour : '{question[:80]}...'")
        all_results = self.vector_store.search(question, top_k=self.top_k)
        
        # Filtrer par score de similarité minimum
        results = [r for r in all_results if r.score >= self.similarity_threshold]
        
        if not results:
            logger.warning("Aucun document pertinent trouvé pour cette question.")
            return RAGResponse(
                answer="Je n'ai pas trouvé d'information pertinente dans les documents pour répondre à cette question.",
                sources=[],
                question=question,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        
        # ── 2. Construction du prompt ────────────────────────────────────────
        context = self._build_context(results)
        prompt  = self._build_prompt(question, context)
        
        # ── 3. Génération ────────────────────────────────────────────────────
        logger.info(f"Génération avec {len(results)} chunks de contexte...")
        answer = self.llm.generate(prompt, system_prompt=self.system_prompt)
        
        latency = (time.perf_counter() - t0) * 1000
        logger.info(f"Réponse générée en {latency:.0f}ms")
        
        return RAGResponse(
            answer=answer,
            sources=results,
            question=question,
            latency_ms=latency,
            prompt_used=prompt,
        )

    def _build_context(self, results: List[RetrievalResult]) -> str:
        """Formate les chunks récupérés en contexte lisible pour le LLM."""
        context_parts = []
        for i, result in enumerate(results, 1):
            meta        = result.document.metadata
            source_name = meta.get("source", "document inconnu")
            page_info   = f", page {meta['page']}" if "page" in meta else ""
            
            context_parts.append(
                f"[Source {i} — {source_name}{page_info}, score: {result.score:.2f}]\n"
                f"{result.document.content}"
            )
        
        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Construit le prompt final avec le contexte injecté."""
        return f"""Voici des extraits de documents pertinents pour répondre à la question :

=== CONTEXTE ===
{context}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
Réponds en te basant uniquement sur le contexte fourni ci-dessus.
Cite les sources entre crochets (ex: [Source 1], [Source 2]).
Si le contexte ne contient pas l'information, indique-le clairement.
"""

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "Tu es un assistant expert en analyse financière et détection de fraude. "
            "Tu réponds uniquement en te basant sur les documents fournis en contexte. "
            "Si l'information n'est pas dans le contexte, dis-le clairement. "
            "Cite toujours les sources utilisées. "
            "Réponds en français, de manière précise et structurée."
        )
