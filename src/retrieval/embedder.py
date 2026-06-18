"""
src/retrieval/embedder.py
Crée des embeddings vectoriels via Ollama (sans PyTorch).
Utilise le modèle nomic-embed-text directement depuis Ollama.
"""
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Produit des embeddings via Ollama — aucune dépendance PyTorch.
    Utilise nomic-embed-text, un modèle léger et efficace.
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self._dim       = None
        self._verify()

    def _verify(self):
        """Vérifie qu'Ollama est accessible."""
        import requests
        try:
            requests.get("http://localhost:11434", timeout=3)
            logger.info(f"Ollama connecté — modèle {self.model_name} prêt.")
        except Exception:
            logger.warning("Ollama non accessible sur localhost:11434")

    def _embed_one(self, text: str) -> np.ndarray:
        import requests
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model_name, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        logger.info(f"Embedding de {len(texts)} textes via Ollama...")
        vectors = []
        for i, text in enumerate(texts):
            vec = self._embed_one(text)
            vectors.append(vec)
            if (i + 1) % 20 == 0:
                logger.info(f"  {i+1}/{len(texts)} embeddings produits...")
        result = np.stack(vectors)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-9)
        return result

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._embed_one(query)
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-9)

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self._embed_one("test"))
        return self._dim