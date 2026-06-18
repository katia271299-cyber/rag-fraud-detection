"""
tests/test_pipeline.py
Tests unitaires et d'intégration du pipeline RAG.

Lancement : pytest tests/ -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import Document, DocumentLoader
from src.ingestion.chunker import TextChunker
from src.generation.rag_chain import RAGChain, RAGResponse


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_documents():
    return [
        Document(
            content="La fraude par usurpation d'identité représente 38% des cas en 2023. "
                    "Les montants moyens s'élèvent à 8 400 euros par dossier.",
            metadata={"source": "rapport_2023.txt", "type": "txt"},
        ),
        Document(
            content="Le pattern de smurfing consiste à fractionner un virement important "
                    "en plusieurs petites transactions pour contourner les seuils de détection.",
            metadata={"source": "guide_fraude.txt", "type": "txt"},
        ),
        Document(
            content="Alerte ALT-2023-10042 : Virement de 95 000 euros vers un compte "
                    "en Roumanie créé il y a 12 jours. Score de fraude : 97. Statut : BLOQUÉE.",
            metadata={"source": "transactions.csv", "type": "csv", "row_id": 42},
        ),
    ]


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=200, chunk_overlap=20)


# ── Tests : Loader ────────────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt_file(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Contenu de test pour la fraude bancaire.", encoding="utf-8")
        
        loader = DocumentLoader()
        docs   = loader.load_file(txt_file)
        
        assert len(docs) == 1
        assert "fraude bancaire" in docs[0].content
        assert docs[0].metadata["source"] == "test.txt"
        assert docs[0].metadata["type"] == "txt"

    def test_load_csv_file(self, tmp_path):
        csv_file = tmp_path / "transactions.csv"
        csv_file.write_text(
            "id,montant,statut\nALT-001,5000,ALERTE\nALT-002,200,OK\n",
            encoding="utf-8",
        )
        
        loader = DocumentLoader()
        docs   = loader.load_file(csv_file)
        
        assert len(docs) == 2
        assert "ALT-001" in docs[0].content
        assert docs[0].metadata["type"] == "csv"

    def test_load_json_list(self, tmp_path):
        import json
        json_file = tmp_path / "alertes.json"
        data = [
            {"id": "A001", "montant": 50000},
            {"id": "A002", "montant": 3000},
        ]
        json_file.write_text(json.dumps(data), encoding="utf-8")
        
        loader = DocumentLoader()
        docs   = loader.load_file(json_file)
        
        assert len(docs) == 2
        assert "A001" in docs[0].content

    def test_unsupported_extension(self, tmp_path):
        bad_file = tmp_path / "test.docx"
        bad_file.write_bytes(b"fake docx content")
        
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="non supportée"):
            loader.load_file(bad_file)

    def test_load_directory(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("Premier document", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("Deuxième document", encoding="utf-8")
        (tmp_path / "ignore.xyz").write_bytes(b"ignored")
        
        loader = DocumentLoader()
        docs   = loader.load_directory(tmp_path)
        
        assert len(docs) == 2


# ── Tests : Chunker ───────────────────────────────────────────────────────────

class TestTextChunker:

    def test_short_document_not_split(self, chunker, sample_documents):
        """Un document plus court que chunk_size ne doit pas être découpé."""
        short_doc = Document(content="Texte court.", metadata={"source": "test.txt"})
        chunks    = chunker.split_documents([short_doc])
        
        assert len(chunks) == 1
        assert chunks[0].content == "Texte court."
        assert chunks[0].metadata["chunk_id"] == 0

    def test_long_document_split(self, chunker):
        """Un document long doit être découpé en plusieurs chunks."""
        long_text = "a " * 300  # 600 caractères
        long_doc  = Document(content=long_text, metadata={"source": "long.txt"})
        chunks    = chunker.split_documents([long_doc])
        
        assert len(chunks) > 1
        # Vérifier le chevauchement : le début du chunk N+1 doit ressembler à la fin du chunk N
        for i in range(len(chunks) - 1):
            assert len(chunks[i].content) <= chunker.chunk_size + 10  # Tolérance de 10

    def test_chunk_metadata_preserved(self, chunker, sample_documents):
        """Les métadonnées originales doivent être préservées dans les chunks."""
        chunks = chunker.split_documents(sample_documents)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_multiple_documents(self, chunker, sample_documents):
        """Tous les documents doivent être traités."""
        chunks = chunker.split_documents(sample_documents)
        
        sources = {c.metadata["source"] for c in chunks}
        assert "rapport_2023.txt" in sources
        assert "guide_fraude.txt" in sources


# ── Tests : RAG Chain ─────────────────────────────────────────────────────────

class TestRAGChain:

    def _make_chain(self):
        """Crée un RAGChain avec des mocks."""
        mock_vs = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "La fraude représente 38% des cas [Source 1]."
        
        # Simuler un résultat de retrieval
        from src.retrieval.vector_store import RetrievalResult
        result = RetrievalResult(
            document=Document(
                content="La fraude par usurpation d'identité représente 38%.",
                metadata={"source": "rapport_2023.txt"},
            ),
            score=0.92,
        )
        mock_vs.search.return_value = [result]
        
        return RAGChain(vector_store=mock_vs, llm=mock_llm)

    def test_ask_returns_response(self):
        chain    = self._make_chain()
        response = chain.ask("Quel est le taux de fraude par usurpation d'identité ?")
        
        assert isinstance(response, RAGResponse)
        assert len(response.answer) > 0
        assert len(response.sources) > 0
        assert response.latency_ms > 0

    def test_ask_no_results_threshold(self):
        """Si le score est sous le seuil, retourner une réponse vide."""
        mock_vs  = MagicMock()
        mock_llm = MagicMock()
        
        from src.retrieval.vector_store import RetrievalResult
        low_score_result = RetrievalResult(
            document=Document(content="Texte non pertinent.", metadata={}),
            score=0.1,  # Sous le seuil par défaut de 0.3
        )
        mock_vs.search.return_value = [low_score_result]
        
        chain    = RAGChain(vector_store=mock_vs, llm=mock_llm, similarity_threshold=0.3)
        response = chain.ask("Question sans réponse dans le corpus")
        
        assert len(response.sources) == 0
        assert "pas" in response.answer.lower() or "aucun" in response.answer.lower()
        mock_llm.generate.assert_not_called()

    def test_format_sources(self):
        chain    = self._make_chain()
        response = chain.ask("Question test")
        formatted = response.format_sources()
        
        assert "rapport_2023.txt" in formatted
        assert "0.92" in formatted


# ── Tests : API ───────────────────────────────────────────────────────────────

class TestAPI:
    """Tests d'intégration de l'API FastAPI."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        
        with patch("api.main.load_pipeline"):
            from api.main import app, vector_store, rag_chain
            # Mocker le pipeline global
        
        return TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── Point d'entrée direct ─────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
