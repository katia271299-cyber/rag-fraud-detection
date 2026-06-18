"""
src/ingestion/chunker.py
Découpe les documents en chunks avec chevauchement pour préserver le contexte.
"""
import logging
from typing import List

from src.ingestion.loader import Document

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Découpe des documents en chunks de taille fixe avec recouvrement.
    
    Pourquoi le chevauchement (overlap) ?
    → Une information clé peut être à cheval entre deux chunks.
      Le chevauchement garantit qu'elle apparaît entière dans au moins l'un d'eux.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Args:
            chunk_size:    Taille max d'un chunk en caractères (~tokens)
            chunk_overlap: Nombre de caractères partagés entre chunks consécutifs
        """
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Découpe une liste de documents en chunks."""
        chunks = []
        for doc in documents:
            doc_chunks = self._split_text(doc.content, doc.metadata)
            chunks.extend(doc_chunks)
            logger.debug(f"{doc.metadata.get('source', '?')} → {len(doc_chunks)} chunks")
        
        logger.info(f"Chunking : {len(documents)} docs → {len(chunks)} chunks")
        return chunks

    def _split_text(self, text: str, base_metadata: dict) -> List[Document]:
        """Découpe un texte en chunks avec recouvrement."""
        if len(text) <= self.chunk_size:
            return [Document(content=text, metadata={**base_metadata, "chunk_id": 0, "total_chunks": 1})]
        
        chunks    = []
        start     = 0
        chunk_id  = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Essayer de couper à la fin d'une phrase ou d'un mot
            if end < len(text):
                # Préférence : couper à un saut de ligne
                cut = text.rfind("\n", start, end)
                if cut == -1 or (end - cut) > 100:
                    # Sinon : couper à un espace
                    cut = text.rfind(" ", start, end)
                if cut != -1 and cut > start:
                    end = cut
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Document(
                    content=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_id":     chunk_id,
                        "char_start":   start,
                        "char_end":     end,
                    }
                ))
                chunk_id += 1
            
            # Avancer en tenant compte du chevauchement
            start = end - self.chunk_overlap
        
        # Ajouter le total de chunks dans les métadonnées
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
