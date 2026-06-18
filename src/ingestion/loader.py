"""
src/ingestion/loader.py
Chargement de documents multi-format : PDF, TXT, CSV, JSON
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


class Document:
    """Représente un document chargé avec son contenu et ses métadonnées."""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content  = content
        self.metadata = metadata
    
    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return f"Document(source={self.metadata.get('source', '?')}, preview='{preview}...')"


class DocumentLoader:
    """
    Charge des documents depuis le disque en fonction de leur extension.
    Supporte : PDF, TXT, CSV, JSON
    """

    def load_directory(self, directory: str | Path) -> List[Document]:
        """Charge tous les documents supportés d'un répertoire."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Répertoire introuvable : {directory}")
        
        docs = []
        for path in directory.rglob("*"):
            if path.is_file():
                try:
                    loaded = self.load_file(path)
                    docs.extend(loaded)
                    logger.info(f"✓ Chargé : {path.name} ({len(loaded)} doc(s))")
                except Exception as e:
                    logger.warning(f"✗ Ignoré {path.name} : {e}")
        
        logger.info(f"Total : {len(docs)} documents chargés")
        return docs

    def load_file(self, path: str | Path) -> List[Document]:
        """Dispatch vers le bon loader selon l'extension."""
        path = Path(path)
        ext  = path.suffix.lower()
        
        loaders = {
            ".pdf":  self._load_pdf,
            ".txt":  self._load_txt,
            ".csv":  self._load_csv,
            ".json": self._load_json,
        }
        
        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(f"Extension non supportée : {ext}")
        
        return loader(path)

    # ── Loaders spécifiques ──────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> List[Document]:
        """Charge un PDF page par page avec PyMuPDF."""
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError("Installez PyMuPDF : pip install pymupdf")
        
        docs = []
        with fitz.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:  # Ignore les pages vides
                    docs.append(Document(
                        content=text,
                        metadata={
                            "source":   path.name,
                            "type":     "pdf",
                            "page":     page_num,
                            "total_pages": len(pdf),
                        }
                    ))
        return docs

    def _load_txt(self, path: Path) -> List[Document]:
        """Charge un fichier texte."""
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        return [Document(
            content=text,
            metadata={"source": path.name, "type": "txt"}
        )]

    def _load_csv(self, path: Path) -> List[Document]:
        """
        Charge un CSV et convertit chaque ligne en document textuel.
        Idéal pour des logs de transactions ou alertes de fraude.
        """
        df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        docs = []
        
        for idx, row in df.iterrows():
            # Formatage lisible pour le LLM
            lines = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            text  = "\n".join(lines)
            docs.append(Document(
                content=text,
                metadata={
                    "source": path.name,
                    "type":   "csv",
                    "row_id": idx,
                    "columns": list(df.columns),
                }
            ))
        return docs

    def _load_json(self, path: Path) -> List[Document]:
        """Charge un JSON (liste d'objets ou objet unique)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        
        # Liste d'objets → un document par item
        if isinstance(data, list):
            return [
                Document(
                    content=json.dumps(item, ensure_ascii=False, indent=2),
                    metadata={"source": path.name, "type": "json", "item_index": i}
                )
                for i, item in enumerate(data)
            ]
        # Objet unique → un seul document
        return [Document(
            content=json.dumps(data, ensure_ascii=False, indent=2),
            metadata={"source": path.name, "type": "json"}
        )]
