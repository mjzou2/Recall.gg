"""Sentence-transformer embedding service for semantic search.

Uses all-MiniLM-L6-v2 (384 dimensions) on CPU to avoid VRAM conflicts
with whisperX. Module-level lazy singleton — loads on first call.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_model = None
_model_load_attempted = False

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIMENSION = 384


def _get_model():
    """Lazy-load the sentence-transformer model on CPU."""
    global _model, _model_load_attempted
    if _model_load_attempted:
        return _model
    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
        logger.info("Loaded embedding model: %s (dim=%d)", MODEL_NAME, MODEL_DIMENSION)
    except Exception as exc:
        logger.warning("Failed to load embedding model '%s': %s", MODEL_NAME, exc)
        _model = None
    return _model


def embed_text(text: str) -> Optional[List[float]]:
    """Embed a single text string. Returns None if model unavailable."""
    model = _get_model()
    if model is None:
        return None
    try:
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as exc:
        logger.warning("Embedding failed for text (len=%d): %s", len(text), exc)
        return None


def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """Batch-embed multiple texts. Returns None if model unavailable."""
    if not texts:
        return []
    model = _get_model()
    if model is None:
        return None
    try:
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]
    except Exception as exc:
        logger.warning("Batch embedding failed for %d texts: %s", len(texts), exc)
        return None
