"""
OpenAI text-embedding-3-large async batch embedder.
"""
import asyncio
import logging
from typing import List, Tuple

from openai import AsyncOpenAI

from backend.config import settings
from backend.models.schemas import ChunkData

logger = logging.getLogger(__name__)

_client = None
BATCH_SIZE = 50  # OpenAI embedding batch limit


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def embed_text(text: str) -> List[float]:
    """Embed a single text string."""
    results = await embed_texts([text])
    return results[0]


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts in batches."""
    if not texts:
        return []

    all_embeddings = []
    client = _get_client()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        try:
            resp = await client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
                dimensions=settings.embedding_dims,
            )
            batch_embeddings = [e.embedding for e in resp.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Embedding batch {i//BATCH_SIZE} failed: {e}")
            # Return zero vectors for failed batch
            zero_vec = [0.0] * settings.embedding_dims
            all_embeddings.extend([zero_vec] * len(batch))

    return all_embeddings


async def embed_chunks(chunks: List[ChunkData]) -> List[Tuple[ChunkData, List[float]]]:
    """
    Embed all chunks, returning (chunk, embedding) pairs.
    """
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    embeddings = await embed_texts(texts)

    result = []
    for chunk, emb in zip(chunks, embeddings):
        result.append((chunk, emb))

    logger.info(f"Embedded {len(result)} chunks")
    return result
