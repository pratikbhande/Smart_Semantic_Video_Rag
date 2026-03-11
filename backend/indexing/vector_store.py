"""
ChromaDB two-collection strategy: _summary and _detail collections per video.
"""
import logging
import os
from typing import List, Tuple, Dict, Any, Optional

# Suppress ChromaDB telemetry before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb

from backend.config import settings
from backend.models.schemas import ChunkData, ChunkType

logger = logging.getLogger(__name__)

_chroma_client = None


def _get_client() -> chromadb.Client:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _chroma_client


def _get_collection(name: str):
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


async def index_chunks(
    embedded_chunks: List[Tuple[ChunkData, List[float]]],
    video_id: str,
) -> None:
    """Index embedded chunks into the appropriate ChromaDB collections."""
    if not embedded_chunks:
        return

    summary_ids, summary_embeddings, summary_docs, summary_metas = [], [], [], []
    detail_ids, detail_embeddings, detail_docs, detail_metas = [], [], [], []

    for chunk, embedding in embedded_chunks:
        meta = {
            "video_id": video_id,
            "chunk_type": chunk.chunk_type.value,
            "start_time": chunk.start_time,
            "end_time": chunk.end_time,
            "topic_title": chunk.topic_title or "",
            "visual_context": (chunk.visual_context or "")[:500],
            **{k: str(v) for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
        }

        if chunk.chunk_type == ChunkType.summary:
            summary_ids.append(chunk.id)
            summary_embeddings.append(embedding)
            summary_docs.append(chunk.text[:2000])
            summary_metas.append(meta)
        else:
            detail_ids.append(chunk.id)
            detail_embeddings.append(embedding)
            detail_docs.append(chunk.text[:2000])
            detail_metas.append(meta)

    summary_col = _get_collection(settings.summary_collection)
    detail_col = _get_collection(settings.detail_collection)

    if summary_ids:
        summary_col.upsert(
            ids=summary_ids,
            embeddings=summary_embeddings,
            documents=summary_docs,
            metadatas=summary_metas,
        )
        logger.info(f"Indexed {len(summary_ids)} summary chunks")

    if detail_ids:
        detail_col.upsert(
            ids=detail_ids,
            embeddings=detail_embeddings,
            documents=detail_docs,
            metadatas=detail_metas,
        )
        logger.info(f"Indexed {len(detail_ids)} detail chunks")


async def search_chunks(
    query_embedding: List[float],
    video_id: str,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Search both collections and merge results."""
    summary_col = _get_collection(settings.summary_collection)
    detail_col = _get_collection(settings.detail_collection)

    where_filter = {"video_id": video_id}
    k_each = max(top_k // 2, 5)

    results = []

    for col, col_type in [(summary_col, "summary"), (detail_col, "detail")]:
        try:
            # Request k_each results; ChromaDB will cap naturally at what exists.
            # Don't use col.count() as a guard — it counts ALL videos, not this one.
            try:
                res = col.query(
                    query_embeddings=[query_embedding],
                    n_results=k_each,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as inner_e:
                if "Number of requested results" in str(inner_e):
                    # Fewer docs exist than requested — retry with 1 to get what's there
                    res = col.query(
                        query_embeddings=[query_embedding],
                        n_results=1,
                        where=where_filter,
                        include=["documents", "metadatas", "distances"],
                    )
                else:
                    raise

            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            distances = res.get("distances", [[]])[0]

            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                similarity = 1.0 - dist  # cosine distance → similarity
                results.append({
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "collection": col_type,
                })
        except Exception as e:
            logger.warning(f"Search in {col_type} collection failed: {e}")

    # Sort by similarity before returning — do NOT dedup across collections;
    # summary and detail chunks have different content and both add signal.
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


async def get_video_top_score(
    query_embedding: List[float],
    video_id: str,
    top_k: int = 5,
) -> float:
    """
    Return the average similarity score of the top-k results for a single video.
    Used by the video discovery endpoint to rank videos by relevance.
    """
    results = await search_chunks(query_embedding, video_id, top_k=top_k)
    if not results:
        return 0.0
    top_scores = [r["similarity"] for r in results[:3]]
    return sum(top_scores) / len(top_scores)


async def get_top_chunk_for_video(
    query_embedding: List[float],
    video_id: str,
) -> Optional[Dict[str, Any]]:
    """Return the single best matching chunk for a video."""
    results = await search_chunks(query_embedding, video_id, top_k=3)
    return results[0] if results else None


async def clear_video_index(video_id: str) -> None:
    """Remove all chunks for a video from both collections."""
    for col_name in [settings.summary_collection, settings.detail_collection]:
        try:
            col = _get_collection(col_name)
            existing = col.get(where={"video_id": video_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                col.delete(ids=ids)
                logger.info(f"Cleared {len(ids)} chunks from {col_name} for {video_id}")
        except Exception as e:
            logger.warning(f"Failed to clear {col_name} for {video_id}: {e}")


def _dedup_by_time(results: List[Dict]) -> List[Dict]:
    """Remove near-duplicate results (same time range, different collections)."""
    seen = set()
    deduped = []
    for r in results:
        meta = r.get("metadata", {})
        key = (round(float(meta.get("start_time", 0)), 1),
               round(float(meta.get("end_time", 0)), 1))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped
