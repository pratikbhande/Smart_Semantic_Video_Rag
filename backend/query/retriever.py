"""
Retriever: embed question, search both ChromaDB collections, merge + dedup results.
"""
import asyncio
import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI

from backend.config import settings
from backend.indexing.embedder import embed_text
from backend.indexing.vector_store import search_chunks, get_video_top_score, get_top_chunk_for_video

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def expand_query(question: str) -> str:
    """
    Expand query with related concepts and synonyms using GPT-4o-mini.
    Returns an enriched query string for better semantic embedding coverage.
    Falls back to the original question on any error.
    """
    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.mini_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer for a video content retrieval system. "
                        "Rewrite the user's question as an enriched search query by adding: "
                        "related concepts, synonyms, alternative phrasings, and domain-specific terms. "
                        "Output ONLY the enriched query — one line, no explanation, no bullet points."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=120,
            temperature=0.2,
        )
        expansion = resp.choices[0].message.content.strip()
        return f"{question} {expansion}"
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}, using original question")
        return question


async def retrieve(
    video_id: str,
    question: str,
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Embed the question and retrieve relevant chunks from ChromaDB.
    Returns list of result dicts with similarity scores.
    """
    k = top_k or settings.retrieval_top_k

    # Expand query for richer semantic coverage, then embed
    expanded_q = await expand_query(question)
    query_embedding = await embed_text(expanded_q)

    # Search vector store
    results = await search_chunks(query_embedding, video_id, top_k=k)

    logger.info(f"Retrieved {len(results)} candidates for '{question[:50]}...'")
    return results


async def retrieve_multi(
    video_ids: List[str],
    question: str,
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Embed the question once and retrieve from multiple videos, merging results.
    Returns list of result dicts sorted by similarity, with video_id in metadata.
    """
    k = top_k or settings.retrieval_top_k

    # Expand query once, embed once — shared across all videos
    expanded_q = await expand_query(question)
    query_embedding = await embed_text(expanded_q)

    # Search each video and merge
    all_results: List[Dict[str, Any]] = []
    for vid in video_ids:
        results = await search_chunks(query_embedding, vid, top_k=k)
        all_results.extend(results)

    # Sort by similarity descending; return top_k*2 to give reranker more signal
    all_results.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(
        f"Multi-retrieve: {len(all_results)} candidates from {len(video_ids)} videos "
        f"for '{question[:50]}...'"
    )
    return all_results[: k * 2]


async def discover_videos(
    video_ids: List[str],
    question: str,
    min_score: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Search all provided video_ids and return per-video relevance scores.
    Used for "which video talks about X?" queries.
    Returns list of {video_id, score, top_chunk} sorted by score descending.
    """
    query_embedding = await embed_text(question)

    # Score all videos in parallel
    async def score_video(vid: str) -> Dict[str, Any]:
        score = await get_video_top_score(query_embedding, vid, top_k=5)
        top_chunk = await get_top_chunk_for_video(query_embedding, vid) if score > min_score else None
        return {"video_id": vid, "score": score, "top_chunk": top_chunk}

    results = await asyncio.gather(*[score_video(vid) for vid in video_ids])
    results = [r for r in results if r["score"] > min_score]
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        f"Video discovery: {len(results)}/{len(video_ids)} videos relevant "
        f"for '{question[:50]}...'"
    )
    return results
