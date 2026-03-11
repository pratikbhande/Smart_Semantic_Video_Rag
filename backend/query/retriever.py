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
    Conservatively expand query: expand acronyms and add exact alternative names only.
    Does NOT add broader related topics — keeps semantic meaning tight so the right
    video is retrieved, not a thematically adjacent one.
    Falls back to original question on any error.
    """
    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.mini_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Given a search query, output the query plus acronym expansions and "
                        "exact alternative names for the same specific concept. "
                        "Do NOT add broader related topics or tangential concepts. "
                        "Output only one line — the enriched query, nothing else.\n"
                        "Examples:\n"
                        "  'what is MCP?' → 'what is MCP Model Context Protocol'\n"
                        "  'explain RAG' → 'explain RAG retrieval augmented generation'\n"
                        "  'how does RLHF work' → 'how does RLHF reinforcement learning human feedback work'"
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        expansion = resp.choices[0].message.content.strip()
        # Only use expansion if it's a modest addition (not a complete rewrite)
        if len(expansion) < len(question) * 4:
            return expansion
        return question
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

    expanded_q = await expand_query(question)
    query_embedding = await embed_text(expanded_q)

    results = await search_chunks(query_embedding, video_id, top_k=k)

    logger.info(f"Retrieved {len(results)} candidates for '{question[:50]}...'")
    return results


async def retrieve_multi(
    video_ids: List[str],
    question: str,
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Embed the question once, search all videos in parallel, merge by similarity.
    Returns top_k*2 candidates for the reranker to score.
    """
    k = top_k or settings.retrieval_top_k

    expanded_q = await expand_query(question)
    query_embedding = await embed_text(expanded_q)

    # Search all videos in parallel — faster than sequential for 10 videos
    tasks = [search_chunks(query_embedding, vid, top_k=k) for vid in video_ids]
    per_video_results = await asyncio.gather(*tasks)

    all_results: List[Dict[str, Any]] = []
    for results in per_video_results:
        all_results.extend(results)

    # Sort by similarity — best chunks rise to top regardless of which video
    all_results.sort(key=lambda x: x["similarity"], reverse=True)

    logger.info(
        f"Multi-retrieve: {len(all_results)} raw candidates from {len(video_ids)} videos "
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
