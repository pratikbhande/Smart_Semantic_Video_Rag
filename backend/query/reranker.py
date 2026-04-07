"""
GPT-4o-mini reranking: score retrieved chunks by relevance to the question.
"""
import json
import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def rerank(
    question: str,
    candidates: List[Dict[str, Any]],
    top_k: int = None,
) -> List[Dict[str, Any]]:
    """
    Rerank candidates using GPT-4o-mini. Returns top_k results sorted by relevance.
    """
    k = top_k or settings.rerank_top_k

    if not candidates:
        return []

    if len(candidates) <= k:
        for c in candidates:
            c["rerank_score"] = c.get("similarity", 0.0)
            c["final_score"] = c.get("similarity", 0.0)
        return candidates[:k]

    # Score top 20 candidates — enough signal without overwhelming the LLM
    rerank_window = min(len(candidates), 20)

    chunks_for_rerank = []
    for i, c in enumerate(candidates[:rerank_window]):
        meta = c.get("metadata", {})
        chunks_for_rerank.append({
            "index": i,
            "start": meta.get("start_time", 0),
            "end": meta.get("end_time", 0),
            "topic": meta.get("topic_title", ""),
            "text": c.get("text", "")[:400],
        })

    system_prompt = (
        "You are a relevance judge for video Q&A. Given a question and candidate passages, "
        "score each 0-10 for how directly it answers the question.\n\n"
        "Return JSON: {\"scores\": [{\"index\": 0, \"score\": 8.5}, ...]}\n"
        "Score 10 = directly and specifically answers the question\n"
        "Score 5  = topically related but not a direct answer\n"
        "Score 0  = completely irrelevant\n"
        "Score every passage."
    )

    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.mini_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{json.dumps(chunks_for_rerank, indent=2)[:3500]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=800,
        )
        result = json.loads(resp.choices[0].message.content)
        scores = {s["index"]: s["score"] / 10.0 for s in result.get("scores", [])}

        for i, c in enumerate(candidates[:rerank_window]):
            c["rerank_score"] = scores.get(i, c.get("similarity", 0.0))

        # Combine: embedding similarity (grounding) + LLM relevance score (precision)
        for c in candidates[:rerank_window]:
            c["final_score"] = (
                0.4 * c.get("similarity", 0.0) +
                0.6 * c.get("rerank_score", 0.0)
            )

        candidates[:rerank_window] = sorted(
            candidates[:rerank_window], key=lambda x: x["final_score"], reverse=True
        )

    except Exception as e:
        logger.warning(f"Reranking failed: {e}, using similarity scores")
        for c in candidates:
            c["final_score"] = c.get("similarity", 0.0)

    return candidates[:k]
