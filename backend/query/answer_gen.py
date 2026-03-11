"""
GPT-4o answer generation with JSON timestamp output.
Falls back to GPT-4o-mini for a natural response if the main call fails.
"""
import json
import logging
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI

from backend.config import settings
from backend.models.schemas import QueryResponse, TimestampRef

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def generate_answer(
    question: str,
    chunks: List[Dict[str, Any]],
    video_duration: Optional[float] = None,
    video_map: Optional[Dict[str, str]] = None,
) -> QueryResponse:
    """
    Generate a natural answer using GPT-4o. Falls back to GPT-4o-mini on failure.
    Handles both transcript-based and visual-only (no audio) videos.
    """
    if not chunks:
        return QueryResponse(
            answer="I couldn't find relevant content in this video to answer your question.",
            timestamps=[],
            primary_timestamp=None,
            relevant_chunks=[],
        )

    is_multi = bool(video_map and len(video_map) > 1)

    # Detect if this is a visual-only video (no transcript segments in chunks)
    is_visual_only = all(
        "Visual content:" in c.get("text", "") or "Visual topic:" in c.get("text", "")
        for c in chunks[:3]
    )

    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks[:8]):
        meta = chunk.get("metadata", {})
        start = float(meta.get("start_time", 0))
        end = float(meta.get("end_time", 0))
        topic = meta.get("topic_title", "")
        vid = meta.get("video_id", "")
        fname = (video_map or {}).get(vid, vid)
        text = chunk.get("text", "")[:1000]
        source_label = f" | {fname}" if is_multi else ""
        context_parts.append(
            f"[Chunk {i+1}{source_label} | {start:.1f}s-{end:.1f}s | {topic}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    duration_info = f"Video duration: {video_duration:.1f}s. " if video_duration else ""

    if is_multi:
        ts_format = '{"start": float, "end": float, "relevance": 0-1, "video_id": "video_id_string"}'
        extra_rules = (
            "- Each timestamp MUST include the video_id field\n"
            "- Cite which video each piece of information comes from\n"
            "- When the answer is about a specific speaker/person, clearly state their timestamp and which video\n"
            "- Even if the exact question phrase isn't in the context, if you find a speaker name and timestamp, report it"
        )
    else:
        ts_format = '{"start": float, "end": float, "relevance": 0-1}'
        extra_rules = "- Only reference timestamps from the provided context"

    if is_visual_only:
        context_desc = (
            "The context contains visual analysis of video keyframes: "
            "frame descriptions, on-screen text (OCR), detected speaker names from lower-third overlays, "
            "and technical content flags. There is no audio transcript."
        )
        answer_guidance = (
            "Answer based on what is visible on screen: descriptions, on-screen text, "
            "speaker names detected in lower-thirds, and visual content. "
            "Be natural — do not expose internal labels like [TOPIC:] or [Chunk N]."
        )
    else:
        context_desc = (
            "The context contains transcript lines prefixed with EXACT timestamps like [47.3s] Speaker: text. "
            "Each [Xs] marker is the precise second that line was spoken in the video."
        )
        answer_guidance = (
            "For primary_timestamp: identify the SINGLE transcript line that most directly answers the question, "
            "then use its exact [Xs] value — NOT the chunk start/end time. "
            "If multiple lines are relevant, pick the one with the most specific, direct answer. "
            "Be natural — do not expose internal labels like [TOPIC:] or chunk boundaries."
        )

    system_prompt = f"""You are an intelligent video content assistant. {duration_info}

{context_desc}

{answer_guidance}

Return a JSON object with:
- "answer": natural, helpful answer in markdown. Do NOT include raw labels like [TOPIC: Full Video] or chunk headers.
- "timestamps": array of {ts_format}
- "primary_timestamp": float — the exact second where the most relevant content appears

Rules:
- {extra_rules}
- If asking about a specific person/speaker, use the timestamp where they appear on screen
- Search the context for any matching names, timestamps, speaker events, or screen text before saying content is absent
- If a speaker name is found in context at any timestamp, report it with that timestamp — even if surrounding text differs from the question phrasing
- For visual-only videos: use keyframe timestamps as primary_timestamp
- primary_timestamp must be a real time value from the context, not 0.0 unless content is at the start"""

    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=1500,
        )
        result = json.loads(resp.choices[0].message.content)

        raw_timestamps = result.get("timestamps", [])
        timestamps = []
        for t in raw_timestamps:
            if isinstance(t, dict):
                try:
                    ts_vid_id = t.get("video_id")
                    ts_vid_fname = (video_map or {}).get(ts_vid_id) if ts_vid_id else None
                    timestamps.append(TimestampRef(
                        start=float(t.get("start", 0)),
                        end=float(t.get("end", 0)),
                        relevance=float(t.get("relevance", 0.5)),
                        video_id=ts_vid_id,
                        video_filename=ts_vid_fname,
                    ))
                except (ValueError, TypeError):
                    continue

        primary_ts = result.get("primary_timestamp")
        if primary_ts is not None:
            try:
                primary_ts = float(primary_ts)
            except (ValueError, TypeError):
                primary_ts = None

        # Validate / snap primary_ts to known segment or frame timestamps
        all_known_ts: List[float] = []
        for c in chunks:
            meta = c.get("metadata", {})
            seg_json = meta.get("segment_timestamps", "")
            if seg_json:
                try:
                    segs = json.loads(seg_json)
                    all_known_ts.extend(float(s["start"]) for s in segs if "start" in s)
                except Exception:
                    pass
            all_known_ts.append(float(meta.get("start_time", 0)))

        all_known_ts = sorted(set(all_known_ts))

        if all_known_ts:
            if primary_ts is not None:
                within_range = any(
                    float(c.get("metadata", {}).get("start_time", 0)) - 1.0
                    <= primary_ts
                    <= float(c.get("metadata", {}).get("end_time", 0)) + 1.0
                    for c in chunks
                )
                if not within_range:
                    primary_ts = min(all_known_ts, key=lambda t: abs(t - primary_ts))
            else:
                # Default to chunk with highest score
                primary_ts = float(chunks[0].get("metadata", {}).get("start_time", 0))

        # Refine: if primary_ts is a chunk boundary, find the most query-relevant segment within it
        if primary_ts is not None:
            primary_ts = _refine_to_segment(question, primary_ts, chunks)

        # Inject fallback timestamps from top chunks if LLM returned none
        if not timestamps and chunks:
            for c in chunks[:3]:
                meta = c.get("metadata", {})
                vid_id = meta.get("video_id")
                vid_fname = (video_map or {}).get(vid_id) if vid_id else None
                timestamps.append(TimestampRef(
                    start=float(meta.get("start_time", 0)),
                    end=float(meta.get("end_time", 0)),
                    relevance=float(c.get("similarity", 0.5)),
                    video_id=vid_id,
                    video_filename=vid_fname,
                ))

        source_vid = chunks[0].get("metadata", {}).get("video_id") if chunks else None

        relevant_chunks = []
        for c in chunks[:5]:
            meta = c.get("metadata", {})
            cid = meta.get("video_id", "")
            relevant_chunks.append({
                "id": c.get("id", ""),
                "text": c.get("text", "")[:300],
                "start_time": meta.get("start_time", 0),
                "end_time": meta.get("end_time", 0),
                "topic_title": meta.get("topic_title", ""),
                "score": c.get("final_score", c.get("similarity", 0)),
                "video_id": cid,
                "video_filename": (video_map or {}).get(cid, ""),
            })

        return QueryResponse(
            answer=result.get("answer", "No answer generated."),
            timestamps=timestamps,
            primary_timestamp=primary_ts,
            relevant_chunks=relevant_chunks,
            source_video_id=source_vid,
        )

    except Exception as e:
        logger.error(f"GPT-4o answer generation failed: {e} — trying mini fallback")
        return await _mini_fallback(question, chunks, context, video_map)


def _refine_to_segment(question: str, primary_ts: float, chunks: List[Dict]) -> float:
    """
    If primary_ts lands on a chunk boundary (imprecise), keyword-score all segments
    within that chunk and snap to the most query-relevant one.

    This handles the case where GPT-4o returns a chunk's start_time instead of the
    exact spoken line — giving the user a more precise seek target.
    """
    # Collect segments from chunks whose window contains primary_ts
    candidate_segments: List[Dict] = []
    for c in chunks:
        meta = c.get("metadata", {})
        chunk_start = float(meta.get("start_time", 0))
        chunk_end = float(meta.get("end_time", chunk_start + 30))
        if chunk_start - 2.0 <= primary_ts <= chunk_end + 2.0:
            seg_json = meta.get("segment_timestamps", "")
            if seg_json:
                try:
                    candidate_segments.extend(json.loads(seg_json))
                except Exception:
                    pass

    if not candidate_segments:
        return primary_ts

    # If primary_ts already aligns with a known segment start (within 1s), it's precise
    for seg in candidate_segments:
        if abs(float(seg.get("start", -9999)) - primary_ts) < 1.0:
            return primary_ts

    # primary_ts is a chunk boundary — find segment with best keyword overlap
    # Skip stopwords and short tokens to focus on meaningful query terms
    stopwords = {"what", "when", "where", "who", "how", "did", "does", "is", "are",
                 "the", "a", "an", "in", "on", "at", "to", "of", "and", "or", "about"}
    q_tokens = {w.lower() for w in question.split() if len(w) > 3 and w.lower() not in stopwords}

    best_score = -1
    best_ts = primary_ts

    for seg in candidate_segments:
        seg_text = seg.get("text", "").lower()
        score = sum(1 for token in q_tokens if token in seg_text)
        if score > best_score:
            best_score = score
            best_ts = float(seg.get("start", primary_ts))

    if best_score > 0:
        logger.debug(f"Timestamp refined {primary_ts:.1f}s → {best_ts:.1f}s (overlap={best_score})")
        return best_ts

    return primary_ts


async def _mini_fallback(
    question: str,
    chunks: List[Dict[str, Any]],
    context: str,
    video_map: Optional[Dict[str, str]],
) -> QueryResponse:
    """
    Natural language fallback using GPT-4o-mini.
    Never exposes raw chunk text to the user.
    """
    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.mini_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a video content assistant. Answer the user's question naturally and helpfully "
                        "based on the video analysis context provided. The context may include transcript text, "
                        "visual descriptions, speaker names, and on-screen text. "
                        "Be conversational — do not expose internal labels like [TOPIC:], [Chunk N], or timestamps "
                        "unless they are directly relevant to the answer."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nVideo context:\n{context[:3000]}"
                }
            ],
            temperature=0.3,
            max_tokens=600,
        )
        answer = resp.choices[0].message.content.strip()

        # Best-effort primary timestamp from the top chunk
        primary_ts = None
        if chunks:
            meta = chunks[0].get("metadata", {})
            seg_json = meta.get("segment_timestamps", "")
            if seg_json:
                try:
                    segs = json.loads(seg_json)
                    if segs:
                        primary_ts = float(segs[0].get("start", meta.get("start_time", 0)))
                except Exception:
                    pass
            if primary_ts is None:
                primary_ts = float(meta.get("start_time", 0))

        relevant_chunks = []
        for c in chunks[:3]:
            meta = c.get("metadata", {})
            cid = meta.get("video_id", "")
            relevant_chunks.append({
                "id": c.get("id", ""),
                "text": c.get("text", "")[:300],
                "start_time": meta.get("start_time", 0),
                "end_time": meta.get("end_time", 0),
                "topic_title": meta.get("topic_title", ""),
                "score": c.get("final_score", c.get("similarity", 0)),
                "video_id": cid,
                "video_filename": (video_map or {}).get(cid, ""),
            })

        source_vid = chunks[0].get("metadata", {}).get("video_id") if chunks else None
        fallback_ts = []
        if primary_ts is not None:
            fallback_ts = [TimestampRef(
                start=primary_ts,
                end=primary_ts + 5.0,
                relevance=0.8,
                video_id=source_vid,
                video_filename=(video_map or {}).get(source_vid) if source_vid else None,
            )]

        return QueryResponse(
            answer=answer,
            timestamps=fallback_ts,
            primary_timestamp=primary_ts,
            relevant_chunks=relevant_chunks,
            source_video_id=source_vid,
        )

    except Exception as e2:
        logger.error(f"Mini fallback also failed: {e2}")
        return QueryResponse(
            answer="I'm having trouble analyzing this video right now. Please try your question again.",
            timestamps=[],
            primary_timestamp=None,
            relevant_chunks=[],
        )
