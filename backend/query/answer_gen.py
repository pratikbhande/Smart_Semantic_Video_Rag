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

    # Build context.
    # Multi-video: include full video_id (opaque hash) in header for timestamp attribution.
    # Filename is intentionally NOT included — GPT-4o will cite filenames in the answer if given.
    # Single-video: no source label needed at all.
    context_limit = 10 if is_multi else 8
    context_parts = []
    for chunk in chunks[:context_limit]:
        meta = chunk.get("metadata", {})
        start = float(meta.get("start_time", 0))
        end = float(meta.get("end_time", 0))
        topic = meta.get("topic_title", "")
        vid = meta.get("video_id", "")
        text = chunk.get("text", "")[:2500]
        if is_multi:
            header = f"[{vid} | {start:.1f}s–{end:.1f}s | {topic}]"
        else:
            header = f"[{start:.1f}s–{end:.1f}s | {topic}]"
        context_parts.append(f"{header}\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    if is_visual_only:
        content_type_note = (
            "The context contains visual analysis: OCR text, scene descriptions, "
            "and speaker names extracted from the video frames. No audio transcript."
        )
        ts_note = "the timestamp of the most relevant keyframe"
    else:
        content_type_note = (
            "The context contains spoken transcript lines prefixed with exact timestamps like [47.3s]. "
            "Each [Xs] marker is the precise second that content appears."
        )
        ts_note = (
            "the exact [Xs] timestamp of the single transcript line that most directly answers the question — "
            "NOT the chunk start/end boundary time"
        )

    if is_multi:
        ts_format = '{"start": float, "end": float, "relevance": 0-1, "video_id": "exact_video_id_from_header"}'
        source_vid_fmt = '"source_video_id": "exact_video_id_from_the_header_of_the_passage_that_most_directly_answers"'
        supporting_fmt = (
            '"supporting_videos": [{"video_id": "exact_video_id_from_header", '
            '"snippet": "one sentence of what relevant content exists on this topic", '
            '"timestamp": float}, ...]'
        )
    else:
        ts_format = '{"start": float, "end": float, "relevance": 0-1}'
        source_vid_fmt = '"source_video_id": null'
        supporting_fmt = '"supporting_videos": []'

    system_prompt = f"""You are a knowledgeable expert. You have been given excerpts from recorded content as your knowledge base. Answer the user's question using the information in those excerpts.

{content_type_note}

RULES:
- Answer DIRECTLY as a subject matter expert. Write about the topic itself.
- NEVER write "Primary Video:", "Also covered in:", "In this recording...", "This video discusses..."
- NEVER mention filenames, video names, or refer to a "video" as a source
- Use markdown (bold, bullets, headers) to organize the topic content
- Quote or paraphrase actual spoken/written content from the excerpts
- Be comprehensive: include specific facts, terminology, examples, and details

Return a JSON object with these exact fields:
- "answer": your expert answer about the topic — no source commentary, no meta-references
- {source_vid_fmt}
- "primary_timestamp": float — {ts_note}
- "timestamps": array of {ts_format} — up to 6 most relevant moments ordered by relevance
- {supporting_fmt}

Rules for primary_timestamp:
- Must NOT be 0.0 unless the answer genuinely begins at the very start
- Use the exact [Xs] value from the transcript, not the chunk boundary time
- Search all excerpts before concluding content is absent"""

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
            max_tokens=2500,
        )
        result = json.loads(resp.choices[0].message.content)

        raw_timestamps = result.get("timestamps", [])
        timestamps = []

        for t in raw_timestamps:
            if isinstance(t, dict):
                try:
                    ts_vid_id = t.get("video_id") or None
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

        # --- Step 1: Determine source_video_id ---
        # Priority: GPT-4o's explicit source_video_id > chunk-based derivation.
        # GPT-4o reads the context headers (which include the full video_id hash)
        # and can directly identify which passage answers the question.
        source_vid = None
        if is_multi:
            gpt_source = result.get("source_video_id", "")
            if gpt_source and gpt_source in (video_map or {}):
                source_vid = gpt_source

        # Fallback: find which chunk's time range contains primary_ts
        if not source_vid and primary_ts is not None:
            for c in chunks:
                meta = c.get("metadata", {})
                cs = float(meta.get("start_time", 0))
                ce = float(meta.get("end_time", cs + 35))
                if cs - 2.0 <= primary_ts <= ce + 2.0:
                    source_vid = meta.get("video_id")
                    break

        # Fallback: use video_id from the highest-relevance timestamp
        if not source_vid and timestamps:
            pt = primary_ts or 0.0
            closest = min(timestamps, key=lambda t: abs(t.start - pt))
            source_vid = closest.video_id

        # Final fallback: top reranked chunk
        if not source_vid:
            source_vid = chunks[0].get("metadata", {}).get("video_id") if chunks else None

        # --- Step 2: Validate / snap primary_ts using SOURCE VIDEO's timestamps only ---
        # By scoping to the source video, we avoid snapping to timestamps from other videos.
        source_chunks = [
            c for c in chunks
            if c.get("metadata", {}).get("video_id") == source_vid
        ] if source_vid else chunks

        known_ts: List[float] = []
        for c in source_chunks:
            meta = c.get("metadata", {})
            seg_json = meta.get("segment_timestamps", "")
            if seg_json:
                try:
                    segs = json.loads(seg_json)
                    known_ts.extend(float(s["start"]) for s in segs if "start" in s)
                except Exception:
                    pass
            known_ts.append(float(meta.get("start_time", 0)))

        known_ts = sorted(set(known_ts))

        if known_ts:
            if primary_ts is not None and primary_ts != 0.0:
                within_range = any(
                    float(c.get("metadata", {}).get("start_time", 0)) - 1.0
                    <= primary_ts
                    <= float(c.get("metadata", {}).get("end_time", 0)) + 1.0
                    for c in source_chunks
                )
                if not within_range:
                    primary_ts = min(known_ts, key=lambda t: abs(t - primary_ts))
            elif primary_ts is None or primary_ts == 0.0:
                # Default: first segment of the source video's top chunk (not t=0)
                first_seg = next(
                    (t for t in known_ts if t > 0.5),
                    known_ts[0] if known_ts else None
                )
                if first_seg is not None:
                    primary_ts = first_seg

        # Refine: snap from chunk boundary to most query-relevant exact segment
        if primary_ts is not None:
            primary_ts = _refine_to_segment(question, primary_ts, source_chunks)

        # Inject fallback timestamps from source video's top chunks if LLM returned none
        if not timestamps and source_chunks:
            for c in source_chunks[:3]:
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

        # Parse supporting_videos returned by GPT-4o
        supporting_videos = []
        for sv in result.get("supporting_videos", []):
            if not isinstance(sv, dict):
                continue
            full_id = sv.get("video_id", "")
            if not full_id or full_id == source_vid:
                continue  # skip primary video and invalid entries
            supporting_videos.append({
                "video_id": full_id,
                "video_filename": (video_map or {}).get(full_id, ""),
                "snippet": sv.get("snippet", ""),
                "timestamp": float(sv.get("timestamp", 0)),
            })

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
            supporting_videos=supporting_videos,
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
