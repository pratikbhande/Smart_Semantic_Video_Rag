"""
GPT-4o-mini topic segmentation: identifies topic boundaries and titles from transcript.
"""
import json
import logging
from typing import List

from openai import AsyncOpenAI

from backend.config import settings
from backend.models.schemas import TranscriptSegment, TopicSegment

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def segment_topics(
    segments: List[TranscriptSegment],
    video_id: str,
    duration: float,
    domain_context: str = "",
) -> List[TopicSegment]:
    """
    Identify topic segments from transcript using GPT-4o-mini.
    Returns list of TopicSegment with boundaries merged to min duration.
    """
    if not segments:
        return [_fallback_topic(video_id, 0.0, duration)]

    # Build compact transcript for segmentation
    transcript_text = _build_transcript_text(segments)

    system_prompt = f"""You are a video content analyzer. Identify distinct topic segments in this transcript.
{f'Domain context: {domain_context}' if domain_context else ''}

Return a JSON object with a "topics" array. Each topic has:
- "start_time": float (seconds)
- "end_time": float (seconds)
- "title": short descriptive title (5-10 words)
- "summary": 1-2 sentence summary
- "key_entities": list of important names, terms, concepts mentioned

Rules:
- Minimum topic duration: {settings.topic_min_duration} seconds
- Cover the full video duration (0 to {duration:.1f}s)
- Merge short adjacent topics if needed
- 3-15 topics total depending on content"""

    try:
        client = _get_client()
        resp = await client.chat.completions.create(
            model=settings.mini_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript:\n{transcript_text[:8000]}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=2000,
        )
        result = json.loads(resp.choices[0].message.content)
        raw_topics = result.get("topics", [])

        topics = []
        for t in raw_topics:
            start = float(t.get("start_time", 0))
            end = float(t.get("end_time", duration))
            if end - start < 5.0:  # skip tiny fragments
                continue
            topics.append(TopicSegment(
                video_id=video_id,
                start_time=round(start, 2),
                end_time=round(end, 2),
                title=t.get("title", "Unknown Topic"),
                summary=t.get("summary", ""),
                key_entities=t.get("key_entities", []),
            ))

        if not topics:
            return [_fallback_topic(video_id, 0.0, duration)]

        # Fix coverage: ensure topics cover full video
        topics = _fix_coverage(topics, 0.0, duration, video_id)
        topics = _merge_short_topics(topics, settings.topic_min_duration, video_id)

        logger.info(f"Segmented into {len(topics)} topics for {video_id}")
        return topics

    except Exception as e:
        logger.error(f"Topic segmentation failed: {e}")
        return [_fallback_topic(video_id, 0.0, duration)]


def _build_transcript_text(segments: List[TranscriptSegment]) -> str:
    lines = []
    for seg in segments:
        ts = f"[{_fmt_time(seg.start_time)}-{_fmt_time(seg.end_time)}]"
        lines.append(f"{ts} {seg.text}")
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fallback_topic(video_id: str, start: float, end: float) -> TopicSegment:
    return TopicSegment(
        video_id=video_id, start_time=start, end_time=end,
        title="Full Video", summary="Complete video content.",
        key_entities=[]
    )


def _fix_coverage(
    topics: List[TopicSegment], start: float, end: float, video_id: str
) -> List[TopicSegment]:
    """Ensure topics cover [start, end] with no gaps."""
    if not topics:
        return [_fallback_topic(video_id, start, end)]

    topics.sort(key=lambda t: t.start_time)

    # Clamp to [start, end]
    topics[0] = topics[0].model_copy(update={"start_time": start})
    topics[-1] = topics[-1].model_copy(update={"end_time": end})

    # Fill gaps
    filled = [topics[0]]
    for t in topics[1:]:
        prev = filled[-1]
        if t.start_time > prev.end_time + 0.5:
            # Insert gap filler
            filled.append(TopicSegment(
                video_id=video_id,
                start_time=prev.end_time, end_time=t.start_time,
                title="Continuation", summary="",
                key_entities=[]
            ))
        filled.append(t)

    return filled


def _merge_short_topics(
    topics: List[TopicSegment], min_duration: float, video_id: str
) -> List[TopicSegment]:
    """Merge topics shorter than min_duration with their neighbor."""
    if len(topics) <= 1:
        return topics

    merged = []
    i = 0
    while i < len(topics):
        t = topics[i]
        if (t.end_time - t.start_time) < min_duration and i + 1 < len(topics):
            # Merge with next
            nxt = topics[i + 1]
            combined = TopicSegment(
                video_id=video_id,
                start_time=t.start_time, end_time=nxt.end_time,
                title=t.title,
                summary=f"{t.summary} {nxt.summary}".strip(),
                key_entities=list(set(t.key_entities + nxt.key_entities))
            )
            topics[i + 1] = combined
            i += 1
        else:
            merged.append(t)
            i += 1
    return merged
