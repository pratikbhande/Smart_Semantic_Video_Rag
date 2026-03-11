"""
Hierarchical chunk creation: summary-level (per topic) + detail-level (per segment window).
For visual-only videos (no transcript), builds rich detail chunks from keyframe analysis.
"""
import json
import logging
from typing import List

from backend.models.schemas import (
    TranscriptSegment, TopicSegment, KeyframeData, ChunkData, ChunkType
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backend.processing.lower_third_scan import SpeakerEvent

logger = logging.getLogger(__name__)

DETAIL_WINDOW_SECONDS = 30.0


def build_chunks(
    video_id: str,
    segments: List[TranscriptSegment],
    topics: List[TopicSegment],
    keyframes: List[KeyframeData],
    speaker_events=None,
    filename: str = "",
) -> List[ChunkData]:
    chunks: List[ChunkData] = []
    events = speaker_events or []

    # ── Video-level overview chunk (always first, always indexed) ────────────
    # This is the PRIMARY retrieval target for cross-video queries.
    # "What is MCP?" should hit the MCP video's overview chunk before anything else.
    overview = _build_video_overview_chunk(
        video_id, filename, topics, segments, keyframes, events
    )
    chunks.append(overview)

    if not segments:
        # Visual-only: build rich chunks from keyframe analysis
        visual_chunks = _build_visual_chunks(video_id, topics, keyframes, events)
        chunks.extend(visual_chunks)
        logger.info(
            f"Built {len(chunks)} visual-only chunks "
            f"({sum(1 for c in chunks if c.chunk_type == ChunkType.summary)} summary, "
            f"{sum(1 for c in chunks if c.chunk_type == ChunkType.detail)} detail)"
        )
        if events:
            chunks.append(_build_speaker_timeline_chunk(video_id, events))
        return chunks

    seg_index = _build_segment_index(segments)

    for topic_idx, topic in enumerate(topics):
        topic_segs = _get_segments_in_range(segments, topic.start_time, topic.end_time)
        topic_frames = _get_frames_in_range(keyframes, topic.start_time, topic.end_time)
        topic_events = _get_events_in_range(events, topic.start_time, topic.end_time)

        summary_text = _build_summary_text(topic, topic_segs, topic_frames, topic_events)
        visual_ctx = _build_visual_context(topic_frames)
        chunk_id = f"{video_id}_topic_{topic_idx}_summary"

        # Segment timestamps stored in summary metadata so _refine_to_segment
        # can snap primary_timestamp to an exact spoken line even from a summary chunk.
        summary_seg_times = json.dumps([
            {"start": s.start_time, "end": s.end_time, "text": s.text[:80]}
            for s in topic_segs[:40]
        ])

        chunks.append(ChunkData(
            id=chunk_id,
            text=summary_text,
            chunk_type=ChunkType.summary,
            start_time=topic.start_time,
            end_time=topic.end_time,
            topic_title=topic.title,
            visual_context=visual_ctx,
            metadata={
                "video_id": video_id,
                "topic_index": topic_idx,
                "key_entities": topic.key_entities,
                "frame_count": len(topic_frames),
                "segment_timestamps": summary_seg_times,
            }
        ))

        detail_chunks = _build_detail_chunks(
            video_id, topic, topic_segs, topic_frames, topic_idx, events
        )
        chunks.extend(detail_chunks)

    # Global speaker timeline chunk — enables "who spoke when" queries across all topics
    if events:
        chunks.append(_build_speaker_timeline_chunk(video_id, events))

    logger.info(
        f"Built {len(chunks)} chunks "
        f"({sum(1 for c in chunks if c.chunk_type == ChunkType.summary)} summary, "
        f"{sum(1 for c in chunks if c.chunk_type == ChunkType.detail)} detail)"
    )
    return chunks


# ── Video-level overview chunk ────────────────────────────────────────────────

def _build_video_overview_chunk(
    video_id: str,
    filename: str,
    topics: List[TopicSegment],
    segments: List[TranscriptSegment],
    keyframes: List[KeyframeData],
    speaker_events=None,
) -> ChunkData:
    """
    One comprehensive chunk covering the ENTIRE video — audio + visual combined.

    This is the PRIMARY retrieval target for cross-video queries.
    When someone asks "what is MCP?", this chunk (which explicitly states
    all topics, key concepts, and the opening transcript) should rank #1
    for the MCP video and far above any chunk from an unrelated video.

    Includes:
    - Filename (often contains the topic, e.g. "mcp_tutorial.webm")
    - All topic titles + summaries
    - All key entities from all topics (the concepts discussed)
    - Full transcript in timestamped format (up to 120 segments)
    - All unique OCR text from slides/screen across the whole video
    - Speaker names
    """
    events = speaker_events or []

    # All key entities from all topics — deduplicated
    all_entities: List[str] = []
    seen_ents: set = set()
    for t in topics:
        for e in (t.key_entities or []):
            if e not in seen_ents:
                all_entities.append(e)
                seen_ents.add(e)

    # Topic outline with timestamps + summaries
    topic_outline = []
    for t in topics:
        line = f"  [{t.start_time:.0f}s–{t.end_time:.0f}s] {t.title}"
        if t.summary:
            line += f": {t.summary[:150]}"
        topic_outline.append(line)

    # Full transcript in timestamped format — up to 120 segments
    transcript_lines = "\n".join(
        f"[{s.start_time:.1f}s] {s.text}" for s in segments[:120]
    )

    # All unique OCR text from the entire video (slides, code, UI, captions)
    video_end = topics[-1].end_time if topics else (
        segments[-1].end_time if segments else 9999.0
    )
    all_ocr = _collect_topic_ocr_texts(events, 0.0, video_end)

    # All speaker names (keyframe OCR + dense scan)
    speaker_names = _collect_speaker_names(keyframes)
    for e in events:
        if e.speaker_name and e.speaker_name not in speaker_names:
            speaker_names.append(e.speaker_name)

    # Build the overview text
    label = filename if filename else video_id
    parts = [f"[VIDEO OVERVIEW: {label}]"]

    if all_entities:
        parts.append(f"Topics and concepts covered: {', '.join(all_entities[:30])}")

    if topic_outline:
        parts.append("Sections:\n" + "\n".join(topic_outline))

    if speaker_names:
        parts.append(f"Speakers: {', '.join(speaker_names[:8])}")

    if transcript_lines:
        parts.append(f"Full transcript:\n{transcript_lines}")
    elif keyframes:
        # Visual-only: use GPT-4o frame descriptions instead
        descs = []
        for f in keyframes[:20]:
            d = f.visual_analysis.get("description", "")
            if d:
                descs.append(f"[{f.timestamp:.1f}s] {d}")
        if descs:
            parts.append("Visual content:\n" + "\n".join(descs))

    if all_ocr:
        parts.append(f"On-screen text (slides, code, UI):\n{all_ocr}")

    # Segment timestamps — first 40 segments for _refine_to_segment precision
    seg_times = json.dumps([
        {"start": s.start_time, "end": s.end_time, "text": s.text[:80]}
        for s in segments[:40]
    ])

    return ChunkData(
        id=f"{video_id}_overview",
        text="\n".join(parts),
        chunk_type=ChunkType.summary,
        start_time=0.0,
        end_time=video_end if video_end < 9999.0 else 0.0,
        topic_title="Video Overview",
        metadata={
            "video_id": video_id,
            "topic_count": len(topics),
            "segment_timestamps": seg_times,
        }
    )


# ── Visual-only chunk builders ────────────────────────────────────────────────

def _build_visual_chunks(
    video_id: str,
    topics: List[TopicSegment],
    keyframes: List[KeyframeData],
    speaker_events=None,
) -> List[ChunkData]:
    """
    For videos with no transcript, build chunks entirely from visual analysis.
    Creates one summary + multiple 30s detail chunks per topic.
    """
    if not keyframes:
        return []

    # If no real topics (just fallback "Full Video"), create from keyframes directly
    chunks = []
    for topic_idx, topic in enumerate(topics):
        topic_frames = _get_frames_in_range(keyframes, topic.start_time, topic.end_time)
        if not topic_frames:
            continue

        # Summary chunk
        topic_events = _get_events_in_range(speaker_events or [], topic.start_time, topic.end_time)
        summary_text = _build_visual_summary_text(topic, topic_frames, topic_events)
        visual_ctx = _build_visual_context(topic_frames)
        chunks.append(ChunkData(
            id=f"{video_id}_topic_{topic_idx}_summary",
            text=summary_text,
            chunk_type=ChunkType.summary,
            start_time=topic.start_time,
            end_time=topic.end_time,
            topic_title=topic.title,
            visual_context=visual_ctx,
            metadata={
                "video_id": video_id,
                "topic_index": topic_idx,
                "frame_count": len(topic_frames),
            }
        ))

        # Detail chunks — 30s windows of keyframe visual analysis + dense OCR text
        window_start = topic.start_time
        window_idx = 0
        while window_start < topic.end_time:
            window_end = min(window_start + DETAIL_WINDOW_SECONDS, topic.end_time)
            window_frames = _get_frames_in_range(topic_frames, window_start, window_end)
            window_events = _get_events_in_range(speaker_events or [], window_start, window_end)

            if window_frames or window_events:
                detail_text = _build_rich_visual_text(
                    window_frames, window_start, window_end, window_events
                )
                seg_times = json.dumps([
                    {"start": f.timestamp, "end": min(f.timestamp + 5.0, window_end)}
                    for f in window_frames
                ] if window_frames else [
                    {"start": e.timestamp, "end": min(e.timestamp + DETAIL_WINDOW_SECONDS, window_end)}
                    for e in window_events[:1]
                ])
                chunks.append(ChunkData(
                    id=f"{video_id}_topic_{topic_idx}_visual_{window_idx}",
                    text=detail_text,
                    chunk_type=ChunkType.detail,
                    start_time=window_start,
                    end_time=window_end,
                    topic_title=topic.title,
                    visual_context=_build_visual_context(window_frames),
                    metadata={
                        "video_id": video_id,
                        "topic_index": topic_idx,
                        "window_index": window_idx,
                        "frame_count": len(window_frames),
                        "segment_timestamps": seg_times,
                    }
                ))

            window_start = window_end
            window_idx += 1

    return chunks


def _build_visual_summary_text(topic: TopicSegment, frames: List[KeyframeData], topic_events=None) -> str:
    """Summary chunk text for visual-only topic."""
    parts = [f"[Visual topic: {topic.title}]"]
    if topic.summary and topic.summary != "Complete video content.":
        parts.append(f"Summary: {topic.summary}")

    # Collect all speaker names across all frames
    speaker_names = _collect_speaker_names(frames)
    if speaker_names:
        parts.append(f"Speakers detected: {', '.join(speaker_names[:5])}")

    # Scene types
    scene_types = list(set(f.scene_type for f in frames if f.scene_type and f.scene_type != "unknown"))
    if scene_types:
        parts.append(f"Content types: {', '.join(scene_types)}")

    # All screen text from dense scan (primary source — covers full video)
    if topic_events:
        all_ocr = _collect_topic_ocr_texts(
            topic_events, topic.start_time, topic.end_time
        )
        if all_ocr:
            parts.append(f"All screen text:\n{all_ocr}")
    else:
        # Fall back to keyframe OCR text
        seen_texts: set = set()
        kf_texts: List[str] = []
        for f in frames:
            txt = f.visual_analysis.get("extracted_text", "").strip()
            if txt and txt not in seen_texts:
                kf_texts.append(txt[:150])
                seen_texts.add(txt)
        if kf_texts:
            parts.append(f"On-screen text: {' | '.join(kf_texts[:5])}")

    # Brief description of first few frames
    descs = []
    for f in frames[:5]:
        d = f.visual_analysis.get("description", "")
        if d:
            descs.append(f"@{f.timestamp:.1f}s: {d}")
    if descs:
        parts.append("Visual descriptions:\n" + "\n".join(descs))

    if topic_events:
        for e in topic_events:
            mins = int(e.timestamp // 60)
            secs = int(e.timestamp % 60)
            if e.speaker_name not in str(parts):
                parts.append(f"Speaker at {mins:02d}:{secs:02d}: {e.display()}")

    return "\n".join(parts)


def _build_rich_visual_text(
    frames: List[KeyframeData], window_start: float, window_end: float,
    window_events=None,
) -> str:
    """
    Detail chunk text for visual-only window.
    Includes: dense OCR text from all scan events + keyframe GPT-4o descriptions.
    """
    parts = [f"[Visual content: {window_start:.1f}s – {window_end:.1f}s]"]

    # ── Speaker identifications from dense scan ──────────────────────────
    if window_events:
        for e in (window_events or []):
            if e.speaker_name:
                mins = int(e.timestamp // 60)
                secs = int(e.timestamp % 60)
                parts.append(f"[{mins:02d}:{secs:02d}] Speaker on screen: {e.display()}")

    # ── All screen text from dense OCR scan (slides, code, UI, captions) ──
    window_ocr = _collect_window_ocr_texts(window_events or [], window_start, window_end)
    if window_ocr:
        parts.append(f"Screen content:\n{window_ocr}")

    # ── Keyframe scene descriptions from GPT-4o ──────────────────────────
    for f in frames:
        desc = f.visual_analysis.get("description", "")
        extracted_text = f.visual_analysis.get("extracted_text", "").strip()
        speaker = f.visual_analysis.get("speaker_name", "").strip()
        ocr_names = f.visual_analysis.get("speaker_names_ocr", [])
        tech = f.visual_analysis.get("technical_content", False)
        key_objects = f.visual_analysis.get("key_objects", [])

        line = f"[{f.timestamp:.1f}s]"
        if desc:
            line += f" {desc}"
        if speaker:
            line += f" [Speaker: {speaker}]"
        elif ocr_names:
            line += f" [Speaker: {ocr_names[0]}]"
        if tech:
            line += " [Technical content]"

        parts.append(line)

        # Only include keyframe OCR text if not already covered by dense scan
        if extracted_text and not window_ocr:
            parts.append(f"  Screen text: {extracted_text[:300]}")

        if key_objects:
            parts.append(f"  Objects: {', '.join(key_objects[:5])}")

    return "\n".join(parts)


# ── Transcript-based chunk builders ──────────────────────────────────────────

def _build_segment_index(segments: List[TranscriptSegment]) -> dict:
    return {s.id: s for s in segments if s.id is not None}


def _get_segments_in_range(
    segments: List[TranscriptSegment], start: float, end: float
) -> List[TranscriptSegment]:
    return [s for s in segments if s.start_time >= start - 0.5 and s.end_time <= end + 0.5]


def _get_frames_in_range(
    frames: List[KeyframeData], start: float, end: float
) -> List[KeyframeData]:
    return [f for f in frames if start - 0.5 <= f.timestamp <= end + 0.5]


def _get_events_in_range(events, start: float, end: float):
    return [e for e in events if start - 1.0 <= e.timestamp <= end + 1.0]


def _build_speaker_timeline_chunk(video_id: str, events) -> "ChunkData":
    """
    A dedicated chunk for 'who appeared when' queries.
    Directly answers: 'when did Madhavi speak', 'show me when John was on screen', etc.
    Only includes events that have a speaker name.
    """
    # Only care about events with actual speaker names
    speaker_events = [e for e in events if e.speaker_name and e.speaker_name.strip()]

    # Deduplicate by speaker name, keep first appearance
    seen: dict = {}
    for e in speaker_events:
        name_key = e.speaker_name.upper().strip()
        if name_key not in seen:
            seen[name_key] = e

    lines = ["[SPEAKER TIMELINE — complete list of on-screen speakers with timestamps]"]
    for e in sorted(seen.values(), key=lambda x: x.timestamp):
        mins = int(e.timestamp // 60)
        secs = int(e.timestamp % 60)
        lines.append(f"  {mins:02d}:{secs:02d} ({e.timestamp:.1f}s) — {e.display()}")

    # All appearances (including repeated appearances)
    if len(speaker_events) > len(seen):
        lines.append("")
        lines.append("[ALL SPEAKER APPEARANCES]")
        for e in speaker_events:
            mins = int(e.timestamp // 60)
            secs = int(e.timestamp % 60)
            lines.append(f"  {mins:02d}:{secs:02d} ({e.timestamp:.1f}s) — {e.display()}")

    text = "\n".join(lines)

    # Use speaker events range for timestamps (not all events)
    first_ts = speaker_events[0].timestamp if speaker_events else (events[0].timestamp if events else 0.0)
    last_ts = speaker_events[-1].timestamp if speaker_events else (events[-1].timestamp if events else 0.0)

    # segment_timestamps lets answer_gen snap primary_ts to exact speaker appearances
    seg_times = json.dumps([
        {"start": e.timestamp, "end": e.timestamp + 5.0, "text": e.display()}
        for e in speaker_events
    ])

    return ChunkData(
        id=f"{video_id}_speaker_timeline",
        text=text,
        chunk_type=ChunkType.summary,
        start_time=first_ts,
        end_time=last_ts,
        topic_title="Speaker Timeline",
        metadata={
            "video_id": video_id,
            "speaker_count": len(seen),
            "event_count": len(speaker_events),
            "segment_timestamps": seg_times,
            "start_time": first_ts,
            "end_time": last_ts,
        }
    )


def _collect_window_ocr_texts(events, window_start: float, window_end: float) -> str:
    """
    Collect all unique full-frame OCR text seen in a time window.
    Deduplicates: if consecutive events have same text, include only once.
    Used to embed slide/code/UI text into every chunk regardless of keyframe placement.
    Returns a single string with unique text blocks separated by newlines.
    """
    if not events:
        return ""
    window_events = _get_events_in_range(events, window_start, window_end)
    seen: set = set()
    unique_blocks: List[str] = []
    for e in window_events:
        txt = getattr(e, "full_text", "") or ""
        txt = txt.strip()
        if txt and txt not in seen:
            seen.add(txt)
            unique_blocks.append(txt)
    return "\n".join(unique_blocks)


def _collect_topic_ocr_texts(events, start: float, end: float) -> str:
    """All unique full OCR text across an entire topic (for summary chunks)."""
    if not events:
        return ""
    topic_events = _get_events_in_range(events, start, end)
    seen: set = set()
    unique: List[str] = []
    for e in topic_events:
        txt = getattr(e, "full_text", "") or ""
        txt = txt.strip()
        if txt and txt not in seen:
            seen.add(txt)
            unique.append(txt[:200])   # cap per-block to avoid embedding overflow
    return "\n---\n".join(unique[:20])  # max 20 unique text blocks per summary


def _collect_speaker_names(frames: List[KeyframeData]) -> List[str]:
    names = []
    seen = set()
    for f in frames:
        for name in [f.visual_analysis.get("speaker_name", "")] + f.visual_analysis.get("speaker_names_ocr", []):
            name = name.strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _build_summary_text(
    topic: TopicSegment,
    segments: List[TranscriptSegment],
    frames: List[KeyframeData],
    topic_events=None,
) -> str:
    parts = [f"[TOPIC: {topic.title}]"]
    if topic.summary:
        parts.append(f"Summary: {topic.summary}")
    if topic.key_entities:
        parts.append(f"Key topics: {', '.join(topic.key_entities)}")

    # Use timestamped lines (same format as detail chunks) so GPT-4o can
    # reference precise spoken moments from summary chunks too.
    if segments:
        timestamped_lines = "\n".join(
            f"[{s.start_time:.1f}s] {s.text}" for s in segments[:80]
        )
        parts.append(f"Transcript:\n{timestamped_lines}")

    scene_types = list(set(f.scene_type for f in frames if f.scene_type != "unknown"))
    if scene_types:
        parts.append(f"Visual content: {', '.join(scene_types)}")

    # Speaker names from keyframe OCR + dense scan
    speaker_names = _collect_speaker_names(frames)
    if topic_events:
        for e in topic_events:
            if e.speaker_name and e.speaker_name not in speaker_names:
                speaker_names.append(e.speaker_name)
    if speaker_names:
        parts.append(f"Speakers: {', '.join(speaker_names[:8])}")

    # Timestamped speaker identifications from dense scan
    if topic_events:
        speaker_events_with_name = [e for e in topic_events if e.speaker_name]
        if speaker_events_with_name:
            speaker_lines = []
            for e in speaker_events_with_name:
                mins = int(e.timestamp // 60)
                secs = int(e.timestamp % 60)
                speaker_lines.append(f"{mins:02d}:{secs:02d} — {e.display()}")
            parts.append("Speaker identifications:\n" + "\n".join(speaker_lines))

    # ALL screen text seen in this topic (slides, code, UI, captions)
    # This is the key data that lets the system answer "what did slide X say"
    all_ocr = _collect_topic_ocr_texts(topic_events, topic.start_time, topic.end_time)
    if all_ocr:
        parts.append(f"All screen text in this segment:\n{all_ocr}")
    else:
        # Fall back to keyframe-extracted text
        visual_texts = []
        for f in frames[:5]:
            txt = f.visual_analysis.get("extracted_text", "")
            if txt:
                visual_texts.append(txt[:150])
        if visual_texts:
            parts.append(f"On-screen text: {' | '.join(visual_texts)}")

    return "\n".join(parts)


def _build_visual_context(frames: List[KeyframeData]) -> str:
    if not frames:
        return ""
    descriptions = []
    for f in frames[:5]:
        desc = f.visual_analysis.get("description", "")
        speaker = f.visual_analysis.get("speaker_name", "")
        entry = f"@{f.timestamp:.1f}s: {desc}" if desc else f"@{f.timestamp:.1f}s"
        if speaker:
            entry += f" [Speaker: {speaker}]"
        if entry.strip():
            descriptions.append(entry)
    return " | ".join(descriptions)


def _build_detail_chunks(
    video_id: str,
    topic: TopicSegment,
    segments: List[TranscriptSegment],
    frames: List[KeyframeData],
    topic_idx: int,
    all_events=None,
) -> List[ChunkData]:
    if not segments:
        return []

    chunks = []
    window_start = topic.start_time
    window_idx = 0

    while window_start < topic.end_time:
        window_end = min(window_start + DETAIL_WINDOW_SECONDS, topic.end_time)
        window_segs = [s for s in segments
                       if s.start_time >= window_start - 1 and s.end_time <= window_end + 1]
        window_frames = _get_frames_in_range(frames, window_start, window_end)

        if window_segs:
            timestamped_lines = [f"[{s.start_time:.1f}s] {s.text}" for s in window_segs]
            transcript_text = "\n".join(timestamped_lines)
            visual_ctx = _build_visual_context(window_frames)

            window_events = _get_events_in_range(all_events or [], window_start, window_end)

            parts = [f"[{topic.title}]"]

            # ── Speaker identifications first (highest priority for "who spoke" queries) ──
            speaker_events_here = [e for e in window_events if e.speaker_name]
            if speaker_events_here:
                for e in speaker_events_here:
                    mins = int(e.timestamp // 60)
                    secs = int(e.timestamp % 60)
                    parts.append(f"[{mins:02d}:{secs:02d}] Speaker on screen: {e.display()}")

            # ── All screen text seen in this window (slides, code, UI, captions) ──
            # This makes the chunk answer "what was shown at X time" queries
            window_ocr = _collect_window_ocr_texts(window_events, window_start, window_end)
            if window_ocr:
                parts.append(f"Screen content:\n{window_ocr}")
            else:
                # Fall back to keyframe-extracted text if dense scan found nothing
                kf_texts = [
                    f.visual_analysis.get("extracted_text", "").strip()
                    for f in window_frames
                    if f.visual_analysis.get("extracted_text", "").strip()
                ]
                if kf_texts:
                    parts.append(f"Screen content: {' | '.join(kf_texts[:3])}")

            # ── Transcript ──
            parts.append(transcript_text)

            if visual_ctx:
                parts.append(f"Visual context: {visual_ctx[:300]}")

            # ── Speaker names summary ──
            window_speakers = _collect_speaker_names(window_frames)
            for e in window_events:
                if e.speaker_name and e.speaker_name not in window_speakers:
                    window_speakers.append(e.speaker_name)
            if window_speakers:
                parts.append(f"Speakers visible: {', '.join(window_speakers[:5])}")

            seg_times = json.dumps([
                {"start": s.start_time, "end": s.end_time, "text": s.text[:80]}
                for s in window_segs
            ])

            chunk_id = f"{video_id}_topic_{topic_idx}_detail_{window_idx}"
            chunks.append(ChunkData(
                id=chunk_id,
                text="\n".join(parts),
                chunk_type=ChunkType.detail,
                start_time=window_start,
                end_time=window_end,
                topic_title=topic.title,
                visual_context=visual_ctx,
                metadata={
                    "video_id": video_id,
                    "topic_index": topic_idx,
                    "window_index": window_idx,
                    "segment_count": len(window_segs),
                    "frame_count": len(window_frames),
                    "segment_timestamps": seg_times,
                }
            ))

        window_start = window_end
        window_idx += 1

    return chunks
