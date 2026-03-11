"""
Parallel chunked transcription using OpenAI Whisper API, then merge segments.
"""
import asyncio
import logging
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from backend.config import settings
from backend.models.schemas import TranscriptSegment
from backend.processing.audio_extract import extract_audio_chunk, split_audio_chunks

logger = logging.getLogger(__name__)

MAX_CONCURRENT = 3  # Conservative concurrency to avoid Whisper 429 rate limits


async def transcribe_audio(
    wav_path: str,
    video_id: str,
    duration: float,
    glossary_prompt: Optional[str] = None,
) -> Tuple[List[TranscriptSegment], bool]:
    """
    Transcribe a WAV file in parallel chunks. Returns (segments, had_errors).
    had_errors=True means at least one chunk failed due to an API error (not silence).
    """
    chunks = split_audio_chunks(wav_path, duration)
    logger.info(f"Transcribing {len(chunks)} chunk(s) for {video_id}")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        _transcribe_chunk(wav_path, start, end, i, video_id, glossary_prompt, sem)
        for i, (start, end, _) in enumerate(chunks)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_segments: List[TranscriptSegment] = []
    had_errors = False
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Chunk {i} transcription failed: {r}")
            had_errors = True
            continue
        all_segments.extend(r)

    merged = _merge_segments(all_segments)
    logger.info(f"Transcription complete: {len(merged)} segments (had_errors={had_errors})")
    return merged, had_errors


async def _transcribe_chunk(
    wav_path: str, start: float, end: float, idx: int,
    video_id: str, prompt: Optional[str], sem: asyncio.Semaphore
) -> List[TranscriptSegment]:
    async with sem:
        # Extract chunk to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            await extract_audio_chunk(wav_path, start, end, tmp.name)
            segments = await _call_whisper(tmp.name, start, video_id, prompt)
            return segments
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


async def _call_whisper(
    chunk_path: str, time_offset: float, video_id: str, prompt: Optional[str]
) -> List[TranscriptSegment]:
    """Call OpenAI Whisper API with verbose_json to get segment-level timestamps.
    Retries up to 3 times on 429 (rate limit) with exponential backoff."""
    with open(chunk_path, "rb") as f:
        audio_bytes = f.read()

    data = {
        "model": settings.whisper_model,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "language": "en",
    }
    if prompt:
        data["prompt"] = prompt[:224]

    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}

    last_exc = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                )
                if resp.status_code == 429:
                    wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    logger.warning(f"Whisper 429 rate limit — retrying in {wait}s (attempt {attempt+1}/3)")
                    await asyncio.sleep(wait)
                    last_exc = Exception(f"429 after {attempt+1} attempts")
                    continue
                resp.raise_for_status()
                result = resp.json()
                break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 10 * (2 ** attempt)
                logger.warning(f"Whisper 429 — retrying in {wait}s (attempt {attempt+1}/3)")
                await asyncio.sleep(wait)
                last_exc = e
                continue
            raise
    else:
        raise last_exc or Exception("Whisper API failed after 3 attempts")

    segments = []
    for seg in result.get("segments", []):
        adjusted_start = seg["start"] + time_offset
        adjusted_end = seg["end"] + time_offset
        text = seg["text"].strip()
        if not text:
            continue
        confidence = seg.get("avg_logprob", 0.0)
        # Convert log prob to 0-1 range (logprob is typically -2 to 0)
        conf_score = max(0.0, min(1.0, (confidence + 2.0) / 2.0))
        segments.append(TranscriptSegment(
            video_id=video_id,
            start_time=round(adjusted_start, 3),
            end_time=round(adjusted_end, 3),
            text=text,
            confidence=round(conf_score, 3),
        ))
    return segments


def _merge_segments(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Sort by start time and remove duplicates from overlapping chunks."""
    if not segments:
        return []

    segments.sort(key=lambda s: s.start_time)
    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]
        # Skip near-duplicate segments (overlap from chunking)
        if seg.start_time < prev.end_time - 0.5:
            # Keep the one with higher confidence
            if seg.confidence > prev.confidence:
                merged[-1] = seg
            continue
        merged.append(seg)

    return merged
