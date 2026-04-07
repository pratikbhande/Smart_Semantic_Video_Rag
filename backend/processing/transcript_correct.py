"""
GPT-4o-mini correction pass: fixes proper nouns, domain terms, and transcription errors.
Uses tiktoken windowing to stay within context limits.
"""
import asyncio
import json
import logging
from typing import List, Optional

import tiktoken
from openai import AsyncOpenAI

from backend.config import settings
from backend.models.schemas import TranscriptSegment

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


async def correct_transcript(
    segments: List[TranscriptSegment],
    glossary_names: Optional[str] = None,
    glossary_terms: Optional[str] = None,
    domain_context: Optional[str] = None,
) -> List[TranscriptSegment]:
    """
    Correct transcript segments using GPT-4o-mini in parallel windows.
    Returns corrected segments with same timing.
    """
    if not segments:
        return segments

    glossary_block = ""
    if glossary_names:
        glossary_block += f"\nProper nouns to preserve exactly: {glossary_names}"
    if glossary_terms:
        glossary_block += f"\nDomain terms: {glossary_terms}"
    if domain_context:
        glossary_block += f"\nDomain context: {domain_context}"

    system_prompt = f"""You are a transcript editor. Fix spelling, grammar, and transcription errors in the provided transcript segments.
{glossary_block}

Rules:
- Preserve all timestamps exactly
- Only fix clear transcription errors (wrong words, merged words, spelling)
- Keep the same segment structure
- Return valid JSON array matching the input format
- Do not add or remove segments"""

    max_tokens = settings.max_tokens_correction
    windows = _create_windows(segments, max_tokens)
    client = _get_client()

    async def correct_window(window: List[TranscriptSegment]) -> List[TranscriptSegment]:
        try:
            window_json = [
                {"id": i, "start": s.start_time, "end": s.end_time, "text": s.text}
                for i, s in enumerate(window)
            ]
            resp = await client.chat.completions.create(
                model=settings.mini_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Correct these segments:\n{json.dumps(window_json, indent=2)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            result = json.loads(resp.choices[0].message.content)
            corrected_window = result.get("segments", result) if isinstance(result, dict) else result

            if isinstance(corrected_window, list):
                return [
                    TranscriptSegment(
                        video_id=orig.video_id,
                        start_time=orig.start_time,
                        end_time=orig.end_time,
                        text=corr.get("text", orig.text) if isinstance(corr, dict) else orig.text,
                        confidence=orig.confidence,
                    )
                    for orig, corr in zip(window, corrected_window)
                ]
            return window
        except Exception as e:
            logger.warning(f"Correction window failed: {e}, keeping originals")
            return window

    # Run all windows in parallel
    results = await asyncio.gather(*[correct_window(w) for w in windows])

    # Flatten, preserving original order
    corrected_all: List[TranscriptSegment] = []
    for segs in results:
        corrected_all.extend(segs)

    return corrected_all


def _create_windows(
    segments: List[TranscriptSegment], max_tokens: int
) -> List[List[TranscriptSegment]]:
    """Split segments into token-bounded windows."""
    windows = []
    current_window = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _count_tokens(seg.text) + 20  # overhead per segment
        if current_tokens + seg_tokens > max_tokens and current_window:
            windows.append(current_window)
            current_window = []
            current_tokens = 0
        current_window.append(seg)
        current_tokens += seg_tokens

    if current_window:
        windows.append(current_window)

    return windows
