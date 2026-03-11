"""
Visual analysis: EasyOCR + batched GPT-4o vision, running CONCURRENTLY for speed.

OCR strategy:
- Fast PIL pre-filter (~10ms/frame) identifies frames likely to have text overlays
- Full EasyOCR runs only on text-candidate frames → reduces OCR load by ~80%
- OCR and GPT-4o vision run concurrently; results merged afterwards
- Speaker name detection from lower-third region (bottom 40% of frame)
- Confidence threshold lowered to 0.3 for better recall on name plates
"""
import asyncio
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from openai import AsyncOpenAI

from backend.config import settings
from backend.models.schemas import KeyframeData
from backend.processing.lower_third_scan import get_shared_ocr_reader

logger = logging.getLogger(__name__)

_gpt4o_global_sem = asyncio.Semaphore(settings.gpt4o_vision_concurrency)

_client = None
# Thread pool for CPU-bound OCR work
_ocr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr")


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


def _image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Fast PIL text-likelihood pre-filter ──────────────────────────────────────

def _has_text_likelihood(frame_path: str) -> bool:
    """
    Fast (~10ms) heuristic: does the lower 40% of the frame contain high-contrast
    regions that suggest text overlays (lower-thirds, subtitles, title cards)?

    Uses PIL to compute edge density in the bottom portion. Text creates sharp
    light/dark transitions detectable as edge energy.
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np

        img = Image.open(frame_path).convert("L").resize((160, 90))
        h = 90
        lower = img.crop((0, int(h * 0.55), 160, h))  # bottom 45%
        edges = lower.filter(ImageFilter.FIND_EDGES)
        arr = np.array(edges, dtype=float)
        # High std dev in edge image = lots of sharp transitions = likely text
        contrast_score = arr.std()
        return contrast_score > 20.0
    except Exception:
        return True  # fail-safe: assume text present


def _batch_text_prefilter(frames: List[KeyframeData]) -> List[bool]:
    """Run fast PIL pre-filter on all frames. Returns bool list."""
    return [_has_text_likelihood(f.frame_path) for f in frames]


# ── EasyOCR per-frame ─────────────────────────────────────────────────────────

def _ocr_one_frame(frame: KeyframeData) -> Dict[str, Any]:
    """
    Run EasyOCR on a single frame.
    Returns dict with text, tier, speaker_names, lower_third_text.
    """
    reader = get_shared_ocr_reader()
    if not reader or not Path(frame.frame_path).exists():
        return _empty_ocr()
    try:
        ocr_out = reader.readtext(frame.frame_path, detail=1)

        # Get image height for lower-third boundary
        try:
            from PIL import Image
            with Image.open(frame.frame_path) as img:
                img_h = img.size[1]
        except Exception:
            img_h = 720

        lower_third_y = img_h * 0.60  # bottom 40% = "lower third" region

        all_texts = []
        lower_items = []

        for bbox, text, conf in ocr_out:
            if conf < 0.30:  # lowered from 0.5 — better recall for name plates
                continue
            all_texts.append(text)
            top_y = bbox[0][1]
            if top_y >= lower_third_y:
                lower_items.append((text, conf, top_y))

        avg_conf = (
            sum(t[2] for t in ocr_out) / len(ocr_out) if ocr_out else 0.0
        )
        combined = " | ".join(all_texts)

        if len(all_texts) > 10:
            tier = "text_heavy"
        elif len(all_texts) > 3:
            tier = "mixed"
        else:
            tier = "visual"

        lower_items.sort(key=lambda x: x[2])  # sort by y position
        lower_third_text = " | ".join(t[0] for t in lower_items)
        speaker_names = _extract_speaker_names(lower_items)

        return {
            "text": combined,
            "confidence": avg_conf,
            "tier": tier,
            "speaker_names": speaker_names,
            "lower_third_text": lower_third_text,
        }
    except Exception as e:
        logger.debug(f"OCR error for {frame.frame_path}: {e}")
        return _empty_ocr()


def _empty_ocr() -> Dict[str, Any]:
    return {
        "text": "", "confidence": 0.0, "tier": "visual",
        "speaker_names": [], "lower_third_text": ""
    }


def _extract_speaker_names(lower_items: List[Tuple]) -> List[str]:
    """
    Heuristic: name plates are 1-6 words, ≤60 chars, not all-caps long strings.
    Filter out common UI elements and subtitle words.
    """
    skip_patterns = {
        "cc", "subtitles", "captions", "live", "breaking",
        "www.", "http", "@", "subscribe", "like", "follow"
    }
    candidates = []
    for text, conf, _ in lower_items:
        cleaned = text.strip()
        word_count = len(cleaned.split())
        if 1 <= word_count <= 6 and len(cleaned) <= 60:
            lower = cleaned.lower()
            if not any(p in lower for p in skip_patterns):
                candidates.append(cleaned)
    return candidates


def _batch_ocr_smart(frames: List[KeyframeData]) -> List[Dict[str, Any]]:
    """
    Smart OCR: fast PIL pre-filter first, then EasyOCR only on text-candidate frames.
    Non-candidate frames get empty OCR result.
    Reduces EasyOCR calls by ~80% on typical talking-head / presentation videos.
    """
    text_candidates = _batch_text_prefilter(frames)
    results = []
    n_ocr = 0
    for frame, is_candidate in zip(frames, text_candidates):
        if is_candidate:
            result = _ocr_one_frame(frame)
            n_ocr += 1
        else:
            result = _empty_ocr()
        results.append(result)
    logger.info(f"Smart OCR: ran EasyOCR on {n_ocr}/{len(frames)} frames (pre-filter saved {len(frames)-n_ocr})")
    return results


# ── Main entry point ──────────────────────────────────────────────────────────

async def analyze_keyframes(
    frames: List[KeyframeData],
    transcript_context: str = "",
    domain_context: str = "",
    skip_ocr: bool = False,  # kept for API compat, ignored
) -> List[KeyframeData]:
    """
    Analyze keyframes: OCR and GPT-4o vision run CONCURRENTLY for maximum speed.
    OCR uses smart pre-filtering to minimize EasyOCR calls.
    Results are merged: OCR speaker names supplement GPT-4o visual analysis.
    """
    if not frames:
        return frames

    loop = asyncio.get_event_loop()

    # Launch OCR and GPT-4o CONCURRENTLY
    ocr_task = loop.run_in_executor(_ocr_executor, _batch_ocr_smart, frames)
    gpt_task = asyncio.create_task(
        _batch_gpt4o_vision(frames, transcript_context, domain_context)
    )

    # Wait for both
    ocr_results, gpt_frames = await asyncio.gather(ocr_task, gpt_task)

    # Merge: inject OCR speaker data into GPT-4o results
    merged = _merge_results(gpt_frames, ocr_results)
    return merged


def _merge_results(
    gpt_frames: List[KeyframeData],
    ocr_results: List[Dict[str, Any]],
) -> List[KeyframeData]:
    """
    Merge OCR speaker names + lower-third text into GPT-4o visual analysis.
    GPT-4o ran without OCR hints (for speed), so we supplement here.
    """
    merged = []
    for frame, ocr in zip(gpt_frames, ocr_results):
        va = dict(frame.visual_analysis)

        # Supplement extracted_text with OCR if GPT-4o missed it
        if ocr.get("text") and not va.get("extracted_text"):
            va["extracted_text"] = ocr["text"]

        # Speaker name: prefer any already found by GPT-4o, fill with OCR
        if not va.get("speaker_name") and ocr.get("speaker_names"):
            va["speaker_name"] = ocr["speaker_names"][0]

        va["speaker_names_ocr"] = ocr.get("speaker_names", [])
        va["lower_third_text"] = ocr.get("lower_third_text", "")
        va["ocr_tier"] = ocr.get("tier", "visual")

        merged.append(frame.model_copy(update={"visual_analysis": va}))
    return merged


# ── GPT-4o vision (no OCR pre-hints — runs concurrently with OCR) ────────────

async def _batch_gpt4o_vision(
    frames: List[KeyframeData],
    transcript_context: str,
    domain_context: str,
) -> List[KeyframeData]:
    """Send frames in batches to GPT-4o vision. No OCR hints needed (merged afterwards)."""
    batch_size = settings.max_frames_per_batch

    batches = [
        frames[i: i + batch_size]
        for i in range(0, len(frames), batch_size)
    ]

    async def analyze_batch(batch: List[KeyframeData], batch_idx: int) -> List[KeyframeData]:
        async with _gpt4o_global_sem:
            return await _call_gpt4o_batch(batch, batch_idx * batch_size, transcript_context, domain_context)

    results_nested = await asyncio.gather(
        *[analyze_batch(b, i) for i, b in enumerate(batches)],
        return_exceptions=True,
    )

    analyzed = []
    for i, r in enumerate(results_nested):
        if isinstance(r, Exception):
            logger.error(f"Vision batch {i} failed: {r}")
            analyzed.extend(batches[i])
        else:
            analyzed.extend(r)
    return analyzed


async def _call_gpt4o_batch(
    batch: List[KeyframeData],
    global_offset: int,
    transcript_context: str,
    domain_context: str,
) -> List[KeyframeData]:
    client = _get_client()

    messages_content = []
    if transcript_context:
        messages_content.append({
            "type": "text",
            "text": f"Transcript context: {transcript_context[:500]}"
        })
    if domain_context:
        messages_content.append({
            "type": "text",
            "text": f"Domain: {domain_context}"
        })

    frame_refs = []
    for local_i, frame in enumerate(batch):
        if not Path(frame.frame_path).exists():
            continue
        try:
            b64 = _image_to_b64(frame.frame_path)
            messages_content.append({
                "type": "text",
                "text": f"Frame {local_i} @ {frame.timestamp:.1f}s"
            })
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
            frame_refs.append((local_i, frame))
        except Exception as e:
            logger.debug(f"Failed to encode frame {frame.frame_path}: {e}")

    if not frame_refs:
        return batch

    system_prompt = """Analyze each video frame. Return a JSON object with an "analyses" array.
Each analysis MUST have:
- "frame_index": int (match the Frame N index provided)
- "scene_type": one of [slide, code, diagram, face, outdoor, indoor, text_overlay, mixed]
- "description": 1-2 sentences describing content
- "extracted_text": ALL visible text — slides, subtitles, lower-thirds, name plates, overlays
- "speaker_name": name of speaker shown in lower-third overlay (e.g. "MADHAVI" or "John Smith | CEO")
  — look carefully at the BOTTOM of the frame for name plates. Return empty string if none visible.
- "people_present": bool
- "technical_content": bool
- "key_objects": list of 3-5 key visual elements

CRITICAL: Lower-third name plates (speaker identifications) appear in the BOTTOM portion of the frame.
They are often white or colored text on a dark strip. Extract them precisely."""

    try:
        resp = await _get_client().chat.completions.create(
            model=settings.vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages_content}
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.2,
        )
        result = json.loads(resp.choices[0].message.content)
        analyses = result.get("analyses", [])
        by_idx = {a["frame_index"]: a for a in analyses if isinstance(a, dict)}

        analyzed = []
        for local_i, orig_frame in frame_refs:
            a = by_idx.get(local_i, {})
            analyzed.append(orig_frame.model_copy(update={
                "scene_type": a.get("scene_type", "unknown"),
                "visual_analysis": {
                    "description": a.get("description", ""),
                    "extracted_text": a.get("extracted_text", ""),
                    "speaker_name": a.get("speaker_name", ""),
                    "people_present": a.get("people_present", False),
                    "technical_content": a.get("technical_content", False),
                    "key_objects": a.get("key_objects", []),
                }
            }))
        return analyzed

    except Exception as e:
        logger.error(f"GPT-4o vision call failed: {e}")
        return batch
