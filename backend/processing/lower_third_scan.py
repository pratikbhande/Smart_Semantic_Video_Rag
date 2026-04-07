"""
Dense full-frame video text scanner — no EasyOCR dependency.

Three-stage pipeline:
  1. ffmpeg: extract full frames at adaptive intervals (2–8s)
  2. PIL: edge-density pre-filter (skip blank frames) + pixel-diff dedup (skip unchanged)
  3. GPT-4o: extract all visible text + speaker name from candidate frames only

GPT-4o receives only text-candidate changed frames → minimal API calls.

Performance (2hr video, 5s interval):
  Stage 1 — 1440 ffmpeg seeks (parallel, semaphore 10)   ≈  30s
  Stage 2 — 1440 PIL ops (in thread pool)                ≈   2s
  Stage 3 — ~200 candidates → 20 GPT-4o batches of 10   ≈  30s  (runs concurrently with Whisper)
  Total dense scan wall-clock: ~35s, adds 0 to pipeline  ✓
"""
import asyncio
import base64
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from openai import AsyncOpenAI

from backend.config import settings

logger = logging.getLogger(__name__)

_client: Optional[AsyncOpenAI] = None
# Semaphore for GPT-4o calls from dense scan (text-only, fast)
_scan_gpt4o_sem = asyncio.Semaphore(settings.gpt4o_vision_concurrency)


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


@dataclass
class SpeakerEvent:
    """
    Timestamped text event from the dense frame scan.

    full_text  — ALL OCR text visible in this frame (slides, code, captions, UI)
    speaker_name / speaker_role — extracted from lower-third region only.
    speaker_name may be empty for frames with slide/code text but no speaker overlay.
    """
    timestamp: float
    speaker_name: str
    speaker_role: str
    raw_text: str        # kept for compat (= lower-third text)
    full_text: str = ""  # ALL text in the frame

    def display(self) -> str:
        if self.speaker_role:
            return f"{self.speaker_name} ({self.speaker_role})"
        return self.speaker_name


# ── Public entry point ────────────────────────────────────────────────────────

async def scan_lower_thirds(
    video_path: str,
    duration: float,
) -> List[SpeakerEvent]:
    """
    Scan the full video for all visible text at regular intervals.
    Returns one SpeakerEvent per unique text state, ordered by timestamp.
    Runs entirely without EasyOCR — uses PIL for detection, GPT-4o for extraction.
    """
    scan_interval = _pick_scan_interval(duration)
    timestamps = _build_timestamps(duration, scan_interval)

    logger.info(
        f"Video text scan: {len(timestamps)} samples at {scan_interval:.1f}s intervals "
        f"for {duration:.0f}s video"
    )

    with tempfile.TemporaryDirectory(prefix="vtxt_scan_") as tmp_dir:
        # Stage 1: extract frames (parallel ffmpeg seeks)
        frame_paths = await _extract_full_frames(video_path, timestamps, tmp_dir)

        # Stage 2: PIL detection in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        candidates: List[Tuple[float, str]] = await loop.run_in_executor(
            None, _pil_detect_candidates, frame_paths, timestamps
        )

        logger.info(
            f"Video text scan: {len(candidates)}/{len(timestamps)} frames pass PIL filter "
            f"(eliminated {len(timestamps) - len(candidates)})"
        )

        if not candidates:
            return []

        # Stage 3: GPT-4o text extraction (async, files still exist in tmp_dir)
        events = await _gpt4o_extract_all_text(candidates)

    logger.info(
        f"Video text scan complete: {len(events)} text events "
        f"({sum(1 for e in events if e.speaker_name)} with speaker name)"
    )
    return events


# ── Stage 1: frame extraction ─────────────────────────────────────────────────

def _pick_scan_interval(duration: float) -> float:
    if duration <= 180:    return 1.0   # ≤3min: every 1s  (catch brief lower-thirds)
    elif duration <= 600:  return 2.0   # ≤10min: every 2s
    elif duration <= 3600: return 4.0   # ≤1hr: every 4s
    else:                  return 6.0   # >1hr: every 6s


def _build_timestamps(duration: float, interval: float) -> List[float]:
    ts = []
    for early in [0.5, 1.5, 3.0]:
        if early < duration:
            ts.append(early)
    t = interval
    while t < duration:
        v = round(t, 1)
        if v not in ts:
            ts.append(v)
        t += interval
    ts.sort()
    return ts


async def _extract_full_frames(
    video_path: str,
    timestamps: List[float],
    tmp_dir: str,
) -> List[Optional[str]]:
    """Extract 640×360 full frame at each timestamp — parallel ffmpeg seeks."""
    sem = asyncio.Semaphore(10)

    async def one(ts: float, idx: int) -> Optional[str]:
        out = os.path.join(tmp_dir, f"vtxt_{idx:06d}.jpg")
        async with sem:
            cmd = [
                "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
                "-vf", "scale=640:360:force_original_aspect_ratio=decrease,"
                       "pad=640:360:(ow-iw)/2:(oh-ih)/2",
                "-frames:v", "1", "-q:v", "4", out,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
        return out if os.path.exists(out) else None

    results = await asyncio.gather(
        *[one(ts, i) for i, ts in enumerate(timestamps)],
        return_exceptions=True,
    )
    return [r if isinstance(r, str) else None for r in results]


# ── Stage 2: PIL detection ────────────────────────────────────────────────────

def _pil_detect_candidates(
    frame_paths: List[Optional[str]],
    timestamps: List[float],
) -> List[Tuple[float, str]]:
    """
    Fast PIL-only filtering — no model, no download.
    Returns (timestamp, path) pairs that are:
      a) likely to contain text (edge density > threshold on full frame)
      b) different from the previous selected frame (pixel diff > threshold)
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np
    except ImportError:
        logger.warning("PIL/numpy unavailable — returning all frames as candidates")
        return [(t, p) for t, p in zip(timestamps, frame_paths) if p]

    TEXT_EDGE_THRESHOLD = 6.0    # full-frame std of edge map; lower = more permissive
    DIFF_THRESHOLD      = 0.03   # mean pixel diff to detect frame change

    candidates: List[Tuple[float, str]] = []
    prev_arr: Optional[np.ndarray] = None

    for path, ts in zip(frame_paths, timestamps):
        if not path or not os.path.exists(path):
            prev_arr = None
            continue
        try:
            img = Image.open(path).convert("L")

            # (a) edge density on full frame
            edges = img.filter(ImageFilter.FIND_EDGES)
            arr = np.array(edges, dtype=np.float32)
            if arr.std() < TEXT_EDGE_THRESHOLD:
                # no text-like content — skip entirely, don't update prev_arr
                continue

            # (b) pixel diff — skip if same content as last candidate
            arr_gray = np.array(img, dtype=np.float32) / 255.0
            if prev_arr is not None:
                diff = float(np.mean(np.abs(arr_gray - prev_arr)))
                if diff < DIFF_THRESHOLD:
                    continue  # unchanged frame

            prev_arr = arr_gray
            candidates.append((ts, path))
        except Exception:
            continue

    return candidates


# ── Stage 3: GPT-4o text extraction ──────────────────────────────────────────

_TEXT_EXTRACT_SYSTEM = """You are extracting text from video frames for a search index.

For EACH frame provided, return:
- "frame_index": int (matches the index in the request)
- "full_text": ALL visible text in the frame joined with " | ".
  Include: slide text, bullet points, code, captions, subtitles, UI labels,
           watermarks, titles, diagrams labels. Empty string if no text visible.
- "speaker_name": a person's name visible ANYWHERE in the frame — lower-third nameplate,
  title card, on-screen label, introduction slide, or any overlay.
  Must be a proper name — ALL CAPS like "MADHAVI" or Title Case like "John Smith".
  1–4 words maximum. Empty string if no person's name is visible.
  DO NOT return subtitles, sentence fragments, generic titles, or filenames as speaker names.
- "speaker_role": job title/role alongside the speaker name (anywhere in frame). Empty string if none.

Return JSON: {"frames": [{"frame_index": int, "full_text": str, "speaker_name": str, "speaker_role": str}]}"""


async def _gpt4o_extract_all_text(
    candidates: List[Tuple[float, str]],
) -> List[SpeakerEvent]:
    """Batch candidates into GPT-4o calls, deduplicate, return events."""
    if not candidates:
        return []

    batch_size = min(settings.max_frames_per_batch, 6)  # 6 frames/batch = better accuracy
    batches = [
        candidates[i: i + batch_size]
        for i in range(0, len(candidates), batch_size)
    ]

    async def process_batch(batch: List[Tuple[float, str]]) -> List[SpeakerEvent]:
        async with _scan_gpt4o_sem:
            return await _call_gpt4o_text_batch(batch)

    results = await asyncio.gather(
        *[process_batch(b) for b in batches],
        return_exceptions=True,
    )

    # Flatten + deduplicate consecutive identical full_text
    all_events: List[SpeakerEvent] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Dense scan GPT-4o batch failed: {r}")
        else:
            all_events.extend(r)

    all_events.sort(key=lambda e: e.timestamp)

    # Dedup: skip events where BOTH full_text AND speaker_name are identical to previous
    # (same slide + same speaker = skip; same slide + NEW speaker = keep)
    deduped: List[SpeakerEvent] = []
    prev_key = ("", "")
    for e in all_events:
        key = (e.full_text.strip(), e.speaker_name.strip())
        if key == ("", ""):
            continue  # nothing detected
        if key != prev_key:
            deduped.append(e)
            prev_key = key

    return deduped


async def _call_gpt4o_text_batch(
    batch: List[Tuple[float, str]],
) -> List[SpeakerEvent]:
    """Single GPT-4o call to extract text from a batch of frames."""
    content = []
    frame_refs: List[Tuple[int, float]] = []  # (local_idx, timestamp)

    for local_i, (ts, path) in enumerate(batch):
        if not Path(path).exists():
            continue
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "text", "text": f"Frame {local_i} @ {ts:.1f}s"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "auto",  # let GPT-4o choose; "low" misses small overlays
                }
            })
            frame_refs.append((local_i, ts))
        except Exception as e:
            logger.debug(f"Failed to encode {path}: {e}")

    if not frame_refs:
        return []

    try:
        resp = await _get_client().chat.completions.create(
            model=settings.vision_model,
            messages=[
                {"role": "system", "content": _TEXT_EXTRACT_SYSTEM},
                {"role": "user",   "content": content},
            ],
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.1,
        )
        result = json.loads(resp.choices[0].message.content)
        frames_data = result.get("frames", [])
        by_idx = {f["frame_index"]: f for f in frames_data if isinstance(f, dict)}
    except Exception as e:
        logger.error(f"GPT-4o text extraction batch failed: {e}")
        return []

    events: List[SpeakerEvent] = []
    for local_i, ts in frame_refs:
        data = by_idx.get(local_i, {})
        full_text    = (data.get("full_text", "") or "").strip()
        speaker_name = (data.get("speaker_name", "") or "").strip()
        speaker_role = (data.get("speaker_role", "") or "").strip()

        if not full_text and not speaker_name:
            continue   # GPT-4o found nothing — skip

        if speaker_name:
            logger.info(f"Speaker @ {ts:.1f}s: {speaker_name!r} / {speaker_role!r}")

        events.append(SpeakerEvent(
            timestamp=ts,
            speaker_name=speaker_name,
            speaker_role=speaker_role,
            raw_text=full_text,   # compat field
            full_text=full_text,
        ))

    return events


# ── Legacy compat (chunking uses _is_name_plate_candidate indirectly via display()) ─

def get_shared_ocr_reader():
    """Stub — EasyOCR removed. Returns None always."""
    return None
