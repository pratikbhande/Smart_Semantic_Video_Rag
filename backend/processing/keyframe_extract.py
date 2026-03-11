"""
Smart adaptive keyframe extraction.

Strategy:
1. Extract dense candidate set: scene cuts + overlay frames + coverage + boundaries
2. Score each frame with fast PIL analysis:
   - text_score: edge density in lower-third + FULL CENTER (slides, code, screen share)
   - motion_score: pixel diff vs previous frame
   - is_scene_change: >20% overall pixel diff  (hard cut)
   - is_slide_change: center content changes 6-25%  (slide/screen transition)
   - is_person_change: center changes while background is stable
3. Adaptive cap based on content type
4. Priority selection: text > scene_change/slide_change > person_change > static

Scene detection uses ffmpeg showinfo for reliable timestamps (not -frame_pts hack).
Dual-threshold: tries 0.3 first, retries at 0.15 if <3 cuts found (catches slide decks).
"""
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from backend.config import settings
from backend.models.schemas import TopicSegment

logger = logging.getLogger(__name__)


async def extract_keyframes(
    video_path: str,
    video_id: str,
    topics: List[TopicSegment],
    duration: float,
) -> List[Tuple[float, str]]:
    out_dir = Path(settings.keyframes_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    use_interval = duration > settings.scene_detect_max_duration

    # ── Phase 1: Extract candidate frames ────────────────────────────────

    # 1a. First frame always
    first_frames = await _extract_frames_at_times(video_path, out_dir, [0.5], prefix="first")

    # 1b. Scene detection or interval
    if use_interval:
        scene_frames = await _extract_interval_frames(video_path, out_dir, duration)
        overlay_frames = []
    else:
        scene_frames = await _extract_scene_frames(video_path, out_dir, duration)
        logger.info(f"Scene detection: {len(scene_frames)} frames")
        overlay_ts = [ts + 1.5 for ts, _ in scene_frames if ts + 1.5 < duration]
        overlay_frames = await _extract_frames_at_times(
            video_path, out_dir, overlay_ts, prefix="overlay"
        )
        logger.info(f"Overlay frames: {len(overlay_frames)} (+1.5s post-cut)")

    # 1c. Topic boundary frames
    boundary_ts = _get_boundary_timestamps(topics, duration)
    boundary_frames = await _extract_frames_at_times(
        video_path, out_dir, boundary_ts, prefix="boundary"
    )

    # 1d. Uniform coverage frames — ensures no long gaps regardless of scene detection
    # Short videos: dense coverage every 5s; medium: proportional; long: sparse
    if duration <= 120:
        coverage_interval = 5.0
    elif duration <= 600:
        coverage_interval = max(5.0, duration / 30)
    else:
        coverage_interval = max(10.0, min(30.0, duration / 25))
    coverage_ts = [
        round(i * coverage_interval, 1)
        for i in range(1, int(duration / coverage_interval) + 1)
        if i * coverage_interval < duration
    ]
    coverage_frames = await _extract_frames_at_times(
        video_path, out_dir, coverage_ts, prefix="cover"
    )
    logger.info(f"Coverage frames: {len(coverage_frames)} (every {coverage_interval:.0f}s)")

    # ── Phase 2: Combine, dedup, score ───────────────────────────────────
    all_raw = first_frames + scene_frames + overlay_frames + boundary_frames + coverage_frames
    all_raw = [(ts, p) for ts, p in all_raw if p and Path(p).exists()]
    all_raw.sort(key=lambda x: x[0])
    all_raw = _deduplicate(all_raw, min_interval=0.5)  # coarse dedup before scoring

    logger.info(f"Scoring {len(all_raw)} candidate frames")
    scored = _score_frames(all_raw)

    # ── Phase 3: Adaptive cap and priority selection ──────────────────────
    selected = _adaptive_select(scored, use_interval, duration)

    # Final dedup at configured min interval
    selected = _deduplicate(selected, min_interval=settings.min_keyframe_interval)
    logger.info(f"Final keyframe count: {len(selected)}")
    return selected


# ── Frame scoring ─────────────────────────────────────────────────────────────

def _score_frames(frames: List[Tuple[float, str]]) -> List[Tuple[float, str, Dict]]:
    """Score each frame with fast PIL analysis."""
    result = []
    prev_path: Optional[str] = None
    for ts, path in frames:
        scores = _analyze_frame(path, prev_path)
        result.append((ts, path, scores))
        prev_path = path
    return result


def _analyze_frame(path: str, prev_path: Optional[str] = None) -> Dict:
    """
    Fast (~5ms) PIL analysis returning:
      has_text, text_score, motion_score, is_scene_change, is_slide_change, is_person_change

    Text detection covers 3 regions:
      - lower-third (speaker name plates, subtitles)
      - top strip (title cards)
      - full center (slides, code screens, screen shares) ← was missing before
    """
    try:
        from PIL import Image, ImageFilter
        import numpy as np

        img = Image.open(path).convert("RGB").resize((160, 90))
        gray = img.convert("L")

        # ── Text detection — 3 regions ──────────────────────────────────
        lower  = gray.crop((0, 50, 160, 90))    # bottom 44%  — name plates / subtitles
        upper  = gray.crop((0, 0,  160, 18))    # top 20%     — title cards
        center = gray.crop((5, 10, 155, 80))    # center 85%  — slides / code / screen shares
        lower_std  = float(np.array(lower.filter(ImageFilter.FIND_EDGES),  dtype=float).std())
        upper_std  = float(np.array(upper.filter(ImageFilter.FIND_EDGES),  dtype=float).std())
        center_std = float(np.array(center.filter(ImageFilter.FIND_EDGES), dtype=float).std())
        # Center region is larger so normalize its contribution slightly
        text_score = max(lower_std, upper_std, center_std * 0.85) / 100.0
        has_text = text_score > 0.28  # slightly lower than before — center is noisier

        # ── Motion / change detection ────────────────────────────────────
        motion_score = 0.5
        is_scene_change  = False
        is_slide_change  = False   # NEW: moderate content change (slide/screen transition)
        is_person_change = False

        if prev_path and Path(prev_path).exists():
            prev_img = Image.open(prev_path).convert("RGB").resize((160, 90))
            arr  = np.array(img,      dtype=float)
            parr = np.array(prev_img, dtype=float)

            overall_diff = float(np.mean(np.abs(arr - parr))) / 255.0
            motion_score = min(1.0, overall_diff * 6)

            # Center content area (rows 8-82, cols 8-152) — where slides live
            content_diff = float(
                np.mean(np.abs(arr[8:82, 8:152] - parr[8:82, 8:152]))
            ) / 255.0

            # Face/upper-body area (rows 5-60, cols 40-120)
            center_diff = float(
                np.mean(np.abs(arr[5:60, 40:120] - parr[5:60, 40:120]))
            ) / 255.0

            # Hard scene cut: >20% overall change
            is_scene_change = overall_diff > 0.20
            # Slide/content transition: content area changes 6-25% without a hard cut
            # Catches: slide deck transitions, screen-share switches, code editor changes
            is_slide_change = (
                not is_scene_change and content_diff > 0.06
            )
            # Person change: face moves while background is stable
            is_person_change = (
                not is_scene_change and not is_slide_change
                and center_diff > 0.12 and 0.04 < overall_diff < 0.20
            )

        return {
            "has_text":        has_text,
            "text_score":      text_score,
            "motion_score":    motion_score,
            "is_scene_change": is_scene_change,
            "is_slide_change": is_slide_change,
            "is_person_change":is_person_change,
        }
    except Exception:
        return {
            "has_text": False, "text_score": 0.0, "motion_score": 0.5,
            "is_scene_change": False, "is_slide_change": False, "is_person_change": False,
        }


def _adaptive_select(
    scored: List[Tuple[float, str, Dict]],
    use_interval: bool,
    duration: float,
) -> List[Tuple[float, str]]:
    """Priority-based selection with content-adaptive cap."""
    if not scored:
        return []

    text_count   = sum(1 for _, _, s in scored if s["has_text"])
    scene_count  = sum(1 for _, _, s in scored if s["is_scene_change"])
    slide_count  = sum(1 for _, _, s in scored if s["is_slide_change"])
    person_count = sum(1 for _, _, s in scored if s["is_person_change"])
    total = len(scored)
    text_ratio = text_count / max(1, total)
    change_count = scene_count + slide_count  # hard cuts + slide transitions

    # ── Duration-aware cap ──────────────────────────────────────────────────
    # Base cap on video length first (1 keyframe per ~8s max), then content type
    # This prevents 50+ keyframes for a 3-minute video with subtitles
    duration_base = max(settings.min_keyframes_floor, int(duration / 8))

    if text_ratio > 0.40:
        # Text-heavy (demos, slides, subtitled content):
        # 1 frame per 8s up to max_keyframes_cap — but NOT unlimited
        cap = min(settings.max_keyframes_cap, duration_base)
    elif text_ratio > 0.15 or slide_count > 3:
        # Mixed: talking head with slides
        cap = min(max(15, settings.max_keyframes_cap * 2 // 3), duration_base)
    elif use_interval:
        # Long video: 1 keyframe per 2 min minimum, plus all change frames
        duration_based = max(20, int(duration / 120))
        cap = min(150, max(change_count + person_count + 8, duration_based))
    else:
        # Short, mostly static/visual (e.g. nature, no-text interview)
        cap = max(settings.min_keyframes_floor, min(20, change_count + person_count + 5))

    logger.info(
        f"Content: {text_count} text / {scene_count} scene / {slide_count} slide / "
        f"{person_count} person of {total} candidates → cap={cap} (text_ratio={text_ratio:.2f})"
    )

    # Priority 1: text frames — always keep (slides, subtitles, name plates, code)
    p1 = [(ts, p) for ts, p, s in scored if s["has_text"]]
    # Priority 2: hard scene cuts AND slide/content transitions — same importance
    p2 = [(ts, p) for ts, p, s in scored
          if (s["is_scene_change"] or s["is_slide_change"]) and not s["has_text"]]
    # Priority 3: person changes (talking head movement)
    p3 = [(ts, p) for ts, p, s in scored
          if s["is_person_change"] and not s["has_text"]
          and not s["is_scene_change"] and not s["is_slide_change"]]
    # Priority 4: static/coverage frames
    p4 = [(ts, p) for ts, p, s in scored
          if not s["has_text"] and not s["is_scene_change"]
          and not s["is_slide_change"] and not s["is_person_change"]]

    # ── Priority selection within cap ───────────────────────────────────────
    # Among text frames (p1), prioritize scene/slide change frames over static coverage.
    # This ensures GPT-4o sees every important visual transition even when capped.
    p1_change   = [(ts, p) for ts, p, s in scored
                   if s["has_text"] and (s["is_scene_change"] or s["is_slide_change"])]
    p1_coverage = [(ts, p) for ts, p, s in scored
                   if s["has_text"] and not s["is_scene_change"] and not s["is_slide_change"]]

    # Step 1: include ALL scene/slide-change text frames (they mark visual transitions)
    if len(p1_change) >= cap:
        # Even-space the important frames when there are too many
        step = len(p1_change) / cap
        selected = [p1_change[int(i * step)] for i in range(cap)]
    else:
        selected = list(p1_change)
        # Step 2: fill remaining slots with evenly-spaced coverage frames
        remaining = cap - len(selected)
        if remaining > 0 and p1_coverage:
            if len(p1_coverage) <= remaining:
                selected.extend(p1_coverage)
            else:
                step = len(p1_coverage) / remaining
                selected.extend([p1_coverage[int(i * step)] for i in range(remaining)])

    for pool in [p2, p3]:
        remaining = cap - len(selected)
        if remaining > 0:
            selected.extend(pool[:remaining])

    # Fill remaining slots with evenly-spaced static frames
    remaining = cap - len(selected)
    if remaining > 0 and p4:
        if len(p4) <= remaining:
            selected.extend(p4)
        else:
            step = len(p4) / remaining
            selected.extend([p4[int(i * step)] for i in range(remaining)])

    selected.sort(key=lambda x: x[0])
    return selected


# ── ffmpeg helpers ────────────────────────────────────────────────────────────

async def _extract_scene_frames(
    video_path: str, out_dir: Path, duration: float
) -> List[Tuple[float, str]]:
    """
    Detect scene cuts using ffmpeg showinfo filter — parses actual pts_time values
    from stderr so timestamps are always correct regardless of video timebase.

    Dual-threshold strategy:
      1. Try configured threshold (default 0.3) — catches hard cuts
      2. If < 3 cuts, retry at 0.15 — catches slide deck transitions (same BG, text changes)
      3. Fall back to interval sampling only if both return 0 cuts
    """
    timestamps: List[float] = []

    for thresh in [settings.scene_threshold, 0.15]:
        timestamps = await _detect_scene_timestamps(video_path, thresh)
        if len(timestamps) >= 2:
            logger.info(f"Scene detection: {len(timestamps)} cuts at threshold={thresh}")
            break

    if not timestamps:
        logger.info("No scene cuts detected — static/talking-head content, using interval sampling")
        return await _extract_interval_frames(video_path, out_dir, duration)

    return await _extract_frames_at_times(video_path, out_dir, timestamps, prefix="scene")


async def _detect_scene_timestamps(video_path: str, threshold: float) -> List[float]:
    """
    Run ffmpeg scene detection and return real timestamps (in seconds) parsed from stderr.
    Uses showinfo filter which outputs: pts_time:X.XXX — always in seconds, always correct.
    """
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "0", "-an", "-f", "null", "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()

    if proc.returncode != 0:
        return []

    timestamps: List[float] = []
    for line in stderr.decode("utf-8", errors="replace").splitlines():
        if "pts_time:" in line:
            m = re.search(r"pts_time:(\d+\.?\d*)", line)
            if m:
                ts = float(m.group(1))
                if ts > 0.1:   # skip the very first frame
                    timestamps.append(ts)

    return sorted(set(round(t, 2) for t in timestamps))


async def _extract_interval_frames(
    video_path: str, out_dir: Path, duration: float
) -> List[Tuple[float, str]]:
    # For long videos: 1 keyframe per 2 minutes (up to 150), not capped at 50
    dynamic_cap = max(
        settings.min_keyframes_floor,
        min(150, int(duration / 120))  # 1 per 2min
    )
    max_frames = max(
        settings.min_keyframes_floor,
        min(dynamic_cap, int(duration / settings.keyframe_interval_secs))
    )
    interval = max(5.0, duration / max_frames)
    timestamps = [
        i * interval for i in range(int(duration / interval) + 1)
        if i * interval < duration
    ]
    logger.info(f"Interval sampling: {len(timestamps)} frames at {interval:.1f}s intervals")
    return await _extract_frames_at_times(video_path, out_dir, timestamps, prefix="interval")


async def _extract_frames_at_times(
    video_path: str,
    out_dir: Path,
    timestamps: List[float],
    prefix: str = "frame",
) -> List[Tuple[float, str]]:
    if not timestamps:
        return []
    sem = asyncio.Semaphore(6)

    async def extract_one(ts: float, idx: int):
        frame_path = str(out_dir / f"{prefix}_{idx:06d}.jpg")
        if os.path.exists(frame_path):
            return (ts, frame_path)
        async with sem:
            cmd = [
                "ffmpeg", "-y", "-ss", f"{ts:.3f}", "-i", video_path,
                "-frames:v", "1", "-q:v", "3", frame_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()
        return (ts, frame_path) if os.path.exists(frame_path) else None

    raw = await asyncio.gather(
        *[extract_one(ts, i) for i, ts in enumerate(timestamps)],
        return_exceptions=True,
    )
    return [r for r in raw if isinstance(r, tuple)]


def _get_boundary_timestamps(topics: List[TopicSegment], duration: float) -> List[float]:
    return [
        topic.start_time + 1.0
        for topic in topics
        if topic.start_time + 1.0 < duration
    ]


def _deduplicate(frames: List[Tuple[float, str]], min_interval: float) -> List[Tuple[float, str]]:
    if not frames:
        return []
    result = [frames[0]]
    for ts, path in frames[1:]:
        if ts - result[-1][0] >= min_interval:
            result.append((ts, path))
    return result
