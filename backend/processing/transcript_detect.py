"""
Detect whether a video already has subtitles — either embedded or as a sidecar file.
Returns the transcript source type and optionally a parsed segment list.
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from backend.models.schemas import TranscriptSegment, TranscriptSource

logger = logging.getLogger(__name__)


async def detect_transcript(
    video_path: str, video_id: str
) -> Tuple[TranscriptSource, List[TranscriptSegment]]:
    """
    Returns (source, segments). If nothing found, returns (TranscriptSource.none, []).
    Priority: embedded subtitles > sidecar SRT/VTT.
    """
    # 1. Check embedded subtitles via ffprobe
    embedded = await _check_embedded(video_path, video_id)
    if embedded:
        return TranscriptSource.embedded, embedded

    # 2. Check sidecar files
    sidecar_path = _find_sidecar(video_path)
    if sidecar_path:
        segments = _parse_sidecar(sidecar_path, video_id)
        if segments:
            logger.info(f"Found sidecar transcript: {sidecar_path}")
            return TranscriptSource.sidecar, segments

    return TranscriptSource.none, []


async def _check_embedded(video_path: str, video_id: str) -> List[TranscriptSegment]:
    """Use ffprobe to detect subtitle streams, then ffmpeg to extract first one."""
    try:
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "s", video_path
        ]
        proc = await asyncio.create_subprocess_exec(
            *probe_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return []

        info = json.loads(stdout.decode())
        streams = info.get("streams", [])
        if not streams:
            return []

        logger.info(f"Found {len(streams)} subtitle stream(s) in {video_path}")
        # Extract first subtitle stream to SRT
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as f:
            srt_path = f.name

        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", "0:s:0", "-c:s", "srt", srt_path
        ]
        proc2 = await asyncio.create_subprocess_exec(
            *extract_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc2.communicate()

        if Path(srt_path).exists() and Path(srt_path).stat().st_size > 0:
            segs = _parse_srt(Path(srt_path).read_text(encoding="utf-8", errors="ignore"), video_id)
            os.unlink(srt_path)
            return segs

        os.unlink(srt_path)
    except Exception as e:
        logger.warning(f"Embedded subtitle check failed: {e}")
    return []


def _find_sidecar(video_path: str) -> Optional[Path]:
    base = Path(video_path).with_suffix("")
    for ext in [".srt", ".vtt", ".SRT", ".VTT"]:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _parse_sidecar(path: Path, video_id: str) -> List[TranscriptSegment]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".vtt":
        return _parse_vtt(text, video_id)
    return _parse_srt(text, video_id)


def _parse_srt(text: str, video_id: str) -> List[TranscriptSegment]:
    segments = []
    # SRT blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        # Find timing line
        timing_line = None
        text_lines = []
        for i, line in enumerate(lines):
            m = re.match(
                r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
                line
            )
            if m:
                timing_line = m
                text_lines = lines[i + 1:]
                break
        if not timing_line:
            continue
        start = _srt_time(timing_line.group(1))
        end = _srt_time(timing_line.group(2))
        content = " ".join(text_lines).strip()
        # Strip HTML tags
        content = re.sub(r"<[^>]+>", "", content).strip()
        if content:
            segments.append(TranscriptSegment(
                video_id=video_id, start_time=start, end_time=end,
                text=content, confidence=1.0
            ))
    return segments


def _parse_vtt(text: str, video_id: str) -> List[TranscriptSegment]:
    # Remove WEBVTT header and NOTE blocks
    text = re.sub(r"WEBVTT.*?\n", "", text, count=1)
    text = re.sub(r"NOTE[^\n]*\n.*?\n\n", "", text, flags=re.DOTALL)
    return _parse_srt(text.replace("-->", "-->"), video_id)


def _srt_time(s: str) -> float:
    s = s.replace(",", ".")
    parts = s.split(":")
    h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + sec
