"""
Extract audio from video using ffmpeg, then split into overlapping chunks for transcription.
"""
import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

from backend.config import settings

logger = logging.getLogger(__name__)


async def extract_audio(video_path: str, video_id: str) -> str:
    """Extract full audio as WAV mono 16kHz. Returns path to WAV file."""
    out_dir = Path(settings.audio_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "full_audio.wav"

    if wav_path.exists():
        logger.info(f"Audio already extracted: {wav_path}")
        return str(wav_path)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path)
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed: {stderr.decode()[-500:]}")

    logger.info(f"Audio extracted to {wav_path}")
    return str(wav_path)


async def check_audio_stream(video_path: str) -> bool:
    """
    Returns True if the video has an extractable audio track.

    Uses two methods to avoid false negatives on Chrome/browser WebM recordings
    where ffprobe's -select_streams filter can miss Opus audio streams:

    1. ffprobe — scan ALL streams, look for codec_type == "audio"
    2. ffmpeg short extraction — if ffprobe misses it, try actually extracting
       3 seconds of audio; success with a non-trivial output file = audio exists
    """
    # Method 1: ffprobe — inspect all streams, match on codec_type
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", video_path  # no -select_streams filter — check everything
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        try:
            streams = json.loads(stdout.decode()).get("streams", [])
            if any(s.get("codec_type") == "audio" for s in streams):
                return True
        except Exception:
            pass

    # Method 2: attempt to extract a 3-second audio sample via ffmpeg.
    # Catches edge cases where ffprobe mislabels the stream (common in WebM).
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        probe_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-t", "3", "-vn",
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp.name
        ]
        proc2 = await asyncio.create_subprocess_exec(
            *probe_cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc2.communicate()
        if proc2.returncode == 0 and os.path.getsize(tmp.name) > 1000:
            logger.info(f"Audio detected via extraction fallback for {Path(video_path).name}")
            return True
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    return False


async def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0


def split_audio_chunks(
    wav_path: str, duration: float,
    chunk_duration: int = None, overlap: int = None
) -> List[Tuple[float, float, str]]:
    """
    Split WAV into overlapping time ranges. Returns list of (start, end, wav_path).
    The wav_path is the same file — we'll use ffmpeg -ss/-t at transcription time.
    """
    chunk_dur = chunk_duration or settings.audio_chunk_duration
    ovlp = overlap or settings.audio_chunk_overlap

    if duration <= chunk_dur:
        return [(0.0, duration, wav_path)]

    chunks = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_dur, duration)
        chunks.append((start, end, wav_path))
        if end >= duration:
            break
        start = end - ovlp  # overlap to avoid cutting words
    return chunks


async def extract_audio_chunk(wav_path: str, start: float, end: float, out_path: str) -> str:
    """Extract a time slice from a WAV file."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", wav_path,
        out_path
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Chunk extraction failed: {stderr.decode()[-300:]}")
    return out_path
