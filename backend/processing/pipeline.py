"""
Master pipeline orchestrator. Runs all processing stages sequentially,
broadcasts progress via WebSocket callbacks, handles errors gracefully.
"""
import asyncio
import logging
from pathlib import Path
from typing import Callable, Awaitable, Optional

from backend.config import settings
from backend.models.schemas import (
    VideoStatus, TranscriptSource, KeyframeData
)
from backend.storage import db
from backend.processing.transcript_detect import detect_transcript
from backend.processing.audio_extract import extract_audio, get_video_duration, check_audio_stream
from backend.processing.transcription import transcribe_audio
from backend.processing.transcript_correct import correct_transcript
from backend.processing.topic_segment import segment_topics
from backend.processing.keyframe_extract import extract_keyframes
from backend.processing.visual_analysis import analyze_keyframes
from backend.processing.chunking import build_chunks
from backend.processing.lower_third_scan import scan_lower_thirds
from backend.indexing.embedder import embed_chunks
from backend.indexing.vector_store import index_chunks, clear_video_index

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], Awaitable[None]]


async def run_pipeline(
    video_id: str,
    video_path: str,
    glossary_names: Optional[str] = None,
    glossary_terms: Optional[str] = None,
    domain_context: Optional[str] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> None:
    """
    Full processing pipeline. Updates DB and calls progress_cb at each stage.
    progress_cb(stage, progress_0_100, current_step)
    """

    async def progress(stage: str, pct: float, step: str):
        await db.log_processing(video_id, stage, "running", step)
        if progress_cb:
            await progress_cb(stage, pct, step)

    async def fail(stage: str, error: str):
        logger.error(f"Pipeline [{video_id}] failed at {stage}: {error}")
        await db.update_video_status(video_id, VideoStatus.failed)
        await db.log_processing(video_id, stage, "error", error)
        if progress_cb:
            await progress_cb(stage, -1, f"ERROR: {error}")

    try:
        await db.update_video_status(video_id, VideoStatus.processing)

        # ── Stage 1: Get duration ──────────────────────────────────────────
        await progress("duration", 2, "Reading video metadata")
        duration = await get_video_duration(video_path)
        await db.update_video_status(video_id, VideoStatus.processing, duration=duration)
        logger.info(f"Video duration: {duration:.1f}s")

        # ── Stage 1b: Launch lower-third scan concurrently (runs in background) ──
        # This scans for speaker name plates independently of scene detection.
        # Runs concurrently with transcription stages — adds ~0 wall-clock time.
        lower_third_task = asyncio.create_task(
            scan_lower_thirds(video_path, duration)
        )

        # ── Stage 2: Detect existing transcript ───────────────────────────
        await progress("transcript_detect", 5, "Checking for existing subtitles")
        transcript_source, segments = await detect_transcript(video_path, video_id)

        glossary_prompt = None
        if glossary_names or glossary_terms:
            parts = []
            if glossary_names:
                parts.append(glossary_names)
            if glossary_terms:
                parts.append(glossary_terms)
            glossary_prompt = ", ".join(parts)

        if transcript_source == TranscriptSource.none:
            # ── Check audio stream before attempting transcription ─────────
            await progress("audio_detect", 8, "Checking audio stream")
            has_audio = await check_audio_stream(video_path)

            if not has_audio:
                logger.info(f"No audio stream found in {video_path}")
                transcript_source = TranscriptSource.no_audio
                segments = []
                await progress("audio_detect", 35, "No audio stream detected — skipping transcription")
            else:
                # ── Stage 3: Audio extraction ──────────────────────────────────
                await progress("audio_extract", 10, "Extracting audio track")
                wav_path = await extract_audio(video_path, video_id)

                # ── Stage 4: Transcription ─────────────────────────────────────
                await progress("transcription", 20, "Transcribing audio (Whisper API)")
                segments, had_errors = await transcribe_audio(wav_path, video_id, duration, glossary_prompt)

                if not segments and had_errors:
                    # API errors (rate limit, network) — don't misclassify as no speech
                    logger.warning(f"Transcription failed due to API errors for {video_id} — treating as empty transcript")
                    transcript_source = TranscriptSource.whisper
                    segments = []
                    await progress("transcription", 35, "Transcription encountered API errors — partial result")
                elif not segments:
                    # Whisper returned nothing with no errors — genuinely no speech
                    logger.info(f"Whisper produced no segments for {video_id} — likely no speech")
                    transcript_source = TranscriptSource.no_speech
                    segments = []
                    await progress("transcription", 35, "No speech detected in audio")
                else:
                    transcript_source = TranscriptSource.whisper

                    # ── Stage 5: Transcript correction ────────────────────────────
                    has_glossary = bool(glossary_names or glossary_terms or domain_context)
                    skip_correction = duration > settings.skip_correction_min_duration and not has_glossary
                    if segments and not skip_correction:
                        await progress("correction", 35, "Correcting transcript")
                        segments = await correct_transcript(
                            segments, glossary_names, glossary_terms, domain_context
                        )
                    elif skip_correction:
                        logger.info(f"Skipping correction for {duration:.0f}s video (no glossary provided)")
                        await progress("correction", 35, "Transcript ready (correction skipped for speed)")
        else:
            logger.info(f"Using existing transcript ({transcript_source.value})")
            await progress("transcript_detect", 35, f"Using {transcript_source.value}")

        # Persist transcript
        if segments:
            await db.delete_transcripts(video_id)
            await db.insert_transcripts(segments)

        await db.update_video_status(
            video_id, VideoStatus.processing,
            transcript_source=transcript_source
        )

        # ── Collect lower-third scan results (should be done by now) ──────
        try:
            speaker_events = await lower_third_task
            logger.info(f"Lower-third scan: {len(speaker_events)} speaker events")
        except Exception as e:
            logger.warning(f"Lower-third scan failed (non-fatal): {e}")
            speaker_events = []

        # ── Stage 6: Topic segmentation ────────────────────────────────────
        await progress("topic_segment", 40, "Segmenting into topics")
        topics = await segment_topics(segments, video_id, duration, domain_context or "")
        await db.delete_topics(video_id)
        await db.insert_topics(topics)
        logger.info(f"Topics: {[t.title for t in topics]}")

        # ── Stage 7: Keyframe extraction ───────────────────────────────────
        await progress("keyframe_extract", 50, "Extracting keyframes")
        frame_tuples = await extract_keyframes(video_path, video_id, topics, duration)

        keyframes = [
            KeyframeData(
                video_id=video_id,
                timestamp=ts,
                frame_path=path,
            )
            for ts, path in frame_tuples
        ]

        # ── Stage 8: Visual analysis ───────────────────────────────────────
        await progress("visual_analysis", 60, f"Analyzing {len(keyframes)} keyframes")
        transcript_ctx = " ".join(s.text for s in segments[:20])
        # Always run OCR — captures speaker name plates (lower-thirds), slide text,
        # burned-in subtitles, and on-screen labels regardless of video length.
        keyframes = await analyze_keyframes(
            keyframes, transcript_ctx, domain_context or ""
        )
        await db.delete_keyframes(video_id)
        await db.insert_keyframes(keyframes)

        # ── Stage 9: Build chunks ──────────────────────────────────────────
        await progress("chunking", 75, "Building hierarchical chunks")
        chunks = build_chunks(video_id, segments, topics, keyframes, speaker_events)

        # ── Stage 10: Clear old index ──────────────────────────────────────
        await progress("indexing", 80, "Clearing old index")
        await clear_video_index(video_id)

        # ── Stage 11: Embed + index ────────────────────────────────────────
        await progress("indexing", 85, f"Embedding {len(chunks)} chunks")
        embedded_chunks = await embed_chunks(chunks)

        await progress("indexing", 92, "Storing in vector database")
        await index_chunks(embedded_chunks, video_id)

        # ── Done ───────────────────────────────────────────────────────────
        await db.update_video_status(video_id, VideoStatus.ready)
        await db.log_processing(video_id, "complete", "success", "Pipeline complete")
        await progress("complete", 100, "Processing complete")
        logger.info(f"Pipeline complete for {video_id}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Pipeline error for {video_id}:\n{tb}")
        await fail("pipeline", str(e)[:500])
