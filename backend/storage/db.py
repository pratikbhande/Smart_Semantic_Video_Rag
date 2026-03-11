import aiosqlite
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.config import settings
from backend.models.schemas import (
    VideoMetadata, VideoStatus, TranscriptSegment, TopicSegment,
    KeyframeData, TranscriptSource
)

logger = logging.getLogger(__name__)

DB_PATH = Path(settings.data_dir) / "video_rag.db"

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    original_path TEXT,
    processed_dir TEXT,
    duration REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    transcript_source TEXT DEFAULT 'None',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    key_entities_json TEXT DEFAULT '[]',
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS keyframes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    frame_path TEXT NOT NULL,
    scene_type TEXT DEFAULT 'unknown',
    visual_analysis_json TEXT DEFAULT '{}',
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);
"""


async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=30000")
        await db.executescript(CREATE_TABLES)
        # Migration: add video_id to processing_log if it was created without it
        try:
            await db.execute("ALTER TABLE processing_log ADD COLUMN video_id TEXT")
            logger.info("Migrated processing_log: added video_id column")
        except Exception:
            pass  # column already exists — normal path
        await db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


async def get_db():
    return await aiosqlite.connect(DB_PATH)


# ── Videos ────────────────────────────────────────────────────────────────────

async def insert_video(video_id: str, filename: str, original_path: str) -> VideoMetadata:
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO videos (id, filename, original_path, status, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
            (video_id, filename, original_path, VideoStatus.pending, now, "{}")
        )
        await db.commit()
    return VideoMetadata(
        id=video_id,
        filename=filename,
        status=VideoStatus.pending,
        created_at=datetime.fromisoformat(now)
    )


async def update_video_status(video_id: str, status: VideoStatus, **kwargs):
    now = datetime.utcnow().isoformat()
    fields = ["status = ?", "updated_at = ?"]
    values = [status.value, now]
    for k, v in kwargs.items():
        if k == "metadata":
            fields.append("metadata_json = ?")
            values.append(json.dumps(v))
        elif k == "transcript_source":
            fields.append("transcript_source = ?")
            values.append(v.value if hasattr(v, "value") else v)
        else:
            fields.append(f"{k} = ?")
            values.append(v)
    values.append(video_id)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(f"UPDATE videos SET {', '.join(fields)} WHERE id = ?", values)
        await db.commit()


async def get_video(video_id: str) -> Optional[VideoMetadata]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM videos WHERE id = ?", (video_id,)) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return _row_to_video(dict(row))


async def list_videos() -> List[VideoMetadata]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM videos ORDER BY created_at DESC") as cur:
            rows = await cur.fetchall()
    return [_row_to_video(dict(r)) for r in rows]


def _row_to_video(row: dict) -> VideoMetadata:
    meta = json.loads(row.get("metadata_json") or "{}")
    src_raw = row.get("transcript_source", "None")
    try:
        src = TranscriptSource(src_raw)
    except ValueError:
        src = TranscriptSource.none
    return VideoMetadata(
        id=row["id"],
        filename=row["filename"],
        duration=row.get("duration"),
        status=VideoStatus(row["status"]),
        transcript_source=src,
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]) if row.get("updated_at") else None,
        metadata=meta,
    )


# ── Transcripts ───────────────────────────────────────────────────────────────

async def insert_transcripts(segments: List[TranscriptSegment]):
    if not segments:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO transcripts (video_id, start_time, end_time, text, confidence) VALUES (?, ?, ?, ?, ?)",
            [(s.video_id, s.start_time, s.end_time, s.text, s.confidence) for s in segments]
        )
        await db.commit()


async def get_transcripts(video_id: str) -> List[TranscriptSegment]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM transcripts WHERE video_id = ? ORDER BY start_time",
            (video_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [TranscriptSegment(
        id=r["id"], video_id=r["video_id"],
        start_time=r["start_time"], end_time=r["end_time"],
        text=r["text"], confidence=r["confidence"]
    ) for r in rows]


async def delete_transcripts(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM transcripts WHERE video_id = ?", (video_id,))
        await db.commit()


async def delete_video(video_id: str):
    """Delete a video and all its associated records from the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA busy_timeout=30000")
        # Child tables reference the video via 'video_id' foreign key
        for table in ("processing_log", "keyframes", "topics", "transcripts"):
            try:
                await db.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))
            except Exception as e:
                logger.warning(f"delete_video: skipping {table}: {e}")
        # The videos table uses 'id' as its primary key, not 'video_id'
        await db.execute("DELETE FROM videos WHERE id = ?", (video_id,))
        await db.commit()


# ── Topics ────────────────────────────────────────────────────────────────────

async def insert_topics(topics: List[TopicSegment]):
    if not topics:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO topics (video_id, start_time, end_time, title, summary, key_entities_json) VALUES (?, ?, ?, ?, ?, ?)",
            [(t.video_id, t.start_time, t.end_time, t.title, t.summary, json.dumps(t.key_entities)) for t in topics]
        )
        await db.commit()


async def delete_topics(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM topics WHERE video_id = ?", (video_id,))
        await db.commit()


async def get_topics(video_id: str) -> List[TopicSegment]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM topics WHERE video_id = ? ORDER BY start_time",
            (video_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [TopicSegment(
        id=r["id"], video_id=r["video_id"],
        start_time=r["start_time"], end_time=r["end_time"],
        title=r["title"], summary=r["summary"],
        key_entities=json.loads(r["key_entities_json"] or "[]")
    ) for r in rows]


# ── Keyframes ─────────────────────────────────────────────────────────────────

async def insert_keyframes(frames: List[KeyframeData]):
    if not frames:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT INTO keyframes (video_id, timestamp, frame_path, scene_type, visual_analysis_json) VALUES (?, ?, ?, ?, ?)",
            [(f.video_id, f.timestamp, f.frame_path, f.scene_type, json.dumps(f.visual_analysis)) for f in frames]
        )
        await db.commit()


async def delete_keyframes(video_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM keyframes WHERE video_id = ?", (video_id,))
        await db.commit()


async def get_keyframes(video_id: str) -> List[KeyframeData]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM keyframes WHERE video_id = ? ORDER BY timestamp",
            (video_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [KeyframeData(
        id=r["id"], video_id=r["video_id"],
        timestamp=r["timestamp"], frame_path=r["frame_path"],
        scene_type=r["scene_type"],
        visual_analysis=json.loads(r["visual_analysis_json"] or "{}")
    ) for r in rows]


# ── Processing Log ────────────────────────────────────────────────────────────

async def log_processing(video_id: str, stage: str, status: str, message: str = ""):
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO processing_log (video_id, stage, status, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (video_id, stage, status, message, now)
        )
        await db.commit()
