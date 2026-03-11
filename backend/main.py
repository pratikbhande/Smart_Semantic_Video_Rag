"""
FastAPI V2 backend — Video RAG.
Endpoints: upload, list, detail, stream, transcript, topics, query, WebSocket progress.
"""
import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import aiofiles
from fastapi import (
    FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
    WebSocketDisconnect, BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models.schemas import QueryRequest, QueryResponse, VideoStatus, VideoDiscoveryRequest
from backend.storage import db
from backend.processing.pipeline import run_pipeline
from backend.query.retriever import retrieve, retrieve_multi, discover_videos
from backend.query.reranker import rerank
from backend.query.answer_gen import generate_answer
from backend.indexing.vector_store import clear_video_index

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="Video RAG V2", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket connection manager ──────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self._connections: Dict[str, list] = {}

    async def connect(self, video_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(video_id, []).append(ws)

    def disconnect(self, video_id: str, ws: WebSocket):
        conns = self._connections.get(video_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, video_id: str, message: dict):
        conns = self._connections.get(video_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(video_id, ws)


manager = ConnectionManager()


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    settings.ensure_dirs()
    await db.init_db()
    logger.info("Video RAG V2 started")


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    glossary_names: Optional[str] = Form(None),
    glossary_terms: Optional[str] = Form(None),
    domain_context: Optional[str] = Form(None),
):
    filename = file.filename or "video.mp4"
    video_id = hashlib.md5(filename.encode()).hexdigest()[:16]

    # Stream upload to disk in 1 MB chunks — no size limit, works for any file size
    upload_path = Path(settings.upload_dir) / video_id / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(upload_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            await f.write(chunk)

    # Create DB record
    video = await db.insert_video(video_id, filename, str(upload_path))

    # Start pipeline in background
    async def progress_callback(stage: str, progress: float, step: str):
        await manager.broadcast(video_id, {
            "type": "progress",
            "data": {
                "video_id": video_id,
                "stage": stage,
                "progress": progress,
                "current_step": step,
            }
        })

    background_tasks.add_task(
        run_pipeline,
        video_id=video_id,
        video_path=str(upload_path),
        glossary_names=glossary_names,
        glossary_terms=glossary_terms,
        domain_context=domain_context,
        progress_cb=progress_callback,
    )

    return {
        "video_id": video_id,
        "filename": filename,
        "status": "processing",
        "message": "Upload received, processing started",
    }


# ── Video list & detail ───────────────────────────────────────────────────────

@app.get("/api/videos")
async def list_videos():
    videos = await db.list_videos()
    return [v.model_dump(mode="json") for v in videos]


@app.get("/api/videos/{video_id}")
async def get_video(video_id: str):
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video.model_dump(mode="json")


# ── Delete video ──────────────────────────────────────────────────────────────

@app.delete("/api/videos/{video_id}")
async def delete_video_endpoint(video_id: str):
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Clear ChromaDB index (non-fatal)
    try:
        await clear_video_index(video_id)
    except Exception as e:
        logger.warning(f"ChromaDB cleanup failed for {video_id}: {e}")

    # Remove files from disk — run in executor so it doesn't block event loop
    loop = asyncio.get_event_loop()
    for dir_path in [
        Path(settings.upload_dir) / video_id,
        Path(settings.keyframes_dir) / video_id,
        Path(settings.audio_dir) / video_id,
        Path(settings.chunks_dir) / video_id,
    ]:
        if dir_path.exists():
            try:
                await loop.run_in_executor(None, shutil.rmtree, str(dir_path))
            except Exception as e:
                logger.warning(f"File cleanup failed for {dir_path}: {e}")

    # Remove all DB records for this video
    try:
        await db.delete_video(video_id)
    except Exception as e:
        logger.error(f"DB delete failed for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error during delete: {str(e)}")

    logger.info(f"Deleted video {video_id}")
    return {"status": "deleted", "video_id": video_id}


# ── Video streaming with Range header support ─────────────────────────────────

MIME_MAP = {
    ".mp4": "video/mp4", ".m4v": "video/mp4",
    ".mkv": "video/x-matroska", ".webm": "video/webm",
    ".avi": "video/x-msvideo", ".mov": "video/quicktime",
}

@app.get("/api/videos/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    upload_dir = Path(settings.upload_dir) / video_id
    video_files = list(upload_dir.glob("*")) if upload_dir.exists() else []
    video_files = [f for f in video_files if f.suffix.lower() in MIME_MAP]

    if not video_files:
        raise HTTPException(status_code=404, detail="Video file not found")

    file_path = video_files[0]
    file_size = file_path.stat().st_size
    mime_type = MIME_MAP.get(file_path.suffix.lower(), "video/mp4")

    range_header = request.headers.get("range")

    if not range_header:
        # No Range request — stream full file with correct Content-Length.
        # Browser will send Range requests once it sees Accept-Ranges.
        async def full_stream():
            async with aiofiles.open(file_path, "rb") as f:
                while True:
                    data = await f.read(1024 * 1024)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            full_stream(),
            status_code=200,
            media_type=mime_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache",
            },
        )

    # Parse Range header
    # Per RFC 7233: bytes=start-end  OR  bytes=-suffix_length
    CHUNK = 4 * 1024 * 1024  # 4 MB per range response
    range_val = range_header.replace("bytes=", "")
    parts = range_val.split("-")

    if not parts[0]:
        # Suffix range: bytes=-N → last N bytes
        suffix = int(parts[1])
        start = max(0, file_size - suffix)
        end = file_size - 1
    else:
        start = int(parts[0])
        end = min(int(parts[1]), file_size - 1) if parts[1] else min(start + CHUNK - 1, file_size - 1)

    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    async def range_stream():
        async with aiofiles.open(file_path, "rb") as f:
            await f.seek(start)
            remaining = chunk_size
            while remaining > 0:
                data = await f.read(min(1024 * 1024, remaining))
                if not data:
                    break
                yield data
                remaining -= len(data)

    return StreamingResponse(
        range_stream(),
        status_code=206,
        media_type=mime_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Cache-Control": "no-cache",
        },
    )


# ── Transcript & Topics ───────────────────────────────────────────────────────

@app.get("/api/videos/{video_id}/transcript")
async def get_transcript(video_id: str):
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    segments = await db.get_transcripts(video_id)
    return {
        "video_id": video_id,
        "source": video.transcript_source.value,
        "segments": [s.model_dump() for s in segments],
    }


@app.get("/api/videos/{video_id}/topics")
async def get_topics(video_id: str):
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    topics = await db.get_topics(video_id)
    return {
        "video_id": video_id,
        "topics": [t.model_dump() for t in topics],
    }


@app.get("/api/videos/{video_id}/keyframes")
async def get_keyframes(video_id: str):
    frames = await db.get_keyframes(video_id)
    return {"video_id": video_id, "keyframes": [f.model_dump() for f in frames]}


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query_video(req: QueryRequest):
    # Resolve which video(s) to query
    video_ids = req.video_ids or ([req.video_id] if req.video_id else None)
    if not video_ids:
        raise HTTPException(status_code=422, detail="video_id or video_ids is required")

    # Validate all videos exist and are ready
    videos = []
    for vid in video_ids:
        video = await db.get_video(vid)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video not found: {vid}")
        if video.status != VideoStatus.ready:
            raise HTTPException(
                status_code=409,
                detail=f"Video '{video.filename}' is not ready (status: {video.status.value})"
            )
        videos.append(video)

    video_map = {v.id: v.filename for v in videos}

    is_multi = len(video_ids) > 1

    # Retrieve
    if not is_multi:
        candidates = await retrieve(video_ids[0], req.question)
    else:
        candidates = await retrieve_multi(video_ids, req.question)

    if not candidates:
        return QueryResponse(
            answer="No relevant content found for your question.",
            timestamps=[], primary_timestamp=None, relevant_chunks=[]
        )

    top_chunks = await rerank(req.question, candidates, top_k=settings.rerank_top_k)

    # Generate answer — pass video_map for multi-video attribution
    single_duration = videos[0].duration if not is_multi else None
    response = await generate_answer(
        req.question, top_chunks,
        video_duration=single_duration,
        video_map=video_map if is_multi else None,
    )
    return response


# ── Video Discovery ───────────────────────────────────────────────────────────

@app.post("/api/discover")
async def discover_relevant_videos(req: VideoDiscoveryRequest):
    """
    Find which ready videos are relevant to a question without specifying video IDs.
    Used for 'which video talks about X?' queries across the entire library.
    Returns videos ranked by relevance with snippet context.
    """
    all_videos = await db.list_videos()
    ready_videos = [v for v in all_videos if v.status == VideoStatus.ready]

    if not ready_videos:
        return {"videos": [], "question": req.question, "total_searched": 0}

    video_ids = [v.id for v in ready_videos]
    scored = await discover_videos(video_ids, req.question)

    id_to_video = {v.id: v for v in ready_videos}
    results = []
    for r in scored:
        vid = id_to_video.get(r["video_id"])
        if not vid:
            continue
        top_chunk = r.get("top_chunk")
        snippet = ""
        topic_title = ""
        if top_chunk:
            snippet = top_chunk.get("text", "")[:300]
            topic_title = top_chunk.get("metadata", {}).get("topic_title", "")
        results.append({
            "video_id": vid.id,
            "filename": vid.filename,
            "duration": vid.duration,
            "transcript_source": vid.transcript_source.value,
            "relevance_score": round(r["score"], 4),
            "snippet": snippet,
            "topic_title": topic_title,
        })

    return {
        "videos": results,
        "question": req.question,
        "total_searched": len(ready_videos),
    }


# ── WebSocket progress ────────────────────────────────────────────────────────

@app.websocket("/api/ws/progress/{video_id}")
async def websocket_progress(websocket: WebSocket, video_id: str):
    await manager.connect(video_id, websocket)
    try:
        # Send current status immediately
        video = await db.get_video(video_id)
        if video:
            await websocket.send_json({
                "type": "status",
                "data": {"video_id": video_id, "status": video.status.value}
            })
        while True:
            # Keep connection alive, wait for client ping
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(video_id, websocket)


# ── Keyframe image serving ────────────────────────────────────────────────────

@app.get("/api/keyframes/{video_id}/{filename}")
async def serve_keyframe(video_id: str, filename: str):
    frame_path = Path(settings.keyframes_dir) / video_id / filename
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(frame_path))


# ── Frontend ──────────────────────────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(index_path.read_text())


@app.get("/{path:path}", response_class=HTMLResponse)
async def serve_spa(path: str):
    """Catch-all for SPA routing."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>Not found</h1>", status_code=404)
