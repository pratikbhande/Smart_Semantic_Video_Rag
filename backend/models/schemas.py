from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class VideoStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class ChunkType(str, Enum):
    summary = "summary"
    detail = "detail"


class TranscriptSource(str, Enum):
    embedded = "Embedded Subtitles"
    sidecar = "Sidecar SRT/VTT"
    whisper = "Whisper Transcription"
    no_audio = "No Audio"
    no_speech = "No Speech Detected"
    none = "None"


class VideoUpload(BaseModel):
    glossary_names: Optional[str] = Field(None, description="Comma-separated proper nouns for transcription accuracy")
    glossary_terms: Optional[str] = Field(None, description="Comma-separated domain-specific terms")
    domain_context: Optional[str] = Field(None, description="Short description of video domain for analysis context")


class VideoMetadata(BaseModel):
    id: str
    filename: str
    duration: Optional[float] = None
    status: VideoStatus = VideoStatus.pending
    transcript_source: TranscriptSource = TranscriptSource.none
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingStatus(BaseModel):
    video_id: str
    stage: str
    progress: float = Field(ge=0, le=100)
    current_step: str
    error: Optional[str] = None


class TranscriptSegment(BaseModel):
    id: Optional[int] = None
    video_id: str
    start_time: float
    end_time: float
    text: str
    confidence: float = 1.0


class TopicSegment(BaseModel):
    id: Optional[int] = None
    video_id: str
    start_time: float
    end_time: float
    title: str
    summary: str
    key_entities: List[str] = Field(default_factory=list)


class KeyframeData(BaseModel):
    id: Optional[int] = None
    video_id: str
    timestamp: float
    frame_path: str
    scene_type: str = "unknown"
    visual_analysis: Dict[str, Any] = Field(default_factory=dict)


class ChunkData(BaseModel):
    id: str
    text: str
    chunk_type: ChunkType
    start_time: float
    end_time: float
    topic_title: Optional[str] = None
    visual_context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TimestampRef(BaseModel):
    start: float
    end: float
    relevance: float
    video_id: Optional[str] = None
    video_filename: Optional[str] = None


class QueryRequest(BaseModel):
    video_id: Optional[str] = None          # single-video query
    video_ids: Optional[List[str]] = None   # multi-video query
    question: str


class QueryResponse(BaseModel):
    answer: str
    timestamps: List[TimestampRef] = Field(default_factory=list)
    primary_timestamp: Optional[float] = None
    relevant_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    source_video_id: Optional[str] = None  # video that primary_timestamp belongs to
    supporting_videos: List[Dict[str, Any]] = Field(default_factory=list)  # other relevant videos for UI display


class WebSocketMessage(BaseModel):
    type: str  # "progress" | "complete" | "error"
    data: Dict[str, Any]


class VideoDiscoveryRequest(BaseModel):
    question: str


class VideoDiscoveryResult(BaseModel):
    video_id: str
    filename: str
    relevance_score: float
    snippet: str = ""
    topic_title: str = ""
    duration: Optional[float] = None
