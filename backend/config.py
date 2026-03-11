from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    data_dir: str = Field(default="/data", env="DATA_DIR")
    chroma_dir: str = Field(default="/data/chroma_db", env="CHROMA_DIR")
    upload_dir: str = Field(default="/data/uploads", env="UPLOAD_DIR")
    keyframes_dir: str = Field(default="/data/keyframes", env="KEYFRAMES_DIR")
    audio_dir: str = Field(default="/data/audio", env="AUDIO_DIR")
    chunks_dir: str = Field(default="/data/chunks", env="CHUNKS_DIR")
    max_upload_size_mb: int = Field(default=2000, env="MAX_UPLOAD_SIZE_MB")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Model config
    vision_model: str = "gpt-4o"
    mini_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 3072
    whisper_model: str = "whisper-1"

    # Processing config
    ssim_threshold: float = 0.85
    min_keyframe_interval: float = 1.5      # seconds between keyframes
    scene_threshold: float = 0.3            # 0.3 = sensitive; captures more scene changes including lower-third appearances
    max_keyframes_cap: int = 50             # hard ceiling; with smart OCR pre-filter 50 frames is affordable
    min_keyframes_floor: int = 3
    keyframe_interval_secs: int = 20        # ~1 frame per 20s for interval sampling (long videos)
    scene_detect_max_duration: float = 600.0
    # Lower-third scan config
    lower_third_diff_threshold: float = 0.06   # mean pixel change threshold for new overlay
    audio_chunk_duration: int = 300         # seconds per Whisper chunk
    audio_chunk_overlap: int = 5
    topic_min_duration: float = 30.0
    max_frames_per_batch: int = 10          # GPT-4o vision frames per API call
    gpt4o_vision_concurrency: int = 5       # concurrent GPT-4o calls
    max_tokens_correction: int = 3000
    skip_correction_min_duration: float = 600.0  # skip correction for long videos without glossary

    # Retrieval config
    summary_collection: str = "video_summary_v2"
    detail_collection: str = "video_detail_v2"
    retrieval_top_k: int = 20
    rerank_top_k: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"

    def ensure_dirs(self):
        for d in [self.data_dir, self.chroma_dir, self.upload_dir,
                  self.keyframes_dir, self.audio_dir, self.chunks_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
