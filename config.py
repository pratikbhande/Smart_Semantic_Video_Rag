"""Configuration for Video RAG."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# OPENAI
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

VISION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072

# ============================================================
# KEYFRAME EXTRACTION - SIMPLE & CLEAR
# ============================================================

# Check video every N seconds
SAMPLE_INTERVAL_SECONDS = 1.0

# For backwards compatibility (same as SAMPLE_INTERVAL_SECONDS)
SAMPLE_FPS = 1

# Minimum time between keyframes (prevent duplicates)
MIN_KEYFRAME_INTERVAL_SECONDS = 0.5

# Text must be <85% similar to trigger (15% different)
TEXT_SIMILARITY_THRESHOLD = 0.85

# 30% of pixels must change
PIXEL_CHANGE_THRESHOLD = 0.80

# Text detection enabled
TEXT_DETECTION_ENABLED = True

# ============================================================
# AUDIO
# ============================================================
ENABLE_AUDIO = True
WHISPER_MODEL = "base"
AUDIO_SAMPLE_RATE = 16000

# ============================================================
# VIDEO CHUNKS
# ============================================================
CHUNK_DURATION = 5.0
CHUNK_BUFFER = 2.5

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
AUDIO_DIR = DATA_DIR / "audio"

for d in [DATA_DIR, UPLOADS_DIR, KEYFRAMES_DIR, PROCESSED_DIR, CHUNKS_DIR, AUDIO_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ============================================================
# CHROMADB
# ============================================================
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")
COLLECTION_NAME = "video_rag_keyframes"

# ============================================================
# GRAPH
# ============================================================
SIMILARITY_THRESHOLD = 0.5
TEMPORAL_DECAY = 0.9
MAX_TEMPORAL_DISTANCE = 10

# ============================================================
# STREAMLIT
# ============================================================
PAGE_TITLE = "Video RAG"
PAGE_ICON = "🎬"
LAYOUT = "wide"

# ============================================================
# DEBUGGING
# ============================================================
DEBUG_MODE = False
VERBOSE_LOGGING = False