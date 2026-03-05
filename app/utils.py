"""Utility functions for video processing and visualization."""

import hashlib
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO


def generate_video_id(video_path: str) -> str:
    """Generate unique ID for video based on filename."""
    return hashlib.md5(Path(video_path).name.encode()).hexdigest()[:8]


def encode_image_to_base64(image: np.ndarray, format: str = "JPEG") -> str:
    """Convert numpy image array to base64 string."""
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Resize if too large (max 512px for efficiency)
    max_size = 512
    if max(pil_image.size) > max_size:
        ratio = max_size / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = BytesIO()
    pil_image.save(buffered, format=format, quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_str}"


def compute_histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute histogram correlation (0=different, 1=identical)."""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def compute_perceptual_hash(frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute perceptual hash for frame."""
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size))
    
    # Compute horizontal gradient
    diff = resized[:, 1:] > resized[:, :-1]
    
    return diff.flatten()


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> float:
    """Compute normalized hamming distance between two hashes."""
    return np.sum(hash1 != hash2) / len(hash1)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    if norm_product == 0:
        return 0.0
    
    return dot_product / norm_product