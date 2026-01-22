"""Audio processing using Whisper."""

import whisper
import numpy as np
from pathlib import Path
from typing import Dict, List
from moviepy.editor import VideoFileClip
import config


class AudioProcessor:
    """Extract and transcribe audio from videos using Whisper."""
    
    def __init__(self):
        """Initialize Whisper model."""
        if config.ENABLE_AUDIO:
            print(f"Loading Whisper model: {config.WHISPER_MODEL}...")
            self.model = whisper.load_model(config.WHISPER_MODEL)
            print("✅ Whisper model loaded\n")
        else:
            self.model = None
    
    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video."""
        try:
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                print("⚠️  No audio track found in video")
                return False
            
            video.audio.write_audiofile(
                output_path,
                fps=config.AUDIO_SAMPLE_RATE,
                verbose=False,
                logger=None
            )
            
            video.close()
            return True
            
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return False
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper."""
        if not config.ENABLE_AUDIO or self.model is None:
            return {
                "full_text": "",
                "segments": [],
                "language": "en"
            }
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False
            )
            
            # Extract segments with timestamps
            segments = []
            for segment in result.get("segments", []):
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip()
                })
            
            full_text = result.get("text", "").strip()
            
            return {
                "full_text": full_text,
                "segments": segments,
                "language": result.get("language", "en")
            }
            
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return {
                "full_text": "",
                "segments": [],
                "language": "en"
            }
    
    def get_audio_at_timestamp(self, segments: List[Dict], timestamp: float, 
                               window: float = 5.0) -> str:
        """Get audio text around a specific timestamp."""
        if not segments:
            return ""
        
        relevant_segments = []
        
        for segment in segments:
            # Check if segment overlaps with window
            if (segment["start"] <= timestamp + window and 
                segment["end"] >= timestamp - window):
                relevant_segments.append(segment["text"])
        
        return " ".join(relevant_segments)