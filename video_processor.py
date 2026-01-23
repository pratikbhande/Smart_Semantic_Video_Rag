"""Simple, reliable keyframe extraction - text first, then pixels."""

import cv2
from pathlib import Path
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass
from openai import OpenAI
import base64
import config
from utils import generate_video_id, format_timestamp
import difflib


@dataclass
class Keyframe:
    """Data class for keyframe information."""
    video_id: str
    video_name: str
    frame_number: int
    timestamp: float
    frame_path: str
    scene_change_score: float
    change_type: str
    frame_data: np.ndarray
    frame_shape: Tuple[int, int, int]
    detection_reasons: List[str]
    extracted_text: str


class AdvancedVideoProcessor:
    """Simple keyframe extraction: text changes OR 30% pixels."""
    
    def __init__(self, video_path: str):
        """Initialize video processor."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.name
        self.video_id = generate_video_id(str(video_path))
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"📹 Video: {self.video_name}")
        print(f"   Duration: {format_timestamp(self.duration)}")
        print(f"   FPS: {self.fps:.1f} | Total Frames: {self.total_frames}")
        print(f"   Checking every {config.SAMPLE_INTERVAL_SECONDS:.1f}s")
        print(f"   Text similarity threshold: {config.TEXT_SIMILARITY_THRESHOLD*100:.0f}%")
        print(f"   Pixel change threshold: {config.PIXEL_CHANGE_THRESHOLD*100:.0f}%")
        print(f"{'='*80}\n")
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame to base64."""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    
    def _extract_text(self, frame: np.ndarray) -> str:
        """
        Extract text using OpenAI Vision API.
        HARDCODED response format to avoid refusals.
        """
        try:
            response = self.client.chat.completions.create(
                model=config.VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a text extraction tool. Extract ONLY visible text overlays, captions, signs, or subtitles from this image.

    RULES:
    - Do NOT describe people, faces, or identify anyone
    - Do NOT say "I'm sorry" or "I can't"
    - ONLY extract visible TEXT (words, captions, signs)
    - If NO text is visible, respond with exactly: NONE

    Examples:
    - Image with "Subscribe Now" overlay → Return: Subscribe Now
    - Image with person but no text → Return: NONE
    - Image with "Episode 5" caption → Return: Episode 5

    Your response:"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_frame(frame)}
                        }
                    ]
                }],
                max_tokens=300,
                temperature=0.0
            )
            
            text = response.choices[0].message.content.strip()
            
            # Filter out any refusal that slips through
            refusal_keywords = ["sorry", "can't", "cannot", "unable", "apologize"]
            text_lower = text.lower()
            
            if any(keyword in text_lower for keyword in refusal_keywords):
                return ""  # Treat as no text
            
            # Check for explicit NONE
            if text.upper() == "NONE" or not text:
                return ""
            
            return text
            
        except Exception as e:
            print(f"❌ OCR error: {e}")
            return ""
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (0.0 = completely different, 1.0 = identical)."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _pixel_change_percent(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate what percentage of pixels changed."""
        # Resize for speed
        f1 = cv2.resize(frame1, (320, 240))
        f2 = cv2.resize(frame2, (320, 240))
        
        # Grayscale
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        
        # Difference
        diff = cv2.absdiff(g1, g2)
        changed_pixels = np.count_nonzero(diff > 15)
        total_pixels = diff.size
        
        return changed_pixels / total_pixels
    
    def extract_keyframes(self) -> List[Keyframe]:
        """
        SMART extraction:
        1. Text APPEARS (was none, now has text) → Extract ✅
        2. Text CHANGES (was X, now Y) → Extract ✅
        3. Text SAME → Skip, ignore pixels ⏭️
        4. NO text → Check 30% pixel change from last keyframe
        """
        keyframes = []
        
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        print("🔍 Processing first frame...")
        first_text = self._extract_text(frame)
        print(f"   Text: {first_text[:80] if first_text else '(none - will use pixel detection)'}\n")
        
        keyframes.append(Keyframe(
            video_id=self.video_id,
            video_name=self.video_name,
            frame_number=0,
            timestamp=0.0,
            frame_path=self._save_frame(frame, 0, 0.0),
            scene_change_score=1.0,
            change_type="initial",
            frame_data=frame.copy(),
            frame_shape=frame.shape,
            detection_reasons=["first_frame"],
            extracted_text=first_text
        ))
        
        # Track LAST KEYFRAME
        last_keyframe_frame = frame.copy()
        last_keyframe_text = first_text
        last_keyframe_time = 0.0
        
        print(f"🚀 Starting SMART extraction...")
        print(f"   Rule 1: Text APPEARS → Extract immediately")
        print(f"   Rule 2: Text CHANGES → Extract")
        print(f"   Rule 3: Text SAME → Skip (ignore pixel changes)")
        print(f"   Rule 4: NO text → Extract when 30%+ pixels changed from last keyframe")
        print(f"   Checking every {config.SAMPLE_INTERVAL_SECONDS:.1f}s\n")
        
        # Stats
        total_frames_read = 0
        frames_checked = 0
        text_appeared = 0
        text_changed = 0
        visual_keyframes = 0
        skipped_text_same = 0
        skipped_pixels_low = 0
        next_check_time = config.SAMPLE_INTERVAL_SECONDS
        
        while True:
            # Read next frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            total_frames_read += 1
            
            # Get ACTUAL timestamp
            actual_timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            actual_timestamp = actual_timestamp_ms / 1000.0
            
            # Should we check this frame?
            if actual_timestamp < next_check_time:
                continue
            
            # Time to check!
            frames_checked += 1
            next_check_time = actual_timestamp + config.SAMPLE_INTERVAL_SECONDS
            
            # Progress
            if frames_checked % 5 == 0:
                progress = (actual_timestamp / self.duration) * 100
                print(f"📊 {actual_timestamp:.1f}s / {self.duration:.0f}s ({progress:.0f}%) | Keyframes: {len(keyframes)}")
            
            # Extract text
            current_text = self._extract_text(frame)
            has_text_now = len(current_text.strip()) > 0
            had_text_before = len(last_keyframe_text.strip()) > 0
            
            # Decision logic
            should_extract = False
            change_type = ""
            score = 0.0
            reason = ""
            
            # ============================================
            # CASE 1: Text APPEARED (no text → has text)
            # ============================================
            if has_text_now and not had_text_before:
                should_extract = True
                change_type = "text_appeared"
                score = 1.0
                reason = "Text appeared on screen"
                print(f"   ✅ {actual_timestamp:.1f}s: TEXT APPEARED!")
                print(f"      New text: {current_text[:60]}")
                text_appeared += 1
            
            # ============================================
            # CASE 2: Text CHANGED (both have text, but different)
            # ============================================
            elif has_text_now and had_text_before:
                text_sim = self._text_similarity(last_keyframe_text, current_text)
                text_is_different = text_sim < config.TEXT_SIMILARITY_THRESHOLD
                
                if text_is_different:
                    should_extract = True
                    change_type = "text_change"
                    score = 1.0 - text_sim
                    reason = f"Text changed (similarity: {text_sim:.2f})"
                    print(f"   ✅ {actual_timestamp:.1f}s: TEXT CHANGED (sim={text_sim:.2f})")
                    print(f"      Old: {last_keyframe_text[:40]}")
                    print(f"      New: {current_text[:40]}")
                    text_changed += 1
                else:
                    # Text exists but is the same - SKIP (even if pixels changed)
                    skipped_text_same += 1
                    if frames_checked % 10 == 0:
                        print(f"   ⚪ {actual_timestamp:.1f}s: Text unchanged, skipping")
            
            # ============================================
            # CASE 3: Text DISAPPEARED (had text → no text)
            # ============================================
            elif not has_text_now and had_text_before:
                should_extract = True
                change_type = "text_disappeared"
                score = 1.0
                reason = "Text disappeared from screen"
                print(f"   ✅ {actual_timestamp:.1f}s: TEXT DISAPPEARED")
                print(f"      Previous text: {last_keyframe_text[:60]}")
                text_appeared += 1  # Count with appeared
            
            # ============================================
            # CASE 4: NO text (before or now) - check pixels from LAST KEYFRAME
            # ============================================
            elif not has_text_now and not had_text_before:
                pixel_pct = self._pixel_change_percent(last_keyframe_frame, frame)
                pixels_changed = pixel_pct >= config.PIXEL_CHANGE_THRESHOLD
                
                if pixels_changed:
                    should_extract = True
                    change_type = "visual_change"
                    score = pixel_pct
                    reason = f"{pixel_pct*100:.1f}% pixels changed from last keyframe"
                    print(f"   ✅ {actual_timestamp:.1f}s: VISUAL CHANGE ({pixel_pct*100:.1f}%)")
                    print(f"      No text detected")
                    visual_keyframes += 1
                else:
                    skipped_pixels_low += 1
                    if frames_checked % 10 == 0:
                        print(f"   ⚪ {actual_timestamp:.1f}s: Only {pixel_pct*100:.1f}% changed (no text)")
            
            # Extract if needed
            if should_extract:
                # Check minimum spacing
                time_since_last = actual_timestamp - last_keyframe_time
                
                if time_since_last >= config.MIN_KEYFRAME_INTERVAL_SECONDS:
                    keyframes.append(Keyframe(
                        video_id=self.video_id,
                        video_name=self.video_name,
                        frame_number=total_frames_read,
                        timestamp=actual_timestamp,
                        frame_path=self._save_frame(frame, total_frames_read, actual_timestamp),
                        scene_change_score=score,
                        change_type=change_type,
                        frame_data=frame.copy(),
                        frame_shape=frame.shape,
                        detection_reasons=[reason],
                        extracted_text=current_text
                    ))
                    
                    print(f"      🔑 KEYFRAME {len(keyframes)} extracted ({change_type})\n")
                    
                    # Update LAST KEYFRAME tracking
                    last_keyframe_frame = frame.copy()
                    last_keyframe_text = current_text
                    last_keyframe_time = actual_timestamp
                else:
                    print(f"      ⏭️  Skipped (too close: {time_since_last:.1f}s)\n")
        
        self.cap.release()
        
        # Calculate REAL FPS
        real_fps = total_frames_read / self.duration if self.duration > 0 else 0
        
        # Summary
        print(f"\n{'='*80}")
        print(f"✅ DONE! Extracted {len(keyframes)} keyframes")
        print(f"\n📊 Video Analysis:")
        print(f"   Metadata FPS: {self.fps:.1f}")
        print(f"   Actual FPS: {real_fps:.1f} ({total_frames_read} frames / {self.duration:.1f}s)")
        print(f"   Frames checked: {frames_checked}")
        print(f"\n🔑 Keyframes Breakdown:")
        print(f"   Text appeared/disappeared: {text_appeared}")
        print(f"   Text changed: {text_changed}")
        print(f"   Visual changes (no text): {visual_keyframes}")
        print(f"   Total: {len(keyframes)}")
        print(f"\n⏭️  Skipped Frames:")
        print(f"   Text same as last keyframe: {skipped_text_same}")
        print(f"   <30% pixel change (no text): {skipped_pixels_low}")
        print(f"{'='*80}\n")
        
        return keyframes

    
    def _save_frame(self, frame: np.ndarray, frame_num: int, timestamp: float) -> str:
        """Save frame to disk."""
        filename = f"{self.video_id}_frame_{int(timestamp*1000):06d}.jpg"
        path = config.KEYFRAMES_DIR / filename
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(path)
    
    def extract_video_chunk(self, start: float, end: float, output: str) -> bool:
        """Extract video chunk."""
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, fps, (w, h))
            
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
            
            while cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 < end:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


VideoProcessor = AdvancedVideoProcessor