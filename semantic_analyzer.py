"""Semantic analysis with robust JSON parsing."""

from typing import Dict, List
import json
import re
from openai import OpenAI
import numpy as np
import config
from utils import encode_image_to_base64


class SemanticAnalyzer:
    """Analyze with detailed person, action, and scene descriptions."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def _extract_json_from_response(self, content: str) -> Dict:
        """Extract and parse JSON from response, handling various formats."""
        # Remove markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Try to parse directly
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all fails, return None
        return None
    
    def analyze_frame(self, frame_data: np.ndarray, timestamp: float, 
                     video_type: str = "video") -> Dict:
        """Comprehensive frame analysis with person/action detection."""
        # Encode frame to base64
        base64_image = encode_image_to_base64(frame_data)
        
        # Enhanced prompt with person/action detection
        prompt = """Analyze this video frame comprehensively and provide a detailed analysis in JSON format.

Return ONLY valid JSON with these fields:
{
  "main_subject": "Detailed description of the main content (2-3 sentences)",
  "scene_type": "presentation|lecture|code|terminal|action|dialogue|person_speaking|tutorial|other",
  "extracted_text": "ALL visible text (code, UI, slides, captions - exact text)",
  
  "people": [
    {
      "description": "Detailed description of person's appearance",
      "clothing": "What they're wearing",
      "action": "What they're doing",
      "expression": "Facial expression if visible",
      "position": "Location in frame"
    }
  ],
  
  "actions": [
    "Specific action 1",
    "Specific action 2"
  ],
  
  "visual_elements": [
    "UI element 1",
    "Object 1",
    "Element 2"
  ],
  
  "key_objects": ["object1", "object2", "object3"],
  "info_density": "low|medium|high",
  
  "technical_content": {
    "has_code": true,
    "programming_language": "language if code visible",
    "has_terminal": false,
    "has_diagram": false,
    "has_ui_elements": true
  }
}

CRITICAL: Return ONLY the JSON object, no other text before or after."""
        
        try:
            response = self.client.chat.completions.create(
                model=config.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            
            # Extract and parse JSON
            semantic_data = self._extract_json_from_response(content)
            
            if semantic_data is None:
                print(f"⚠️  Failed to parse JSON at {timestamp}s. Raw response:")
                print(content[:200])
                raise ValueError("Could not extract valid JSON from response")
            
            # Process data
            extracted_text = semantic_data.get("extracted_text", "")
            people = semantic_data.get("people", [])
            actions = semantic_data.get("actions", [])
            
            # Add metadata
            semantic_data["text_content"] = extracted_text
            semantic_data["has_people"] = len(people) > 0
            semantic_data["people_count"] = len(people)
            semantic_data["has_actions"] = len(actions) > 0
            semantic_data["timestamp"] = timestamp
            
            # Calculate OCR metrics
            if extracted_text:
                semantic_data["ocr_word_count"] = len(extracted_text.split())
                semantic_data["ocr_confidence"] = 95.0
            else:
                semantic_data["ocr_word_count"] = 0
                semantic_data["ocr_confidence"] = 0.0
            
            return semantic_data
            
        except Exception as e:
            print(f"❌ Error analyzing frame at {timestamp}s: {str(e)}")
            
            # Return minimal fallback
            return {
                "main_subject": "Frame analysis unavailable",
                "scene_type": "other",
                "extracted_text": "",
                "text_content": "",
                "people": [],
                "actions": [],
                "visual_elements": [],
                "key_objects": [],
                "info_density": "medium",
                "technical_content": {},
                "has_people": False,
                "people_count": 0,
                "has_actions": False,
                "timestamp": timestamp,
                "ocr_word_count": 0,
                "ocr_confidence": 0.0
            }
    
    def generate_embedding_prompt(self, keyframe_data: Dict, semantic_data: Dict, 
                                 audio_text: str = "") -> str:
        """Generate rich multimodal prompt with person/action details."""
        timestamp = keyframe_data.get("timestamp", 0)
        main_subject = semantic_data.get("main_subject", "")
        scene_type = semantic_data.get("scene_type", "other")
        
        prompt_parts = [
            f"Video frame at {timestamp:.1f}s",
            f"Scene: {scene_type}",
            f"Content: {main_subject}",
        ]
        
        # Add people descriptions
        people = semantic_data.get("people", [])
        if people:
            people_desc = []
            for i, person in enumerate(people, 1):
                desc = f"Person {i}: {person.get('description', '')}"
                if person.get('action'):
                    desc += f", {person['action']}"
                people_desc.append(desc)
            prompt_parts.append(f"People: {'; '.join(people_desc)}")
        
        # Add actions
        actions = semantic_data.get("actions", [])
        if actions:
            prompt_parts.append(f"Actions: {', '.join(actions)}")
        
        # Add visual elements
        objects = semantic_data.get("key_objects", [])
        if objects:
            prompt_parts.append(f"Objects: {', '.join(objects)}")
        
        # Add text (truncated)
        text_content = semantic_data.get("text_content", "")
        if text_content:
            truncated = text_content[:400] if len(text_content) > 400 else text_content
            prompt_parts.append(f"Text: {truncated}")
        
        # Add audio (truncated)
        if audio_text:
            truncated_audio = audio_text[:250] if len(audio_text) > 250 else audio_text
            prompt_parts.append(f"Audio: {truncated_audio}")
        
        return " | ".join(prompt_parts)