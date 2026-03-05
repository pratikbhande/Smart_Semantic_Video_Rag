"""OCR processing using OpenAI Vision API."""

from typing import Dict
import numpy as np
from openai import OpenAI
import config
from utils import encode_image_to_base64


class OCRProcessor:
    """Extract text from frames using OpenAI Vision API."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    def extract_text(self, frame: np.ndarray) -> Dict:
        """Extract text from frame using GPT-4o Vision."""
        if not config.ENABLE_OCR:
            return {"full_text": "", "confidence": 0.0, "word_count": 0}
        
        try:
            # Encode frame to base64
            base64_image = encode_image_to_base64(frame)
            
            # Create focused OCR prompt
            prompt = """Extract ALL visible text from this image.

Return ONLY the text you see, maintaining the original layout and order.
If there is no text, return "NO_TEXT".

Do not add any explanations, just the extracted text."""
            
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
                                    "detail": "high"  # High detail for OCR
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.0  # Deterministic for OCR
            )
            
            # Get extracted text
            extracted_text = response.choices[0].message.content.strip()
            
            # Clean up
            if extracted_text == "NO_TEXT" or not extracted_text:
                return {"full_text": "", "confidence": 0.0, "word_count": 0}
            
            word_count = len(extracted_text.split())
            
            # High confidence since it's GPT-4o Vision
            return {
                "full_text": extracted_text,
                "confidence": 95.0,  # GPT-4o is very accurate
                "word_count": word_count
            }
            
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return {"full_text": "", "confidence": 0.0, "word_count": 0}