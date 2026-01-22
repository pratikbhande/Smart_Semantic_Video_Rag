"""ML-based scene detection with adjusted sensitivity."""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Tuple
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity


class CNNSceneDetector:
    """Scene detection using pre-trained ResNet50 for feature extraction."""
    
    def __init__(self):
        """Initialize CNN model for feature extraction."""
        print("🤖 Loading ResNet50 model for scene detection...")
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"✅ ResNet50 loaded on {self.device}\n")
    
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract deep features from frame."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        input_tensor = self.transform(frame_rgb)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(input_batch)
        
        # Flatten and normalize
        features = features.squeeze().cpu().numpy()
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def compute_scene_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute semantic similarity between frames."""
        # Extract features
        features1 = self.extract_features(frame1)
        features2 = self.extract_features(frame2)
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            features1.reshape(1, -1),
            features2.reshape(1, -1)
        )[0][0]
        
        # Convert to dissimilarity
        dissimilarity = 1.0 - similarity
        
        return dissimilarity
    
    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray, 
                          threshold: float = 0.30) -> Tuple[bool, float, str]:
        """
        Detect scene change between frames.
        
        Returns:
            (is_scene_change, change_score, change_type)
        """
        # CNN-based dissimilarity
        cnn_score = self.compute_scene_similarity(frame1, frame2)
        
        # Histogram for backup
        hist_score = self._compute_histogram_difference(frame1, frame2)
        
        # Weighted combination - favor CNN (80/20 instead of 70/30)
        combined_score = 0.8 * cnn_score + 0.2 * hist_score
        
        # Determine change type with more granularity
        if combined_score > 0.6:
            change_type = "major_scene_cut"
            is_change = True
        elif combined_score > 0.45:
            change_type = "scene_change"
            is_change = True
        elif combined_score > threshold:
            change_type = "moderate_change"
            is_change = True
        else:
            change_type = "minor_change"
            is_change = False
        
        return is_change, combined_score, change_type
    
    def _compute_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute histogram difference (backup method)."""
        # Convert to HSV
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Compute histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [8, 8, 8], 
                            [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [8, 8, 8], 
                            [0, 180, 0, 256, 0, 256])
        
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        # Compute correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        dissimilarity = max(0.0, 1.0 - similarity)
        
        return dissimilarity
