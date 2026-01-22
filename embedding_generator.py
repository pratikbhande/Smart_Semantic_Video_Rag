"""Generate and store embeddings using OpenAI and ChromaDB."""

from typing import List, Dict
import numpy as np
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import config


class EmbeddingGenerator:
    """Generate embeddings and store in ChromaDB."""
    
    def __init__(self):
        """Initialize OpenAI client and ChromaDB."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=config.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"description": "Video RAG keyframes with semantic embeddings"}
        )
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts."""
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts,
                dimensions=config.EMBEDDING_DIMENSIONS
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), config.EMBEDDING_DIMENSIONS))
    
    def generate_differential_embeddings(self, absolute_embeddings: np.ndarray) -> np.ndarray:
        """Generate differential embeddings (delta between consecutive frames)."""
        differential = np.zeros_like(absolute_embeddings)
        
        for i in range(1, len(absolute_embeddings)):
            differential[i] = absolute_embeddings[i] - absolute_embeddings[i-1]
        
        return differential
    
    def store_keyframes(self, keyframes_data: List[Dict]):
        """Store keyframes with embeddings in ChromaDB."""
        if not keyframes_data:
            return
        
        # Generate embedding prompts
        prompts = [kf["embedding_prompt"] for kf in keyframes_data]
        
        print(f"Generating embeddings for {len(prompts)} keyframes...")
        
        # Generate embeddings
        absolute_embeddings = self.generate_embeddings(prompts)
        differential_embeddings = self.generate_differential_embeddings(absolute_embeddings)
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, kf_data in enumerate(keyframes_data):
            doc_id = f"{kf_data['video_id']}_{kf_data['frame_number']:06d}"
            
            ids.append(doc_id)
            embeddings.append(absolute_embeddings[i].tolist())
            documents.append(kf_data["embedding_prompt"])
            
            # Compute differential norm
            diff_norm = float(np.linalg.norm(differential_embeddings[i]))
            
            metadata = {
                "video_id": kf_data["video_id"],
                "frame_number": kf_data["frame_number"],
                "timestamp": kf_data["timestamp"],
                "frame_path": kf_data["frame_path"],
                "scene_change_score": kf_data["scene_change_score"],
                "main_subject": kf_data["semantic_data"]["main_subject"],
                "scene_type": kf_data["semantic_data"]["scene_type"],
                "text_content": kf_data["semantic_data"].get("text_content", ""),
                "differential_norm": diff_norm
            }
            
            metadatas.append(metadata)
        
        # Store in ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"Stored {len(ids)} keyframes in ChromaDB\n")
        
        return absolute_embeddings, differential_embeddings
    
    def get_all_keyframes(self, video_id: str = None) -> Dict:
        """Retrieve all keyframes from ChromaDB."""
        if video_id:
            results = self.collection.get(
                where={"video_id": video_id},
                include=["metadatas", "documents", "embeddings"]
            )
        else:
            results = self.collection.get(
                include=["metadatas", "documents", "embeddings"]
            )
        
        return results
    
    def clear_collection(self):
        """Clear all data from collection."""
        self.chroma_client.delete_collection(config.COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.COLLECTION_NAME
        )