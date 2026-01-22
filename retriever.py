"""Query and retrieval system with video chunk support."""

from typing import List, Dict
import numpy as np
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import config
from graph_builder import TemporalGraphBuilder
from video_processor import AdvancedVideoProcessor


class VideoRAGRetriever:
    """Retrieve relevant keyframes and video chunks."""
    
    def __init__(self, graph_builder: TemporalGraphBuilder = None):
        """Initialize retriever."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.graph_builder = graph_builder
        
        # Connect to ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=config.CHROMA_PERSIST_DIR,
            anonymized_telemetry=False
        ))
        
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.COLLECTION_NAME
            )
        except:
            self.collection = None
    
    def query(self, query_text: str, top_k: int = 5, 
              include_context: bool = True,
              return_chunks: bool = True) -> List[Dict]:
        """Query for relevant keyframes with video chunks."""
        if not self.collection:
            return []
        
        # Generate query embedding
        try:
            response = self.client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=[query_text],
                dimensions=config.EMBEDDING_DIMENSIONS
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return []
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        # Process results
        retrieved_keyframes = []
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity
            similarity = 1.0 - distance
            
            # Get importance from graph
            importance = 0.5
            if self.graph_builder and doc_id in self.graph_builder.graph:
                importance = self.graph_builder.graph.nodes[doc_id].get("importance", 0.5)
            
            # Combined score
            final_score = 0.7 * similarity + 0.3 * importance
            
            # Calculate video chunk timestamps
            keyframe_time = metadata["timestamp"]
            chunk_start = max(0, keyframe_time - config.CHUNK_BUFFER)
            chunk_end = keyframe_time + config.CHUNK_BUFFER
            
            result = {
                "id": doc_id,
                "metadata": metadata,
                "similarity": similarity,
                "importance": importance,
                "score": final_score,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "chunk_duration": chunk_end - chunk_start
            }
            
            # Add temporal context
            if include_context and self.graph_builder:
                context_nodes = self.graph_builder.get_temporal_context(doc_id, window_size=2)
                result["context_nodes"] = context_nodes
            
            retrieved_keyframes.append(result)
        
        # Sort by final score
        retrieved_keyframes.sort(key=lambda x: x["score"], reverse=True)
        
        return retrieved_keyframes[:top_k]
    
    def get_video_path_from_id(self, video_id: str) -> str:
        """Get original video path from video_id."""
        # Check uploads directory
        for video_file in config.UPLOADS_DIR.iterdir():
            if video_file.is_file():
                from utils import generate_video_id
                if generate_video_id(str(video_file)) == video_id:
                    return str(video_file)
        return None
    
    def extract_chunk_for_result(self, result: Dict) -> str:
        """Extract video chunk for a result."""
        video_id = result["metadata"]["video_id"]
        chunk_start = result["chunk_start"]
        chunk_end = result["chunk_end"]
        
        # Get video path
        video_path = self.get_video_path_from_id(video_id)
        if not video_path:
            return None
        
        # Generate chunk filename
        chunk_filename = f"{video_id}_chunk_{int(chunk_start)}_{int(chunk_end)}.mp4"
        chunk_path = config.CHUNKS_DIR / chunk_filename
        
        # Check if already exists
        if chunk_path.exists():
            return str(chunk_path)
        
        # Extract chunk
        processor = AdvancedVideoProcessor(video_path)
        success = processor.extract_video_chunk(chunk_start, chunk_end, str(chunk_path))
        
        if success:
            return str(chunk_path)
        
        return None