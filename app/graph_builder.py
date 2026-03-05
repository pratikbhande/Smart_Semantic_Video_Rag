"""Build temporal semantic graph with per-video tracking."""

from typing import List, Dict, Tuple
import networkx as nx
import numpy as np
from collections import defaultdict
import config
from utils import cosine_similarity


class TemporalGraphBuilder:
    """Build and manage temporal semantic graph with video-level insights."""
    
    def __init__(self):
        """Initialize directed graph."""
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.node_metadata = {}
        self.video_stats = defaultdict(lambda: {
            "keyframe_count": 0,
            "nodes": [],
            "avg_importance": 0.0
        })
    
    def build_graph(self, keyframes_data: List[Dict], 
                   embeddings: np.ndarray) -> nx.DiGraph:
        """Build temporal graph with semantic and temporal edges."""
        print("\n" + "="*60)
        print("🔗 Building Temporal Semantic Graph")
        print("="*60)
        
        # Add nodes
        for i, kf_data in enumerate(keyframes_data):
            node_id = f"{kf_data['video_id']}_{kf_data['frame_number']:06d}"
            video_id = kf_data['video_id']
            
            self.graph.add_node(
                node_id,
                timestamp=kf_data["timestamp"],
                frame_number=kf_data["frame_number"],
                video_id=video_id,
                video_name=kf_data.get("video_name", "unknown"),
                scene_change_score=kf_data["scene_change_score"]
            )
            
            self.node_embeddings[node_id] = embeddings[i]
            self.node_metadata[node_id] = kf_data
            
            # Track per-video stats
            self.video_stats[video_id]["keyframe_count"] += 1
            self.video_stats[video_id]["nodes"].append(node_id)
        
        print(f"📊 Added {self.graph.number_of_nodes()} nodes")
        
        # Add semantic edges (within and across videos)
        semantic_edges = 0
        nodes = list(self.graph.nodes())
        
        print("🔍 Computing semantic similarities...")
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i = nodes[i]
                node_j = nodes[j]
                
                # Compute similarity
                similarity = cosine_similarity(
                    self.node_embeddings[node_i],
                    self.node_embeddings[node_j]
                )
                
                # Check if same video or cross-video
                video_i = self.graph.nodes[node_i]["video_id"]
                video_j = self.graph.nodes[node_j]["video_id"]
                same_video = (video_i == video_j)
                
                # Temporal distance (only relevant for same video)
                if same_video:
                    frame_i = self.graph.nodes[node_i]["frame_number"]
                    frame_j = self.graph.nodes[node_j]["frame_number"]
                    temporal_distance = abs(frame_j - frame_i)
                else:
                    temporal_distance = 999999  # Large number for cross-video
                
                # Add edge if similarity is high enough
                if similarity >= config.SIMILARITY_THRESHOLD:
                    # Apply temporal decay for same video
                    if same_video and temporal_distance <= config.MAX_TEMPORAL_DISTANCE:
                        temporal_factor = config.TEMPORAL_DECAY ** (temporal_distance / 10)
                        edge_weight = similarity * temporal_factor
                    else:
                        edge_weight = similarity
                    
                    self.graph.add_edge(
                        node_i, node_j,
                        weight=edge_weight,
                        similarity=similarity,
                        edge_type="semantic",
                        same_video=same_video,
                        temporal_distance=temporal_distance if same_video else None
                    )
                    semantic_edges += 1
        
        print(f"🔗 Added {semantic_edges} semantic edges")
        
        # Add temporal edges (consecutive frames in same video)
        temporal_edges = 0
        
        # Group nodes by video
        video_nodes = defaultdict(list)
        for node in nodes:
            video_id = self.graph.nodes[node]["video_id"]
            video_nodes[video_id].append(node)
        
        # Sort each video's nodes by frame number
        for video_id, v_nodes in video_nodes.items():
            sorted_nodes = sorted(
                v_nodes,
                key=lambda n: self.graph.nodes[n]["frame_number"]
            )
            
            # Connect consecutive frames
            for i in range(len(sorted_nodes) - 1):
                node_curr = sorted_nodes[i]
                node_next = sorted_nodes[i + 1]
                
                time_gap = (
                    self.graph.nodes[node_next]["timestamp"] - 
                    self.graph.nodes[node_curr]["timestamp"]
                )
                
                self.graph.add_edge(
                    node_curr, node_next,
                    weight=1.0,  # Strongest connection
                    edge_type="temporal",
                    same_video=True,
                    time_gap=time_gap
                )
                temporal_edges += 1
        
        print(f"⏱️  Added {temporal_edges} temporal edges")
        
        # Compute PageRank importance
        if self.graph.number_of_nodes() > 0:
            importance_scores = nx.pagerank(self.graph, weight="weight")
            nx.set_node_attributes(self.graph, importance_scores, "importance")
            
            # Update video stats with average importance
            for video_id, stats in self.video_stats.items():
                importances = [
                    importance_scores[node] 
                    for node in stats["nodes"]
                ]
                stats["avg_importance"] = np.mean(importances) if importances else 0.0
        
        # Print per-video statistics
        print("\n" + "-"*60)
        print("📹 Per-Video Statistics:")
        print("-"*60)
        
        for video_id, stats in self.video_stats.items():
            video_name = self.graph.nodes[stats["nodes"][0]]["video_name"]
            print(f"   {video_name}")
            print(f"      Keyframes: {stats['keyframe_count']}")
            print(f"      Avg Importance: {stats['avg_importance']:.4f}")
        
        print("="*60 + "\n")
        
        return self.graph
    
    def get_temporal_context(self, node_id: str, window_size: int = 3) -> List[str]:
        """Get temporal context (nearby frames) for a node."""
        if node_id not in self.graph:
            return []
        
        video_id = self.graph.nodes[node_id]["video_id"]
        frame_number = self.graph.nodes[node_id]["frame_number"]
        
        # Find nodes in temporal window
        context_nodes = []
        for node in self.graph.nodes():
            if self.graph.nodes[node]["video_id"] == video_id:
                node_frame = self.graph.nodes[node]["frame_number"]
                if abs(node_frame - frame_number) <= window_size and node != node_id:
                    context_nodes.append(node)
        
        # Sort by frame number
        context_nodes.sort(key=lambda n: self.graph.nodes[n]["frame_number"])
        
        return context_nodes
    
    def get_video_subgraph(self, video_id: str) -> nx.DiGraph:
        """Get subgraph for a specific video."""
        if video_id not in self.video_stats:
            return nx.DiGraph()
        
        nodes = self.video_stats[video_id]["nodes"]
        return self.graph.subgraph(nodes).copy()
    
    def get_graph_data(self, video_id: str = None) -> Dict:
        """Get graph data for visualization."""
        if video_id:
            graph = self.get_video_subgraph(video_id)
        else:
            graph = self.graph
        
        nodes_data = []
        edges_data = []
        
        for node in graph.nodes():
            node_data = dict(graph.nodes[node])
            node_data["id"] = node
            nodes_data.append(node_data)
        
        for edge in graph.edges(data=True):
            edge_data = {
                "source": edge[0],
                "target": edge[1],
                **edge[2]
            }
            edges_data.append(edge_data)
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "stats": self.video_stats.get(video_id, {}) if video_id else None
        }