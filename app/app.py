"""Streamlit application for Semantic Flow Video RAG with intelligent keyframe extraction."""

import streamlit as st
from pathlib import Path
import time
from PIL import Image
import plotly.graph_objects as go
import config
from video_processor import AdvancedVideoProcessor
from semantic_analyzer import SemanticAnalyzer
from audio_processor import AudioProcessor
from embedding_generator import EmbeddingGenerator
from graph_builder import TemporalGraphBuilder
from retriever import VideoRAGRetriever
from utils import format_timestamp, generate_video_id
import config


# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .detection-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .text-change {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .visual-change {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    /* Increase font sizes for better screen recording */
    .stMarkdown, .stText, p, div, span {
        font-size: 18px !important;
    }
    .stCaption {
        font-size: 16px !important;
    }
    code {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    .stCodeBlock {
        font-size: 16px !important;
    }
    /* Increase metric font sizes */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
    }
    /* Increase button text */
    .stButton button {
        font-size: 18px !important;
    }
    /* Increase input text */
    .stTextInput input {
        font-size: 18px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = []
if "graph_builder" not in st.session_state:
    st.session_state.graph_builder = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


def process_video(video_file):
    """Process video with intelligent keyframe extraction and multimodal analysis."""
    # Save uploaded file
    video_path = config.UPLOADS_DIR / video_file.name
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # ============================================================
        # STEP 1: Intelligent Keyframe Extraction
        # ============================================================
        status_text.text("🎬 Extracting keyframes with intelligent detection (Text-first + Visual)...")
        progress_bar.progress(5)
        
        processor = AdvancedVideoProcessor(str(video_path))
        keyframes = processor.extract_keyframes()
        
        # Build detection summary
        detection_summary = {"text": 0, "visual": 0, "initial": 0}
        text_changes_list = []
        visual_changes_list = []
        
        for kf in keyframes:
            if kf.change_type == "initial_frame":
                detection_summary["initial"] += 1
            elif kf.change_type == "text_change":
                detection_summary["text"] += 1
                text_changes_list.append(kf)
            elif kf.change_type == "visual_change":
                detection_summary["visual"] += 1
                visual_changes_list.append(kf)
        
        # Display extraction results
        if len(keyframes) == 1:
            st.warning(f"⚠️ Only 1 keyframe detected (initial frame). "
                      f"Video might be very static or single-scene.")
        else:
            st.success(f"✅ Extracted {len(keyframes)} meaningful keyframes")
            
            # Show detection breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Text Changes", detection_summary["text"], 
                         help="Keyframes where text content changed")
            with col2:
                st.metric("🎨 Visual Changes", detection_summary["visual"],
                         help="Keyframes where visuals changed significantly")
            with col3:
                pct_text = (detection_summary["text"] / len(keyframes) * 100) if len(keyframes) > 1 else 0
                st.metric("Text-based %", f"{pct_text:.0f}%",
                         help="Percentage of keyframes detected via text changes")
        
        progress_bar.progress(30)
        
        # ============================================================
        # STEP 2: Audio Processing (Optional)
        # ============================================================
        status_text.text("🎵 Processing audio track...")
        audio_processor = AudioProcessor()
        
        video_id = generate_video_id(str(video_path))
        audio_path = config.AUDIO_DIR / f"{video_id}_audio.wav"
        
        audio_transcription = {"full_text": "", "segments": []}
        if config.ENABLE_AUDIO:
            if audio_processor.extract_audio(str(video_path), str(audio_path)):
                audio_transcription = audio_processor.transcribe_audio(str(audio_path))
                if audio_transcription["segments"]:
                    word_count = len(audio_transcription["full_text"].split())
                    st.success(f"🎵 Audio Transcribed: {len(audio_transcription['segments'])} segments, {word_count} words")
                    # ✅ ADD: Show sample transcription
                    if audio_transcription["full_text"]:
                        with st.expander("📝 View Transcription Sample"):
                            sample = audio_transcription["full_text"][:300]
                            st.markdown(f"<p style='font-size: 18px; line-height: 1.8;'>{sample}{'...' if len(audio_transcription['full_text']) > 300 else ''}</p>", unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Audio track found but transcription is empty")
            else:
                st.warning("⚠️ No audio track found in video")
        
        progress_bar.progress(40)
        
        # ============================================================
        # STEP 3: Semantic Analysis with Vision AI
        # ============================================================
        status_text.text(f"🧠 Analyzing {len(keyframes)} keyframes with Vision AI...")
        analyzer = SemanticAnalyzer()
        
        keyframes_data = []
        people_detected = 0
        frames_with_actions = 0
        frames_with_code = 0
        
        for i, kf in enumerate(keyframes):
            # Semantic analysis (people, actions, scene understanding)
            semantic_data = analyzer.analyze_frame(
                kf.frame_data, 
                kf.timestamp,
                "general"  # Can be enhanced with video type detection
            )
            
            # Update statistics
            if semantic_data.get("has_people"):
                people_detected += 1
            
            if semantic_data.get("has_actions"):
                frames_with_actions += 1
            
            if semantic_data.get("technical_content", {}).get("has_code"):
                frames_with_code += 1
            
            # Get audio context for this timestamp
            audio_text = ""
            if config.ENABLE_AUDIO and audio_transcription["segments"]:
                audio_text = audio_processor.get_audio_at_timestamp(
                    audio_transcription["segments"],
                    kf.timestamp,
                    window=5.0
                )
            
            # Generate rich embedding prompt
            embedding_prompt = analyzer.generate_embedding_prompt(
                {"timestamp": kf.timestamp},
                semantic_data,
                audio_text
            )
            
            # Store keyframe data with extracted text from detection phase
            keyframes_data.append({
                "video_id": kf.video_id,
                "video_name": kf.video_name,
                "frame_number": kf.frame_number,
                "timestamp": kf.timestamp,
                "frame_path": kf.frame_path,
                "scene_change_score": kf.scene_change_score,
                "motion_score": kf.scene_change_score,
                "change_type": kf.change_type,
                "detection_reasons": kf.detection_reasons,
                "extracted_text": kf.extracted_text,  # From OCR during extraction
                "semantic_data": semantic_data,
                "audio_text": audio_text,
                "embedding_prompt": embedding_prompt
            })
            
            # Progress update
            progress_bar.progress(40 + int(40 * (i + 1) / len(keyframes)))
            
            # Detailed status
            if semantic_data.get("has_people"):
                people_count = semantic_data.get("people_count", 1)
                status_text.text(f"🧠 Frame {i+1}/{len(keyframes)}: "
                               f"Analyzing {people_count} person(s)...")
            elif kf.extracted_text:
                status_text.text(f"🧠 Frame {i+1}/{len(keyframes)}: "
                               f"Processing text content...")
            else:
                status_text.text(f"🧠 Frame {i+1}/{len(keyframes)}: "
                               f"Analyzing visual scene...")
        
        # ============================================================
        # STEP 4: Generate Embeddings
        # ============================================================
        status_text.text("🔢 Generating multimodal embeddings...")
        progress_bar.progress(85)
        
        embedder = EmbeddingGenerator()
        absolute_embs, differential_embs = embedder.store_keyframes(keyframes_data)
        
        progress_bar.progress(92)
        
        # ============================================================
        # STEP 5: Build Temporal Semantic Graph
        # ============================================================
        status_text.text("🔗 Building temporal semantic graph...")
        
        if st.session_state.graph_builder is None:
            st.session_state.graph_builder = TemporalGraphBuilder()
        
        st.session_state.graph_builder.build_graph(keyframes_data, absolute_embs)
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # ============================================================
        # Display Statistics
        # ============================================================
        st.markdown("### 📊 Processing Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔑 Keyframes", len(keyframes))
        with col2:
            st.metric("👥 People", people_detected)
        with col3:
            st.metric("🎬 Actions", frames_with_actions)
        with col4:
            frames_with_text = sum(1 for kf in keyframes if kf.extracted_text)
            st.metric("📝 Text Frames", frames_with_text)
        
        # Detailed breakdown
        st.info(
            f"**🔬 Detection Analysis:**\n\n"
            f"• **Total Keyframes:** {len(keyframes)} extracted "
            f"({len(keyframes) / (processor.duration / 60):.1f} per minute)\n"
            f"• **Text-based Detection:** {detection_summary['text']} keyframes "
            f"({detection_summary['text']/len(keyframes)*100:.1f}%)\n"
            f"• **Visual-based Detection:** {detection_summary['visual']} keyframes "
            f"({detection_summary['visual']/len(keyframes)*100:.1f}%)\n"
            f"• **People Detected:** {people_detected} frames with people\n"
            f"• **Actions Detected:** {frames_with_actions} frames with actions\n"
            f"• **Audio Segments:** {len(audio_transcription.get('segments', []))} transcribed"
        )
        
        time.sleep(1.5)
        progress_bar.empty()
        status_text.empty()
        
        return keyframes_data, audio_transcription
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


def visualize_video_graph(graph_builder, video_id=None):
    """Visualize temporal semantic graph with interactive elements."""
    if not graph_builder or graph_builder.graph.number_of_nodes() == 0:
        st.info("📊 No graph data available. Process videos first.")
        return
    
    graph_data = graph_builder.get_graph_data(video_id)
    
    if not graph_data["nodes"]:
        st.info("No nodes in graph")
        return
    
    # Create positions for nodes (timeline-based)
    pos = {}
    for node in graph_data["nodes"]:
        pos[node["id"]] = {
            "x": node["timestamp"],
            "y": node.get("importance", 0.5)
        }
    
    # Separate edge types
    semantic_edges = [e for e in graph_data["edges"] if e["edge_type"] == "semantic"]
    temporal_edges = [e for e in graph_data["edges"] if e["edge_type"] == "temporal"]
    
    traces = []
    
    # Temporal edges (blue, thick)
    for edge in temporal_edges:
        source_pos = pos[edge["source"]]
        target_pos = pos[edge["target"]]
        
        traces.append(
            go.Scatter(
                x=[source_pos["x"], target_pos["x"], None],
                y=[source_pos["y"], target_pos["y"], None],
                mode="lines",
                line=dict(color="rgba(100,150,255,0.6)", width=3),
                hoverinfo="none",
                showlegend=False,
                name="Temporal"
            )
        )
    
    # Semantic edges (red for same video, gray for cross-video)
    for edge in semantic_edges:
        source_pos = pos[edge["source"]]
        target_pos = pos[edge["target"]]
        
        color = "rgba(255,100,100,0.4)" if edge.get("same_video") else "rgba(150,150,150,0.2)"
        width = 1.5 if edge.get("same_video") else 0.5
        
        traces.append(
            go.Scatter(
                x=[source_pos["x"], target_pos["x"], None],
                y=[source_pos["y"], target_pos["y"], None],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="none",
                showlegend=False,
                name="Semantic"
            )
        )
    
    # Nodes (keyframes)
    node_x = [pos[node["id"]]["x"] for node in graph_data["nodes"]]
    node_y = [pos[node["id"]]["y"] for node in graph_data["nodes"]]
    
    # Rich hover text
    node_text = [
        f"<b>{node.get('video_name', 'Unknown')}</b><br>"
        f"⏱ Time: {format_timestamp(node['timestamp'])}<br>"
        f"🔢 Frame: {node['frame_number']}<br>"
        f"⭐ Importance: {node.get('importance', 0):.4f}<br>"
        f"🎯 Type: {node.get('change_type', 'unknown')}"
        for node in graph_data["nodes"]
    ]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=12,
            color=[node.get("importance", 0.5) for node in graph_data["nodes"]],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Importance",
                thickness=15,
                len=0.7
            ),
            line=dict(width=2, color="white")
        ),
        text=node_text,
        hoverinfo="text",
        name="Keyframes"
    )
    
    traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Title
    title = "🔗 Temporal Semantic Graph - All Videos"
    if video_id and video_id in graph_builder.video_stats:
        video_name = graph_data["nodes"][0].get("video_name", "Unknown")
        stats = graph_builder.video_stats[video_id]
        title = (f"🔗 {video_name}<br>"
                f"<sub>Keyframes: {stats['keyframe_count']} | "
                f"Avg Importance: {stats['avg_importance']:.4f}</sub>")
    
    fig.update_layout(
        title=title,
        xaxis_title="Timeline (seconds)",
        yaxis_title="Importance Score (PageRank)",
        showlegend=False,
        hovermode="closest",
        height=600,
        plot_bgcolor="rgba(240,240,240,0.5)",
        xaxis=dict(gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(gridcolor="rgba(200,200,200,0.3)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Graph statistics
    st.markdown("### 📈 Graph Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔑 Keyframes", len(graph_data["nodes"]))
    with col2:
        st.metric("🔗 Temporal Links", len(temporal_edges))
    with col3:
        st.metric("🎯 Semantic Links", len(semantic_edges))
    with col4:
        same_video_semantic = len([e for e in semantic_edges if e.get("same_video")])
        st.metric("📍 Within-Video", same_video_semantic)


def main():
    """Main Streamlit application."""
    st.title("🎬 Intelligent Video RAG System")
    st.markdown("**Text-First Detection + AI Semantic Analysis**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        st.markdown("### 🎯 Detection Strategy")
        st.success(
            "**Priority 1: Text Detection**\n"
            "• OpenAI Vision OCR\n"
            "• Even 1 word change = new keyframe\n\n"
            "**Priority 2: Visual Detection**\n"
            "• 30% pixel change threshold\n"
            "• Fallback when no text"
        )
        
        st.markdown("### 🤖 AI Models")
        st.info(
            f"**Vision:** {config.VISION_MODEL}\n\n"
            f"**Embedding:** {config.EMBEDDING_MODEL}\n\n"
            f"**Whisper:** {config.WHISPER_MODEL if config.ENABLE_AUDIO else 'Disabled'}"
        )
        
        st.markdown("### 🔧 Features")
        st.success(
            "✅ Intelligent Text-First Detection\n\n"
            "✅ OpenAI Vision Analysis\n\n"
            f"{'✅' if config.ENABLE_AUDIO else '❌'} Audio Transcription\n\n"
            "✅ Semantic Graph Building"
        )
        
        st.markdown("---")
        
        if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
            try:
                embedder = EmbeddingGenerator()
                embedder.clear_collection()
                st.session_state.processed_videos = []
                st.session_state.graph_builder = None
                st.session_state.retriever = None
                st.success("✅ All data cleared!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Process",
        "🔍 Query & Retrieve",
        "📊 Graph Analysis",
        "🖼️ Keyframes Gallery"
    ])
    
    # ============================================================
    # TAB 1: Upload & Process
    # ============================================================
    with tab1:
        st.header("📤 Upload Videos for Processing")
        
        with st.expander("💡 How It Works", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Intelligent Detection:**
                
                **Step 1: Text Detection (Priority)**
                - OpenAI Vision API for OCR
                - Compares text word-by-word
                - Single word change = new keyframe
                - Perfect for slides, tutorials, code
                
                **Step 2: Visual Detection (Fallback)**
                - Only if no text change detected
                - 30% pixel change threshold
                - Catches visual scene changes
                - Fast numpy-based comparison
                """)
            
            with col2:
                st.markdown("""
                **🧠 AI Enhancement:**
                
                **Semantic Analysis:**
                - People & action detection
                - Scene understanding
                - Technical content extraction
                - Rich metadata generation
                
                **Audio Processing:**
                - Whisper transcription
                - Links audio to visual moments
                - Contextual retrieval
                """)
        
        uploaded_files = st.file_uploader(
            "Choose video files",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            accept_multiple_files=True,
            help="Upload videos for intelligent keyframe extraction"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Videos", type="primary", use_container_width=True):
                for video_file in uploaded_files:
                    st.markdown(f"### 📹 Processing: {video_file.name}")
                    
                    with st.container():
                        keyframes_data, audio_data = process_video(video_file)
                        
                        if keyframes_data:
                            # Statistics
                            frames_with_people = sum(
                                1 for kf in keyframes_data 
                                if kf['semantic_data'].get('has_people')
                            )
                            frames_with_text = sum(
                                1 for kf in keyframes_data 
                                if kf.get('extracted_text')
                            )
                            frames_with_actions = sum(
                                1 for kf in keyframes_data 
                                if kf['semantic_data'].get('has_actions')
                            )
                            
                            text_based = sum(
                                1 for kf in keyframes_data
                                if kf['change_type'] == 'text_change'
                            )
                            visual_based = sum(
                                1 for kf in keyframes_data
                                if kf['change_type'] == 'visual_change'
                            )
                            
                            # Store in session state
                            st.session_state.processed_videos.append({
                                "name": video_file.name,
                                "video_id": keyframes_data[0]["video_id"],
                                "keyframes_count": len(keyframes_data),
                                "text_based_count": text_based,
                                "visual_based_count": visual_based,
                                "frames_with_people": frames_with_people,
                                "frames_with_text": frames_with_text,
                                "frames_with_actions": frames_with_actions,
                                "keyframes_data": keyframes_data,
                                "audio_data": audio_data
                            })
                    
                    st.markdown("---")
                
                # Initialize retriever
                st.session_state.retriever = VideoRAGRetriever(
                    st.session_state.graph_builder
                )
                
                st.success("🎉 All videos processed successfully!")
        
        # Display processed videos
        if st.session_state.processed_videos:
            st.markdown("---")
            st.subheader("📹 Processed Videos")
            
            for idx, video in enumerate(st.session_state.processed_videos, 1):
                with st.expander(f"📹 {idx}. {video['name']}", expanded=False):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("🔑 Keyframes", video['keyframes_count'])
                    with col2:
                        st.metric("📝 Text-based", video['text_based_count'])
                    with col3:
                        st.metric("🎨 Visual-based", video['visual_based_count'])
                    with col4:
                        st.metric("👥 People", video.get('frames_with_people', 0))
                    with col5:
                        audio_segments = len(video.get('audio_data', {}).get('segments', []))
                        st.metric("🎵 Audio", audio_segments)
                    
                    # Detection breakdown
                    if video['keyframes_data']:
                        text_pct = (video['text_based_count'] / video['keyframes_count'] * 100)
                        visual_pct = (video['visual_based_count'] / video['keyframes_count'] * 100)
                        
                        st.progress(text_pct / 100, text=f"📝 Text Detection: {text_pct:.0f}%")
                        st.progress(visual_pct / 100, text=f"🎨 Visual Detection: {visual_pct:.0f}%")
    
    # ============================================================
    # TAB 2: Query & Retrieve
    # ============================================================
    with tab2:
        st.header("🔍 Query Keyframes & Retrieve Moments")
        
        if not st.session_state.retriever:
            st.info("👆 Please process videos first in the 'Upload & Process' tab")
        else:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                query_text = st.text_input(
                    "Enter your search query",
                    placeholder="e.g., 'introduction slide' or 'person typing' or 'code example'",
                    help="Search for specific content, people, actions, or text"
                )
            
            with col2:
                top_k = st.slider("Results", 1, 10, 5)
            
            # Example queries
            with st.expander("💡 Example Queries", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Text Content:**")
                    if st.button("title slide"):
                        query_text = "title slide"
                    if st.button("bullet points"):
                        query_text = "bullet points"
                    if st.button("code example"):
                        query_text = "code example"
                
                with col2:
                    st.markdown("**People & Actions:**")
                    if st.button("person typing"):
                        query_text = "person typing"
                    if st.button("person presenting"):
                        query_text = "person presenting"
                    if st.button("person explaining"):
                        query_text = "person explaining"
                
                with col3:
                    st.markdown("**Visual Elements:**")
                    if st.button("diagram"):
                        query_text = "diagram"
                    if st.button("chart or graph"):
                        query_text = "chart or graph"
                    if st.button("terminal screen"):
                        query_text = "terminal screen"
            
            if st.button("🔍 Search", type="primary", use_container_width=True) and query_text:
                with st.spinner("🔍 Searching through keyframes..."):
                    results = st.session_state.retriever.query(
                        query_text, 
                        top_k=top_k,
                        return_chunks=True
                    )
                
                if results:
                    st.success(f"✨ Found {len(results)} relevant moments")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"🎯 Result {i} - Score: {result['score']:.3f} | "
                            f"Time: {format_timestamp(result['metadata']['timestamp'])}", 
                            expanded=(i == 1)
                        ):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Display keyframe
                                frame_path = result["metadata"]["frame_path"]
                                if Path(frame_path).exists():
                                    st.image(frame_path, use_container_width=True)
                                    
                                    metric_col1, metric_col2 = st.columns(2)
                                    with metric_col1:
                                        st.metric("Similarity", f"{result['similarity']:.3f}")
                                    with metric_col2:
                                        st.metric("Importance", f"{result['importance']:.3f}")
                            
                            with col2:
                                metadata = result["metadata"]
                                
                                # Basic info
                                st.markdown(f"**📹 Video:** {metadata.get('video_name', 'Unknown')}")
                                st.markdown(f"**⏱️ Timestamp:** {format_timestamp(metadata['timestamp'])}")
                                
                                # Detection type badge
                                change_type = metadata.get('change_type', 'unknown')
                                if change_type == 'text_change':
                                    st.markdown('<span class="detection-badge text-change">📝 TEXT CHANGE</span>', 
                                              unsafe_allow_html=True)
                                elif change_type == 'visual_change':
                                    st.markdown('<span class="detection-badge visual-change">🎨 VISUAL CHANGE</span>', 
                                              unsafe_allow_html=True)
                                
                                st.markdown(f"**📝 Description:**")
                                st.write(metadata['main_subject'])
                                
                                # Extracted text from detection phase
                                if metadata.get('extracted_text'):
                                    with st.expander("📝 Extracted Text (OpenAI OCR)", expanded=True):
                                        st.code(metadata['extracted_text'], language=None)
                                
                                # ✅ ADD: Display audio context
                                if metadata.get('audio_context'):
                                    with st.expander("🎤 Audio Transcript", expanded=False):
                                        st.markdown(f"<p style='font-size: 18px; line-height: 1.8;'>{metadata['audio_context']}</p>", unsafe_allow_html=True)
                                
                                # People information
                                if metadata.get('people'):
                                    st.markdown("**👤 People:**")
                                    for j, person in enumerate(metadata['people'], 1):
                                        st.caption(f"• {person.get('description', 'N/A')}")
                                        if person.get('action'):
                                            st.caption(f"  → Action: {person['action']}")
                                
                                # Actions
                                if metadata.get('actions'):
                                    st.markdown("**🎬 Actions:**")
                                    for action in metadata['actions']:
                                        st.caption(f"• {action}")
                                
                                # Video chunk
                                st.info(f"🎞️ **Video Chunk:** "
                                       f"{format_timestamp(result['chunk_start'])} → "
                                       f"{format_timestamp(result['chunk_end'])} "
                                       f"({result['chunk_duration']:.1f}s)")
                
                else:
                    st.warning("😕 No results found. Try a different query.")
    
    # ============================================================
    # TAB 3: Graph Analysis
    # ============================================================
    with tab3:
        st.header("📊 Temporal Semantic Graph Analysis")
        
        if st.session_state.graph_builder:
            if st.session_state.processed_videos:
                st.markdown("### 🎥 Select Video")
                
                video_options = ["🌐 All Videos"] + [
                    f"📹 {v['name']}" for v in st.session_state.processed_videos
                ]
                selected = st.selectbox("Choose a video", video_options)
                
                if selected == "🌐 All Videos":
                    visualize_video_graph(st.session_state.graph_builder, video_id=None)
                else:
                    video_name = selected.replace("📹 ", "")
                    video_id = next(
                        (v["video_id"] for v in st.session_state.processed_videos 
                         if v["name"] == video_name),
                        None
                    )
                    
                    if video_id:
                        visualize_video_graph(st.session_state.graph_builder, video_id=video_id)
            else:
                visualize_video_graph(st.session_state.graph_builder)
        else:
            st.info("📊 Process videos first to build the graph")
    
    # ============================================================
    # TAB 4: Keyframes Gallery
    # ============================================================
    with tab4:
        st.header("🖼️ Keyframes Gallery")
        
        if st.session_state.processed_videos:
            for video in st.session_state.processed_videos:
                st.markdown(f"### 📹 {video['name']}")
                st.caption(
                    f"Keyframes: {video['keyframes_count']} | "
                    f"Text-based: {video['text_based_count']} | "
                    f"Visual-based: {video['visual_based_count']}"
                )
                
                # Display in grid
                cols = st.columns(4)
                for i, kf_data in enumerate(video["keyframes_data"]):
                    col_idx = i % 4
                    
                    with cols[col_idx]:
                        frame_path = kf_data["frame_path"]
                        if Path(frame_path).exists():
                            st.image(frame_path, use_container_width=True)
                            
                            st.caption(f"⏱ {format_timestamp(kf_data['timestamp'])}")
                            
                            # Badge
                            if kf_data['change_type'] == 'text_change':
                                st.caption("📝 Text")
                            elif kf_data['change_type'] == 'visual_change':
                                st.caption("🎨 Visual")
                            else:
                                st.caption("🎬 Initial")
                            
                            # Details popover
                            with st.popover("ℹ️"):
                                st.markdown(f"**Frame #{kf_data['frame_number']}**")
                                st.markdown(f"**Time:** {format_timestamp(kf_data['timestamp'])}")
                                st.markdown(f"**Type:** {kf_data['change_type']}")
                                
                                if kf_data.get('extracted_text'):
                                    st.markdown("**Text:**")
                                    text = kf_data['extracted_text']
                                    st.caption(text[:100] + "..." if len(text) > 100 else text)
                                
                                st.write(kf_data['semantic_data']['main_subject'])
                
                st.markdown("---")
        else:
            st.info("📹 No keyframes to display. Upload videos first!")


if __name__ == "__main__":
    main()