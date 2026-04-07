# Video RAG (Retrieval-Augmented Generation)
<p align="center">
  <em>Transforming videos from unsearchable "black boxes" into interactive, hyper-specific knowledge bases.</em>
</p>
## Overview
Video RAG is an intelligent semantic search system that allows you to "chat" with your video library. While standard Document RAG works great for PDFs and text, video contains rich, unstructured data across both audio and visual dimensions. 
Instead of wasting time scrubbing through timelines or relying on poor chapter markers, you can ask complex questions about your video content. The system doesn't just give you a text answer—it pinpoints the **exact timestamps and video snippets** where the information appears.
## 🚀 Key Features
- **Multimodal Understanding:** Understands both spoken audio (via Whisper) and on-screen visuals (via Vision models).
- **Pinpoint Video Retrieval:** Instantly returns precision 5-second video chunks containing your answer.
- **Accurately Indexes Silent Video:** Capable of searching silent screen recordings or UI demos by reading on-screen text and visual context.
- **High-Precision Graph Search:** Uses advanced graph algorithms combined with vector search to understand the chronological and semantic flow of the video.
## 🧠 Architecture (The 5-Step Pipeline)
Video RAG solves the multimodal search problem using a robust, automated pipeline under the hood:
1. **Intelligent Keyframe Extraction** 
   Analyzes video using a text-first, pixel-differencing strategy to detect significant visual changes (like slide transitions). This captures only crucial frames without generating thousands of useless duplicates.
2. **Audio Transcription** 
   Extracts audio tracks and passes them through OpenAI's **Whisper** model to generate highly accurate transcripts mapped to precise timestamps.
3. **Semantic Visual Analysis** 
   Passes the extracted keyframes to **GPT-4o Vision**, extracting rich metadata—identifying scene types, reading on-screen text, recognizing people, and describing actions.
4. **Multimodal Embedding & Storage**
   Merges visual descriptions with corresponding audio transcripts to create a unified context. This context is converted into high-dimensional vectors using **`text-embedding-3-large`** and stored in a **ChromaDB** vector database.
5. **Graph-Based Retrieval**
   Constructs a directed **NetworkX Graph** that connects frames chronologically and semantically. By applying **PageRank** combined with vector similarity, the system retrieves the most relevant video chunks with incredible accuracy.
## 🎯 Primary Use Cases
* **Online Education & Lectures:** Ask a recorded lecture to *"explain the architecture diagram"* and jump straight to the exact explanation.
* **Corporate Meetings & Podcasts:** Find exactly when a specific decision was made without rewatching the entire hour-long recording.
* **Technical Tutorials:** Search for when a specific piece of code or terminal command was shown on-screen, even if the speaker didn't read it out loud.
* **Agentic Workflows:** Act as a visual memory layer for AI agents to process and recall video intelligence as easily as standard text.
## 📁 Project Structure
* `backend/` - Core processing pipeline, routing, vector database logic, and API endpoints.
* `frontend/` - UI for chatting, video playback, and dynamic timestamp mapping.
* `data/` - Vector database storage and processed metadata.
* `demo_videos/` - Sample videos for indexing and testing.
## 🛠️ Getting Started
*(Add your specific setup and installation instructions here, e.g., Docker commands, environment variables needed, and startup scripts).*
1. Clone the repository.
2. Check `.env.example` and set up your `.env` file with required API keys (OpenAI, etc.).
3. Start the system (using Docker or local Python environment).
