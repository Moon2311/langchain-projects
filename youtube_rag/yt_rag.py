# ========================================
# ENHANCED YOUTUBE VIDEO SUMMARIZER WITH RAG + MULTI-QUERY
# Features: Full Transcript → Multi-Query Retrieval → HuggingFace API
# ========================================

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import re
import requests
import json

# -----------------------------
# CONFIGURATION
# -----------------------------
@st.cache_resource
def load_embeddings():
    """Load embedding model (cached for performance)"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# -----------------------------
# HUGGING FACE API INTEGRATION
# -----------------------------
import requests

import requests

def query_huggingface_api(prompt: str, api_key: str, max_tokens: int = 1024) -> str:
    """
    Query HuggingFace Inference API with Phi-3 Mini (supported in Dec 2025).
    Free tier: 1000 requests/day. Excellent for reasoning, summaries, and instruction tasks.
    Model: microsoft/Phi-3-mini-4k-instruct (3.8B params, 4K context).
    """
    API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"  # Updated to Phi-3 (confirmed supported)
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95,
            "do_sample": True,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "").strip()
        return "No output generated."
    except requests.exceptions.Timeout:
        return "Error: Request timed out. The model might be loading—retry in 20-30 seconds."
    except requests.exceptions.HTTPError as e:
        status_code = getattr(response, 'status_code', 'Unknown')
        if status_code in [410, 404]:
            return "Error: Model endpoint unavailable (410/404). Try microsoft/Phi-3-mini-128k-instruct or run locally with Ollama."
        return f"HTTP Error: {status_code} - {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# TRANSCRIPT FETCHING - FULL VERSION
# -----------------------------
def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

def get_transcript(video_id: str) -> str:
    """
    Fetch the transcript of a YouTube video.
    Uses auto-generated or manual captions if available.
    """
    try:
        # Fetch the transcript (your working method)
        transcript_list = YouTubeTranscriptApi()
        data = transcript_list.fetch(video_id)
        print("Raw transcript data:", data)

        # Flatten into plain text
        full_text =  " ".join(snippet.text for snippet in data.snippets)

        # Clean unwanted parts
        full_text = full_text.replace("[Music]", " ").replace("[Applause]", " ")
        full_text = re.sub(r"\s+", " ", full_text).strip()

        # Metadata (basic because no transcript object exists in your method)
        metadata = {
            "video_id": video_id,
            "language": "en",          # fetch() returns plain dict, so language info not available
            "is_generated": None,       # cannot detect without Transcript object
            "duration": len(data),
            "char_count": len(full_text)
        }

        return full_text, metadata

    except TranscriptsDisabled:
        return "", {"error": "Transcripts are disabled for this video"}

    except NoTranscriptFound:
        return "", {"error": "No English transcript found"}

    except Exception as e:
        return "", {"error": f"Error: {str(e)}"}


# -----------------------------
# VECTOR STORE CREATION
# -----------------------------
def create_vector_store(text: str):
    """Split text into optimized chunks and create FAISS vector store"""
    if not text or len(text) < 100:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for better retrieval
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"chunk_id": i}) 
                 for i, chunk in enumerate(chunks)]

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

# -----------------------------
# MULTI-QUERY RETRIEVAL
# -----------------------------
def generate_multiple_queries(original_query: str, api_key: str) -> list:
    """
    Generate multiple related queries to improve retrieval coverage
    """
    prompt = f"""Generate 3 different versions of the following question to retrieve relevant information from a video transcript. Make them diverse but related.

Original question: {original_query}

Provide only the 3 alternative questions, one per line, without numbering or explanation."""

    response = query_huggingface_api(prompt, api_key, max_tokens=150)
    
    if response and not response.startswith("Error"):
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        queries = [original_query] + queries[:3]  # Include original + 3 alternatives
        return queries
    
    return [original_query]  # Fallback to original query

def multi_query_retrieval(vector_store, queries: list, k: int = 4) -> list:
    """
    Retrieve documents using multiple query variations and deduplicate
    """
    all_docs = []
    seen_content = set()
    
    for query in queries:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        
        for doc in docs:
            content = doc.page_content
            if content not in seen_content:
                all_docs.append(doc)
                seen_content.add(content)
    
    return all_docs[:8]  # Return top 8 unique chunks

# -----------------------------
# RAG GENERATION
# -----------------------------
def generate_answer(vector_store, question: str, api_key: str, use_multi_query: bool = True) -> str:
    """Generate answer using multi-query retrieval and HuggingFace API"""
    
    # Multi-query retrieval
    if use_multi_query:
        with st.spinner("Generating query variations..."):
            queries = generate_multiple_queries(question, api_key)
            st.caption(f"🔍 Searching with {len(queries)} query variations")
        docs = multi_query_retrieval(vector_store, queries)
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(question)
    
    # Combine context
    context = "\n\n".join([f"[Chunk {i+1}]: {doc.page_content}" 
                           for i, doc in enumerate(docs)])
    
    # Create RAG prompt
    prompt = f"""You are an expert assistant analyzing a YouTube video transcript.

Context from video:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- Be detailed and specific
- If the information is not in the context, say "This information is not covered in the video"
- Use natural, conversational language

Answer:"""

    return query_huggingface_api(prompt, api_key, max_tokens=800)

# ========================================
# STREAMLIT UI - MODERN & ENHANCED
# ========================================

st.set_page_config(
    page_title="YouTube RAG Summarizer Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF0000, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .stat-box {
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎬 YouTube RAG Summarizer Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Mistral-7B + Multi-Query Retrieval | Extract insights from any YouTube video</p>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "HuggingFace API Key",
        type="password",
        help="Get your free API key from https://huggingface.co/settings/tokens"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your HuggingFace API key to continue")
        st.markdown("""
        **How to get your API key:**
        1. Go to [HuggingFace](https://huggingface.co/settings/tokens)
        2. Create a new token (Read access)
        3. Copy and paste it above
        
        **Model:** Mistral-7B-Instruct-v0.2
        **Free Tier:** 1000 requests/day
        """)
    
    st.divider()
    
    use_multi_query = st.checkbox(
        "Enable Multi-Query Retrieval",
        value=True,
        help="Generate multiple query variations for better retrieval"
    )
    
    st.divider()
    
    st.markdown("""
    ### 📊 Features
    - ✅ Full transcript extraction
    - ✅ Multi-query retrieval
    - ✅ Mistral-7B AI model
    - ✅ Smart chunking
    - ✅ Context-aware answers
    """)

# Main Content
if not api_key:
    st.info("👈 Enter your HuggingFace API key in the sidebar to get started")
    st.stop()

url = st.text_input(
    "🔗 Enter YouTube URL",
    placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    help="Paste any YouTube video URL with available subtitles"
)

if url:
    video_id = extract_video_id(url)
    
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please check and try again.")
        st.stop()
    
    # Display video
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(url)
    
    # Fetch transcript
    with st.spinner("🔄 Fetching full transcript..."):
        transcript, metadata = get_transcript(video_id)
    
    if not transcript:
        st.error(f"❌ {metadata.get('error', 'Could not fetch transcript')}")
        st.info("💡 Make sure the video has English subtitles/captions enabled")
        st.stop()
    
    # Display metadata
    with col2:
        st.markdown("### 📈 Video Stats")
        st.metric("Transcript Length", f"{metadata['char_count']:,} chars")
        st.metric("Segments", metadata['duration'])
        st.metric("Type", "Auto-generated" if metadata['is_generated'] else "Manual")
    
    st.success(f"✅ Transcript loaded successfully!")
    
    # Create vector store
    with st.spinner("🔧 Building vector database..."):
        vector_store = create_vector_store(transcript)
        if not vector_store:
            st.error("Failed to create vector store")
            st.stop()
    
    st.success("✅ Vector database ready!")
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📝 Generate Summary", "📄 View Transcript"])
    
    with tab1:
        st.subheader("Ask Anything About the Video")
        st.markdown("Use natural language to ask specific questions about the video content.")
        
        question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the main points discussed in this video?",
            key="question_input"
        )
        
        col_a, col_b = st.columns([1, 5])
        with col_a:
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        
        if ask_button and question:
            with st.spinner("🤔 Analyzing video and generating answer..."):
                answer = generate_answer(vector_store, question, api_key, use_multi_query)
            
            st.markdown("### 💡 Answer")
            if answer.startswith("Error"):
                st.error(answer)
            else:
                st.markdown(f"**Q:** {question}")
                st.markdown(answer)
        elif ask_button:
            st.warning("⚠️ Please enter a question first")
    
    with tab2:
        st.subheader("Generate Comprehensive Summary")
        st.markdown("Get an AI-generated summary of the entire video content.")
        
        summary_length = st.select_slider(
            "Summary detail level:",
            options=["Brief", "Moderate", "Detailed"],
            value="Moderate"
        )
        
        if st.button("📝 Generate Summary", type="primary", use_container_width=True):
            summary_prompts = {
                "Brief": "Provide a concise 3-4 sentence summary of the main points.",
                "Moderate": "Provide a comprehensive summary covering all key topics and important details.",
                "Detailed": "Provide an extensive, detailed summary covering all major points, supporting details, and examples."
            }
            
            summary_question = f"{summary_prompts[summary_length]} What is this video about?"
            
            with st.spinner(f"📊 Generating {summary_length.lower()} summary..."):
                summary = generate_answer(vector_store, summary_question, api_key, use_multi_query)
            
            st.markdown(f"### 📋 {summary_length} Summary")
            if summary.startswith("Error"):
                st.error(summary)
            else:
                st.markdown(summary)
                
                # Download button
                st.download_button(
                    label="⬇️ Download Summary",
                    data=summary,
                    file_name=f"youtube_summary_{video_id}.txt",
                    mime="text/plain"
                )
    
    with tab3:
        st.subheader("Raw Transcript")
        st.markdown(f"**Length:** {len(transcript):,} characters | **Words:** ~{len(transcript.split()):,}")
        
        st.text_area(
            "Full Transcript",
            transcript,
            height=400,
            help="Complete transcript extracted from the video"
        )
        
        st.download_button(
            label="⬇️ Download Transcript",
            data=transcript,
            file_name=f"transcript_{video_id}.txt",
            mime="text/plain"
        )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Made with ❤️ using Streamlit | Powered by HuggingFace 🤗 Mistral-7B
</div>
""", unsafe_allow_html=True)