# ========================================
# ADVANCED YOUTUBE RAG SUMMARIZER PRO
# Features: Domain Routing, Hybrid MMR+Ranking, Context Compression, Citations
# ========================================

import streamlit as st
import re
import requests
from typing import List, Dict, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from collections import Counter
import numpy as np

# -----------------------------
# EMBEDDINGS (cached)
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# -----------------------------
# LOCAL LLM: Phi-3 via Ollama
# -----------------------------
def query_phi3(prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
    """Query Phi-3 via Ollama with configurable parameters"""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": max_tokens,
            "repeat_penalty": 1.1  # Reduce repetition
        }
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama not running. Run `ollama run phi3` in terminal."
    except Exception as e:
        return f"LLM Error: {str(e)}"

# -----------------------------
# EXTRACT VIDEO ID
# -----------------------------
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

# -----------------------------
# TRANSCRIPT METHOD (PRESERVED - NO CHANGES)
# -----------------------------
def get_transcript(video_id: str):
    """
    Fetch the transcript using your preferred method (fetch + snippets)
    """
    try:
        transcript_list = YouTubeTranscriptApi()
        data = transcript_list.fetch(video_id)

        # Flatten into plain text
        full_text = " ".join(snippet.text for snippet in data.snippets)

        # Clean up
        full_text = full_text.replace("[Music]", " ").replace("[Applause]", " ")
        full_text = re.sub(r"\s+", " ", full_text).strip()

        metadata = {
            "video_id": video_id,
            "language": "en",
            "is_generated": None,
            "duration": len(data.snippets),
            "char_count": len(full_text)
        }

        return full_text, metadata

    except TranscriptsDisabled:
        return "", {"error": "Transcripts are disabled for this video"}
    except NoTranscriptFound:
        return "", {"error": "No transcript found (try videos with English captions)"}
    except Exception as e:
        return "", {"error": f"Transcript error: {str(e)}"}

# -----------------------------
# DOMAIN-AWARE ROUTING
# -----------------------------
class DomainRouter:
    """Routes queries to appropriate retrieval strategies based on domain"""
    
    DOMAINS = {
        "technical": ["how", "what is", "explain", "algorithm", "process", "system", "architecture"],
        "factual": ["who", "when", "where", "date", "name", "definition"],
        "analytical": ["why", "compare", "difference", "advantage", "disadvantage", "pros", "cons"],
        "summarization": ["summary", "summarize", "overview", "main points", "key takeaways"],
        "procedural": ["steps", "how to", "tutorial", "guide", "instructions"]
    }
    
    @staticmethod
    def detect_domain(query: str) -> str:
        """Detect query domain for optimal retrieval"""
        query_lower = query.lower()
        scores = {}
        
        for domain, keywords in DomainRouter.DOMAINS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[domain] = score
        
        detected = max(scores, key=scores.get)
        return detected if scores[detected] > 0 else "general"
    
    @staticmethod
    def get_retrieval_params(domain: str) -> Dict:
        """Get optimal retrieval parameters per domain"""
        params = {
            "technical": {"k": 8, "fetch_k": 20, "lambda_mult": 0.5},
            "factual": {"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
            "analytical": {"k": 10, "fetch_k": 25, "lambda_mult": 0.4},
            "summarization": {"k": 12, "fetch_k": 30, "lambda_mult": 0.3},
            "procedural": {"k": 8, "fetch_k": 20, "lambda_mult": 0.5},
            "general": {"k": 8, "fetch_k": 20, "lambda_mult": 0.5}
        }
        return params.get(domain, params["general"])

# -----------------------------
# VECTOR STORE WITH METADATA
# -----------------------------
def create_vector_store(text: str):
    """Create vector store with rich metadata for ranking"""
    if len(text) < 100:
        return None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # Add metadata for each chunk
    docs = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "chunk_id": i,
            "position": i / len(chunks),  # Relative position in video
            "length": len(chunk),
            "word_count": len(chunk.split())
        }
        docs.append(Document(page_content=chunk, metadata=metadata))
    
    return FAISS.from_documents(docs, embeddings)

# -----------------------------
# MULTI-QUERY GENERATION WITH DOMAIN AWARENESS
# -----------------------------
def generate_multiple_queries(question: str, domain: str) -> List[str]:
    """Generate domain-aware query variations"""
    domain_prompts = {
        "technical": "technical variations focusing on mechanisms and implementations",
        "factual": "factual variations asking for specific information",
        "analytical": "analytical variations exploring comparisons and reasoning",
        "summarization": "comprehensive variations for summarization",
        "procedural": "step-by-step variations for instructions"
    }
    
    domain_hint = domain_prompts.get(domain, "different variations")
    
    prompt = f"""Generate 3 {domain_hint} of this question for searching a video transcript:

Original: {question}

Requirements:
- Make them diverse but related
- Keep them concise (one sentence each)
- Number them 1, 2, 3

Variations:"""

    resp = query_phi3(prompt, max_tokens=200, temperature=0.5)
    lines = [l.strip() for l in resp.split("\n") if l.strip()]
    
    # Extract numbered items or just use lines
    variations = []
    for line in lines:
        cleaned = re.sub(r'^[\d\.\)\-]+\s*', '', line)
        if len(cleaned) > 10 and len(cleaned) < 200:
            variations.append(cleaned)
    
    return [question] + variations[:3]

# -----------------------------
# HYBRID RETRIEVAL: MMR + BM25-style Ranking
# -----------------------------
def hybrid_mmr_retrieval(vector_store, queries: List[str], params: Dict) -> List[Document]:
    """
    Hybrid retrieval combining:
    1. MMR (Maximal Marginal Relevance) for diversity
    2. BM25-style keyword matching for precision
    3. Cross-query ranking for robustness
    """
    k = params["k"]
    fetch_k = params["fetch_k"]
    lambda_mult = params["lambda_mult"]
    
    all_docs_with_scores = []
    
    for query_idx, query in enumerate(queries):
        # MMR retrieval for diversity
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult
            }
        )
        docs = retriever.invoke(query)
        
        # Score each document
        for doc in docs:
            # Keyword overlap score (BM25-style)
            query_words = set(query.lower().split())
            doc_words = set(doc.page_content.lower().split())
            keyword_score = len(query_words & doc_words) / len(query_words) if query_words else 0
            
            # Position bonus (earlier chunks slightly preferred)
            position_score = 1.0 - (doc.metadata.get("position", 0.5) * 0.2)
            
            # Length normalization (prefer moderate length)
            length = doc.metadata.get("word_count", 100)
            length_score = 1.0 if 50 <= length <= 200 else 0.8
            
            # Combined score
            combined_score = (
                0.5 * (1.0 / (query_idx + 1)) +  # Query rank bonus
                0.3 * keyword_score +
                0.1 * position_score +
                0.1 * length_score
            )
            
            all_docs_with_scores.append((doc, combined_score))
    
    # Deduplicate and rank
    seen = set()
    ranked_docs = []
    for doc, score in sorted(all_docs_with_scores, key=lambda x: x[1], reverse=True):
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            ranked_docs.append(doc)
    
    return ranked_docs[:k]

# -----------------------------
# POST-RETRIEVAL: CONTEXT COMPRESSION
# -----------------------------
def compress_context(docs: List[Document], query: str, max_chunks: int = 6) -> List[Document]:
    """
    Compress retrieved context by:
    1. Removing redundant information
    2. Extracting most relevant sentences
    3. Limiting total token count
    """
    if len(docs) <= max_chunks:
        return docs
    
    # Calculate relevance scores
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc in docs:
        sentences = re.split(r'[.!?]+', doc.page_content)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
        if relevant_sentences:
            # Keep top sentences from this chunk
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            compressed = ". ".join([s[0] for s in relevant_sentences[:3]])
            
            compressed_doc = Document(
                page_content=compressed,
                metadata=doc.metadata
            )
            scored_docs.append((compressed_doc, sum(s[1] for s in relevant_sentences)))
    
    # Return top compressed chunks
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:max_chunks]]

# -----------------------------
# CONTEXT WINDOW OPTIMIZATION
# -----------------------------
def optimize_context_window(docs: List[Document], max_chars: int = 3000) -> str:
    """Trim context to fit within optimal window while preserving coherence"""
    context_parts = []
    current_chars = 0
    
    for i, doc in enumerate(docs):
        chunk_text = f"[Source {i+1}]: {doc.page_content}"
        chunk_chars = len(chunk_text)
        
        if current_chars + chunk_chars > max_chars:
            # Trim last chunk if needed
            remaining = max_chars - current_chars
            if remaining > 100:
                trimmed = chunk_text[:remaining] + "..."
                context_parts.append(trimmed)
            break
        
        context_parts.append(chunk_text)
        current_chars += chunk_chars
    
    return "\n\n".join(context_parts)

# -----------------------------
# HALLUCINATION PREVENTION
# -----------------------------
def create_hallucination_prevention_prompt(context: str, question: str) -> str:
    """Create prompt with strong hallucination prevention"""
    return f"""You are a precise video transcript analyzer. Follow these rules STRICTLY:

1. ONLY use information from the provided context
2. If information is NOT in the context, say "This is not mentioned in the video"
3. DO NOT infer, assume, or add external knowledge
4. Cite sources using [Source N] notation
5. If uncertain, express it clearly

Context from video:
{context}

Question: {question}

CRITICAL: Answer ONLY based on the context above. Include [Source N] citations.

Answer:"""

# -----------------------------
# GENERATE ANSWER WITH CITATIONS
# -----------------------------
def generate_answer_with_citations(
    vector_store,
    question: str,
    use_advanced: bool = True
) -> Tuple[str, Dict]:
    """
    Advanced answer generation with:
    - Domain routing
    - Hybrid retrieval
    - Context compression
    - Hallucination prevention
    - Citation tracking
    """
    metadata = {"domain": "general", "queries": 1, "chunks_retrieved": 0, "chunks_used": 0}
    
    try:
        # 1. DOMAIN-AWARE ROUTING
        domain = DomainRouter.detect_domain(question)
        params = DomainRouter.get_retrieval_params(domain)
        metadata["domain"] = domain
        
        if use_advanced:
            # 2. MULTI-QUERY GENERATION
            queries = generate_multiple_queries(question, domain)
            metadata["queries"] = len(queries)
            
            # 3. HYBRID MMR + RANKING RETRIEVAL
            docs = hybrid_mmr_retrieval(vector_store, queries, params)
        else:
            # Simple retrieval
            queries = [question]
            retriever = vector_store.as_retriever(search_kwargs={"k": 8})
            docs = retriever.invoke(question)
        
        metadata["chunks_retrieved"] = len(docs)
        
        # 4. POST-RETRIEVAL: CONTEXT COMPRESSION
        compressed_docs = compress_context(docs, question, max_chunks=6)
        metadata["chunks_used"] = len(compressed_docs)
        
        # 5. CONTEXT WINDOW OPTIMIZATION
        context = optimize_context_window(compressed_docs, max_chars=3000)
        
        # 6. HALLUCINATION PREVENTION PROMPT
        prompt = create_hallucination_prevention_prompt(context, question)
        
        # 7. GENERATION WITH CITATIONS
        answer = query_phi3(prompt, max_tokens=1000, temperature=0.2)
        
        # 8. POST-PROCESS: Verify citations exist
        if "[Source" not in answer and not answer.startswith("ERROR"):
            # Add citation reminder
            answer += "\n\n*Note: Answer based on retrieved video segments.*"
        
        return answer, metadata
        
    except Exception as e:
        return f"Error generating answer: {str(e)}", metadata

# ========================================
# STREAMLIT UI - ENHANCED
# ========================================

st.set_page_config(
    page_title="YouTube RAG Pro - Advanced",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF3366;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .info-badge {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎬 YouTube RAG Summarizer Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Edition: Hybrid Retrieval • Context Compression • Citation-Based Answers</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    
    st.subheader("🔧 System Status")
    st.success("✓ Phi-3 via Ollama")
    st.code("ollama run phi3", language="bash")
    
    st.divider()
    
    st.subheader("🧠 Retrieval Options")
    use_advanced = st.checkbox(
        "Advanced Retrieval Pipeline",
        value=True,
        help="Enables domain routing, hybrid MMR, and context compression"
    )
    
    if use_advanced:
        st.info("""
        **Enabled Features:**
        - 🎯 Domain-aware routing
        - 🔄 Multi-query generation
        - 🎨 Hybrid MMR + Ranking
        - 📦 Context compression
        - ✂️ Context window optimization
        - 🛡️ Hallucination prevention
        - 📝 Citation tracking
        """)
    else:
        st.warning("Using basic retrieval mode")
    
    st.divider()
    
    st.subheader("📊 Features")
    st.markdown("""
    - ✅ Your transcript method (preserved)
    - ✅ Domain-aware routing
    - ✅ Hybrid MMR retrieval
    - ✅ Context compression
    - ✅ Citation-based answers
    - ✅ 100% local & private
    """)

# Main Content
url = st.text_input(
    "🔗 Enter YouTube URL",
    placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
)

if url:
    video_id = extract_video_id(url)
    
    if not video_id:
        st.error("❌ Invalid YouTube URL")
        st.stop()
    
    # Video Preview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(url)
    
    # Fetch Transcript
    with st.spinner("🔄 Fetching transcript (your custom method)..."):
        transcript, meta = get_transcript(video_id)
    
    if not transcript:
        st.error(f"❌ {meta.get('error')}")
        st.stop()
    
    # Display Stats
    with col2:
        st.markdown("### 📈 Video Stats")
        st.metric("Characters", f"{meta['char_count']:,}")
        st.metric("Segments", meta['duration'])
        st.metric("Status", "✓ Ready")
    
    # Build Vector Store
    with st.spinner("🔧 Building vector database..."):
        vector_store = create_vector_store(transcript)
        if not vector_store:
            st.error("Failed to create vector store")
            st.stop()
    
    st.success("✅ System ready! Ask questions or generate summary.")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Ask Questions",
        "📝 Generate Summary",
        "📄 View Transcript",
        "📊 System Info"
    ])
    
    with tab1:
        st.subheader("Ask Questions with Advanced Retrieval")
        
        question = st.text_input(
            "Your question:",
            placeholder="What are the main points discussed?",
            key="question_input"
        )
        
        col_a, col_b = st.columns([1, 4])
        with col_a:
            ask_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
        
        if ask_button and question:
            with st.spinner("🤔 Analyzing with advanced pipeline..."):
                answer, metadata = generate_answer_with_citations(
                    vector_store,
                    question,
                    use_advanced
                )
            
            # Display metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Domain", metadata["domain"].title())
            with col2:
                st.metric("Queries", metadata["queries"])
            with col3:
                st.metric("Retrieved", metadata["chunks_retrieved"])
            with col4:
                st.metric("Used", metadata["chunks_used"])
            
            st.markdown("### 💡 Answer")
            if answer.startswith("ERROR"):
                st.error(answer)
            else:
                st.markdown(answer)
                
                # Download option
                st.download_button(
                    "⬇️ Download Answer",
                    answer,
                    f"answer_{video_id}.txt",
                    mime="text/plain"
                )
        
        elif ask_button:
            st.warning("⚠️ Please enter a question")
    
    with tab2:
        st.subheader("Generate Comprehensive Summary")
        
        level = st.select_slider(
            "Summary detail level:",
            options=["Brief", "Moderate", "Detailed"],
            value="Moderate"
        )
        
        summary_prompts = {
            "Brief": "Provide a concise 3-4 sentence summary of the video's main message.",
            "Moderate": "Summarize all key topics and important points discussed in the video.",
            "Detailed": "Provide an extensive summary covering all major topics, supporting details, examples, and conclusions."
        }
        
        if st.button("📝 Generate Summary", type="primary", use_container_width=True):
            summary_question = f"{summary_prompts[level]} Include citations."
            
            with st.spinner(f"📊 Generating {level.lower()} summary..."):
                summary, metadata = generate_answer_with_citations(
                    vector_store,
                    summary_question,
                    use_advanced
                )
            
            st.markdown(f"### 📋 {level} Summary")
            if summary.startswith("ERROR"):
                st.error(summary)
            else:
                st.markdown(summary)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"Domain: {metadata['domain']} | Sources: {metadata['chunks_used']}")
                with col2:
                    st.download_button(
                        "⬇️ Download",
                        summary,
                        f"summary_{video_id}.txt",
                        use_container_width=True
                    )
    
    with tab3:
        st.subheader("Full Transcript")
        st.markdown(f"**Length:** {len(transcript):,} characters | **Words:** ~{len(transcript.split()):,}")
        
        st.text_area(
            "Complete Transcript",
            transcript,
            height=500,
            help="Your custom transcript extraction method"
        )
        
        st.download_button(
            "⬇️ Download Transcript",
            transcript,
            f"transcript_{video_id}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with tab4:
        st.subheader("📊 System Information")
        
        st.markdown("### Pipeline Components")
        
        components = {
            "Transcript Extraction": "✓ Your custom method (preserved)",
            "Text Splitting": "✓ RecursiveCharacterTextSplitter (800 chars)",
            "Embeddings": "✓ sentence-transformers/all-MiniLM-L6-v2",
            "Vector Store": "✓ FAISS (in-memory)",
            "Domain Router": "✓ 5 domain categories" if use_advanced else "✗ Disabled",
            "Multi-Query": "✓ 4 query variations" if use_advanced else "✗ Disabled",
            "Retrieval": "✓ Hybrid MMR + Ranking" if use_advanced else "✓ Basic similarity",
            "Compression": "✓ Context compression active" if use_advanced else "✗ Disabled",
            "LLM": "✓ Phi-3 via Ollama (local)",
            "Hallucination Guard": "✓ Enabled",
            "Citations": "✓ Source tracking"
        }
        
        for component, status in components.items():
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(component)
            with col2:
                if "✓" in status:
                    st.success(status)
                else:
                    st.warning(status)
        
        st.divider()
        
        st.markdown("### Domain Categories")
        st.json({
            "Technical": ["how", "explain", "algorithm", "process"],
            "Factual": ["who", "when", "where", "definition"],
            "Analytical": ["why", "compare", "difference"],
            "Summarization": ["summary", "overview", "key points"],
            "Procedural": ["steps", "how to", "tutorial"]
        })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>Advanced YouTube RAG Pro</strong> | 100% Local & Private<br>
    Powered by Phi-3 + Ollama | Your transcript method preserved ❤️
</div>
""", unsafe_allow_html=True)