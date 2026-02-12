# app.py - AI Research Literature Assistant (Production Auth)
"""
AI Research Literature Assistant
Production-ready persistent authentication using query parameters
"""

import streamlit as st
import os
from datetime import datetime
import uuid
import hashlib
import logging

# Third-party imports
import chromadb
from sqlalchemy.orm import Session as DBSession

# Local imports
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.preprocess import ResearchPaperChunker
from ingestion.embed_store import store_documents_enhanced, CHROMA_DIR
from rag.pipeline import RAGPipeline
from citation_generator import IEEECitationGenerator
from auth import auth_dialog, logout
from database import Session as ChatSession, Message, SessionLocal, init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="AI Research Literature Assistant",
    page_icon="📚",
    layout="wide"
)

# ==================== CSS STYLING ====================
def load_css():
    """Load custom CSS styles"""
    try:
        with open("static/style.css", "r") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

# Initialize database
init_db()

# Load custom CSS immediately after set_page_config
load_css()

# Utility imports
from utils.document_manager import DocumentManager
from utils.document_selector import DocumentSelector
from utils.metrics_tracker import MetricsTracker, ConfidenceCalculator
from utils.ui_components import SearchModeSelector
from utils.export_tools import ChatExporter
from utils.comparison_tools import DocumentComparator
from utils.auth_helper import init_session_auth, cleanup_expired_sessions

# ==================== CRITICAL: INITIALIZE AUTH FIRST ====================
# This validates session from URL and restores authentication state
init_session_auth()

# Cleanup old sessions periodically (runs once per page load)
cleanup_expired_sessions()

# ==================== SESSION INITIALIZATION ====================
# Initialize authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = set()

if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []

if 'doc_manager' not in st.session_state:
    st.session_state.doc_manager = DocumentManager()

if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "semantic"

if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.7

if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "chat"

if 'sidebar_expanded' not in st.session_state:
    st.session_state.sidebar_expanded = False


# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def get_pipeline():
    """Retrieve or create the RAG pipeline."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            collection = client.get_collection("research_papers")
            if collection.count() == 0:
                return None
        except ValueError:
            return None  # Collection doesn't exist
        
        pipeline = RAGPipeline()
        # pipeline.initialize() # Removed as it doesn't exist on the class
        return pipeline
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        return None

def load_chat_session(session_id):
    """Load conversation history from database"""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).get(session_id)
        if session:
            st.session_state.current_session_id = session.id
            st.session_state.conversation = []
            
            messages = db.query(Message).filter(
                Message.session_id == session.id
            ).order_by(Message.timestamp).all()
            
            for msg in messages:
                st.session_state.conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
    except Exception as e:
        logger.error(f"Error loading chat session: {e}")
        st.error("Failed to load chat session")
    finally:
        db.close()

def save_message(role, content):
    """Save message to database, creating session if needed"""
    if not st.session_state.authenticated:
        return
    
    db = SessionLocal()
    
    try:
        current_sid = st.session_state.get('current_session_id')
        if not current_sid:
            title = content[:30] + "..." if len(content) > 30 else content
            new_sess = ChatSession(
                user_id=st.session_state.user_id,
                title=title
            )
            db.add(new_sess)
            db.commit()
            db.refresh(new_sess)
            current_sid = new_sess.id
            st.session_state.current_session_id = current_sid
        
        msg = Message(
            session_id=current_sid,
            role=role,
            content=str(content)
        )
        db.add(msg)
        
        sess = db.query(ChatSession).get(current_sid)
        if sess:
            sess.last_updated = datetime.utcnow()
        
        db.commit()
    except Exception as e:
        logger.error(f"Error saving message: {e}")
    finally:
        db.close()

def _process_single_file(file_name, file_content, doc_manager):
    """Process a single PDF file (thread-safe, no Streamlit calls)."""
    file_size = len(file_content)
    
    if doc_manager.document_exists(file_name, file_size):
        return None  # Skip already processed
    
    file_hash = hashlib.md5(file_content).hexdigest()[:12]
    
    doc_info = doc_manager.add_document(file_name, file_size, file_hash)
    
    # Extract text from PDF
    text = extract_text_from_pdf(file_content)
    
    # Chunk the document
    chunker = ResearchPaperChunker()
    chunks = chunker.chunk_document({
        'content': text,
        'paper_id': file_hash,
        'title': file_name
    })
    
    doc_manager.update_document(
        file_hash, 
        status='completed',
        num_chunks=len(chunks)
    )
    
    return {
        'file_name': file_name,
        'file_hash': file_hash,
        'chunks': chunks,
    }


@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and cache it."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-mpnet-base-v2")
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

def process_uploaded_files(uploaded_files) -> bool:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    try:
        # Load model early to ensure it's ready
        embedding_model = load_embedding_model()
        if not embedding_model:
            st.error("Failed to load embedding model.")
            return False

        # Pre-read all file contents (must happen in main thread)
        file_data = []
        for f in uploaded_files:
            content = f.read()
            file_data.append((f.name, content))
        
        total = len(file_data)
        progress_bar = st.progress(0, text="Starting parallel processing...")
        status_text = st.empty()
        
        all_chunks = []
        all_chunk_ids = []
        completed = 0
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=min(4, total)) as executor:
            futures = {
                executor.submit(
                    _process_single_file, name, content, st.session_state.doc_manager
                ): name 
                for name, content in file_data
            }
            
            for future in as_completed(futures):
                file_name = futures[future]
                completed += 1
                
                try:
                    result = future.result()
                    if result:
                        for chunk in result['chunks']:
                            all_chunks.append(chunk)
                            all_chunk_ids.append(chunk.chunk_id)
                        st.session_state.uploaded_files.add(result['file_name'])
                        status_text.text(f"✅ {result['file_name']}")
                    else:
                        status_text.text(f"⏭️ {file_name} (already processed)")
                except Exception as e:
                    status_text.text(f"❌ {file_name}: {str(e)}")
                    logger.error(f"Error processing {file_name}: {e}")
                
                progress_bar.progress(completed / total, text=f"Processing {completed}/{total} files...")
        
        if all_chunks:
            progress_bar.progress(0.95, text="Storing embeddings...")
            status_text.text(f"Generating embeddings for {len(all_chunks)} chunks...")
            
            # Use the cached model!
            store_documents_enhanced(all_chunks, all_chunk_ids, embedding_model=embedding_model)
            
            progress_bar.progress(1.0, text="✅ All done!")
            status_text.empty()
            return True
        
        progress_bar.progress(1.0, text="✅ Complete (no new files to process)")
        status_text.empty()
        return False
            
    except Exception as e:
        import traceback
        logger.error(f"Error processing files: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Error processing files: {str(e)}")
        return False

def render_chat_message(qa: dict):
    with st.chat_message(qa["role"], avatar="AI" if qa["role"] == "assistant" else "User"):
        st.markdown(qa["content"])
        
        if st.session_state.show_sources and qa.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(qa["sources"][:3]):
                    st.markdown(f"**Source {i+1}:** {source.get('title', 'Unknown')}")
                    st.markdown(f"- Page {source.get('page', 'N/A')}")
                    st.markdown(f"- Relevance: {source.get('relevance_score', 'N/A'):.2f}")

# ==================== TOP PROFILE BUTTON ====================
def render_top_profile():
    """Render the profile button in the top right corner"""
    if st.session_state.authenticated:
        # Wrap the popover in a container with a key for reliable CSS targeting
        with st.container(key="nav-profile-popover"):
            with st.popover(f"{st.session_state.username}"):
                st.write(f"Logged in: **{st.session_state.username}**")
                if st.button("Log Out", key="logout-btn-top", use_container_width=True):
                    logout()
    else:
        # Wrap the login button in a container with a key for reliable CSS targeting
        with st.container(key="nav-profile-top"):
            if st.button("Login / Sign Up", key="login-btn-trigger"):
                auth_dialog()

render_top_profile()

# ==================== SIDEBAR NAVIGATION ====================

sidebar_width = "250px" if st.session_state.sidebar_expanded else "80px"
sidebar_align = "left" if st.session_state.sidebar_expanded else "center"
btn_padding = "0.5rem 1rem" if st.session_state.sidebar_expanded else "0rem"
btn_align = "left" if st.session_state.sidebar_expanded else "center"
btn_justify = "flex-start" if st.session_state.sidebar_expanded else "center"

st.markdown(f"""
    <style>
    :root {{
        --sidebar-width: {sidebar_width};
        --sidebar-text-align: {sidebar_align};
        --sidebar-btn-padding: {btn_padding};
        --sidebar-btn-content-align: {btn_align};
        --sidebar-btn-justify: {btn_justify};
    }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    toggle_icon = "◀" if st.session_state.sidebar_expanded else "▶"
    
    if st.button(toggle_icon, key="toggle-sidebar", help="Toggle sidebar"):
        st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
        st.rerun()
    
    if st.session_state.sidebar_expanded:
        if st.button("🗨 Chat", key="nav-chat", help="Chat", use_container_width=True):
            st.session_state.current_tab = "chat"
            st.rerun()
        
        if st.button("🗎 Documents", key="nav-docs", help="Documents", use_container_width=True):
            st.session_state.current_tab = "documents"
            st.rerun()
        
        if st.button("⚲ Compare", key="nav-compare", help="Compare", use_container_width=True):
            st.session_state.current_tab = "compare"
            st.rerun()
        
        if st.button("⬇ Export", key="nav-export", help="Export", use_container_width=True):
            st.session_state.current_tab = "export"
            st.rerun()
        
        if st.session_state.authenticated:
            if st.button("➕     New Session", key="nav-new-session", help="Start New Session", use_container_width=True, type="primary"):
                st.session_state.conversation = []
                st.session_state.current_session_id = None
                st.session_state.doc_manager = DocumentManager()
                st.session_state.current_tab = "chat"
                st.rerun()
            
            st.markdown("---")
            st.markdown('<p class="sidebar-section-label">HISTORY</p>', unsafe_allow_html=True)
            
            try:
                db = SessionLocal()
                user_sessions = db.query(ChatSession).filter(
                    ChatSession.user_id == st.session_state.user_id
                ).order_by(ChatSession.last_updated.desc()).limit(10).all()
                
                for s in user_sessions:
                    label = s.title if s.title else "New Chat"
                    if len(label) > 18:
                        label = label[:16] + "..."
                    
                    if st.button(f"🗨 {label}", key=f"hist_{s.id}", use_container_width=True):
                        load_chat_session(s.id)
                        st.session_state.current_tab = "chat"
                        st.rerun()
                db.close()
            except Exception as e:
                logger.error(f"Failed to load history: {e}")

    else:
        if st.button("🗨", key="nav-chat-icon", help="Chat"):
            st.session_state.current_tab = "chat"
            st.rerun()
        
        if st.button("🗎", key="nav-docs-icon", help="Documents"):
            st.session_state.current_tab = "documents"
            st.rerun()
        
        if st.button("⚲", key="nav-compare-icon", help="Compare"):
            st.session_state.current_tab = "compare"
            st.rerun()
        
        if st.button("⬇", key="nav-export-icon", help="Export"):
            st.session_state.current_tab = "export"
            st.rerun()
        
        if st.session_state.authenticated:
            if st.button("➕", key="nav-new-session-icon", help="Start New Session", type="primary"):
                st.session_state.conversation = []
                st.session_state.current_session_id = None
                st.session_state.doc_manager = DocumentManager()
                st.session_state.current_tab = "chat"
                st.rerun()

# ==================== MAIN HEADER ====================
st.markdown("""
    <div class="main-header-container">
        <h1>AI Research Literature Assistant</h1>
        <p>Intelligent document analysis powered by RAG</p>
    </div>
""", unsafe_allow_html=True)

# ==================== CHAT TAB ====================
if st.session_state.current_tab == "chat":
    if st.session_state.doc_manager.get_document_count() == 0:
        if not st.session_state.authenticated:
            with st.container(key="auth-trigger-upload"):
                if st.button("Login\nto Upload Research PDFs", key="auth-btn-trigger", type="primary"):
                    auth_dialog()
        else:
            with st.container(key="authenticated-upload"):
                uploaded_files = st.file_uploader(
                    "Upload Research PDFs",
                    type=["pdf"],
                    accept_multiple_files=True,
                    key="uploader-component"
                )
            
            if uploaded_files:
                # Deduplicate by filename
                unique_files_dict = {f.name: f for f in uploaded_files}
                unique_files = list(unique_files_dict.values())
                
                with st.container(key="selected-files-list-section"):
                    st.markdown("---")
                    st.markdown("### Selected Documents")
                    if len(unique_files) < len(uploaded_files):
                        dup_count = len(uploaded_files) - len(unique_files)
                        # Only show toast once per duplicate detection
                        current_dup_key = f"dup_{len(uploaded_files)}_{len(unique_files)}"
                        if st.session_state.get("last_dup_key") != current_dup_key:
                            st.session_state.last_dup_key = current_dup_key
                            st.toast(f"{dup_count} duplicate file(s) removed from selection.", icon="⚠️")
                        
                    for file in unique_files:
                        st.markdown(f"**{file.name}** ({ (len(file.read())/1024/1024):.2f} MB)")
                        file.seek(0)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("Process Documents", key="process-docs-btn"):
                        with st.spinner("Processing documents..."):
                            if process_uploaded_files(unique_files):
                                st.success("Documents processed successfully!")
                                st.rerun()
    else:
        with st.expander("Search Settings"):
            search_mode, alpha = SearchModeSelector.show()
            st.session_state.search_mode = search_mode
            st.session_state.alpha = alpha
            show_sources = st.checkbox("Show source citations", value=st.session_state.show_sources)
            st.session_state.show_sources = show_sources

        for qa in st.session_state.conversation:
            render_chat_message(qa)

        user_query = st.chat_input("Ask a question about your documents...")
        
        if user_query:
            pipeline = get_pipeline()
            
            if pipeline:
                with st.spinner("Thinking..."):
                    result = pipeline.query(
                        user_query,
                        search_mode=st.session_state.search_mode,
                        alpha=st.session_state.alpha
                    )
                
                st.session_state.conversation.append({"role": "user", "content": user_query})
                if st.session_state.authenticated:
                    save_message("user", user_query)
                
                response_content = result.get('response', '')
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response_content,
                    "sources": result.get('sources', [])
                })
                if st.session_state.authenticated:
                    save_message("assistant", response_content)
                
                st.rerun()
            else:
                st.error("Pipeline not initialized. Please upload documents first.")

# ==================== DOCUMENTS TAB ====================
elif st.session_state.current_tab == "documents":
    st.subheader("Document Library")

    if st.session_state.doc_manager.get_document_count() > 0:
        docs = st.session_state.doc_manager.get_all_documents()
        
        for doc in docs:
            with st.expander(f"{doc.get('title', 'Unknown')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Authors:**", ', '.join(doc.get('authors', [])) if doc.get('authors') else 'N/A')
                    st.write("**Uploaded:**", doc.get('upload_time'))
                with col2:
                    st.write("**Chunks:**", doc.get('chunk_count'))
                    st.write("**File:**", doc.get('file_name'))
    else:
        st.info("No documents uploaded yet. Go to Chat tab to upload PDFs.")

# ==================== COMPARE TAB ====================
elif st.session_state.current_tab == "compare":
    st.subheader("Compare Research Papers")

    if st.session_state.doc_manager.get_document_count() > 1:
        doc_names = st.session_state.doc_manager.get_document_names()
        
        selected_docs = st.multiselect("Select papers to compare:", doc_names)
        question = st.text_input("What would you like to compare?")
        
        if st.button("Compare", type="primary"):
            if len(selected_docs) < 2:
                st.warning("Please select at least 2 documents")
            elif not question:
                st.warning("Please enter a comparison question")
            else:
                st.info("Comparison engine will analyze the selected papers...")
    else:
        st.info("Upload at least 2 documents to use the comparison feature")

# ==================== EXPORT TAB ====================
elif st.session_state.current_tab == "export":
    st.subheader("Export Chat History")

    if st.session_state.conversation:
        exporter = ChatExporter()
        
        format_type = st.selectbox("Select export format:", ["JSON", "CSV", "TXT"])
        
        if st.button("Export Chat", type="primary"):
            try:
                path = exporter.export_chat(
                    st.session_state.conversation,
                    format_type.lower(),
                    st.session_state.session_id
                )
                
                st.success("Export ready!")
                
                with open(path, 'r') as f:
                    st.download_button(
                        "Download File",
                        f.read(),
                        file_name=f"chat_export_{st.session_state.session_id}.{format_type.lower()}",
                        mime=f"text/{format_type.lower()}"
                    )
            except Exception as e:
                st.error(f"Export failed: {e}")
    else:
        st.info("No chat history to export. Start a conversation first!")

# ==================== PIPELINE ====================