# app.py - AI Research Literature Assistant (Production Auth)
"""
AI Research Literature Assistant
Production-ready persistent authentication using query parameters
"""

import streamlit as st
import os
from datetime import datetime
import uuid
import logging
import shutil

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

# Research metadata DB
from db.database import SessionLocal as ResearchSessionLocal, init_db as init_research_db
from db.models import ResearchPaperMetadata

# Chat/auth DB
from database import Session as ChatSession, Message, SessionLocal as ChatSessionLocal, init_db as init_chat_db

from ingestion.embed_store import get_shared_client
from ingestion.ingest_all import ingest_uploaded_files, IngestResult, CHROMA_DIR
from utils.ai_comparison_engine import generate_comparison_insight
from rag.pipeline import RAGPipeline
from auth import auth_dialog, logout

from dotenv import load_dotenv
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="AI Research Literature Assistant",
    page_icon="",
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
init_research_db()
init_chat_db()


# Load custom CSS immediately after set_page_config
load_css()

# Utility imports
from utils.document_manager import DocumentManager
from utils.document_selector import DocumentSelector
from utils.metrics_tracker import MetricsTracker, ConfidenceCalculator
from utils.ui_components import SearchModeSelector
from utils.export_tools import ChatExporter
# from utils.comparison_tools import DocumentComparator
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

if 'documents_ready' not in st.session_state:
    st.session_state.documents_ready = False


# ==================== HELPER FUNCTIONS ====================
@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    try:
        if st.session_state.documents_ready:
            client = get_shared_client()
            try:
                collection = client.get_collection("research_papers")
                if collection.count() > 0:
                    return RAGPipeline()
            except ValueError:
                return None
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}")
        return None


def load_chat_session(session_id):
    """Load conversation history from database"""
    db = ChatSessionLocal()

    try:
        session = db.query(ChatSession).get(session_id)
        if session:
            st.session_state.current_session_id = session.id
            st.session_state.conversation = []
            
            messages = db.query(Message).filter(
                Message.session_id == session.id
            ).order_by(Message.timestamp).all()
            
            for msg in messages:
                entry = {
                    "role": msg.role,
                    "content": msg.content
                }
                # Try to restore sources from JSON if stored
                if msg.sources_json:
                    try:
                        import json
                        entry["sources"] = json.loads(msg.sources_json)
                    except Exception:
                        pass
                st.session_state.conversation.append(entry)
    except Exception as e:
        logger.error(f"Error loading chat session: {e}")
        st.error("Failed to load chat session")
    finally:
        db.close()

def enforce_max_sessions(user_id, max_sessions=20):
    """Enforce max N chat sessions per user, deleting oldest beyond the limit."""
    db = ChatSessionLocal()

    try:
        all_sessions = db.query(ChatSession).filter(
            ChatSession.user_id == user_id
        ).order_by(ChatSession.last_updated.desc()).all()
        
        if len(all_sessions) > max_sessions:
            sessions_to_delete = all_sessions[max_sessions:]
            for s in sessions_to_delete:
                db.delete(s)
            db.commit()
            logger.info(f"Pruned {len(sessions_to_delete)} old sessions for user {user_id}")
    except Exception as e:
        logger.error(f"Error enforcing max sessions: {e}")
    finally:
        db.close()

def generate_conversation_title(first_message):
    """Generate a meaningful title from the first message"""
    import re
    
    # Common question words to remove from beginning
    question_starters = [
        'what is', 'what are', 'what does', 'what can', 'what would',
        'how does', 'how can', 'how would', 'how is', 'how are',
        'why is', 'why are', 'why does', 'why do',
        'when does', 'when can', 'when will',
        'where is', 'where are', 'where can',
        'who is', 'who are', 'who can',
        'which is', 'which are', 'which can',
        'can you', 'could you', 'would you', 'should you',
        'is there', 'are there', 'do we', 'does it'
    ]
    
    content = first_message.lower().strip()
    
    # Remove question starters
    for starter in question_starters:
        if content.startswith(starter):
            content = content[len(starter):].strip()
            break
    
    # Remove question marks and other punctuation at the end
    content = re.sub(r'[?!.]+$', '', content)
    
    # Get first meaningful phrase (up to 40 chars)
    if len(content) > 40:
        # Try to break at word boundary
        for i in range(40, 30, -1):
            if content[i] == ' ':
                content = content[:i]
                break
        else:
            content = content[:40]
    
    # Capitalize properly
    words = content.split()
    if words:
        # Capitalize first word
        words[0] = words[0].capitalize()
        # Keep proper nouns (words that start with capital in original)
        original_words = first_message.split()
        for i, word in enumerate(words):
            if i < len(original_words) and original_words[i][0].isupper():
                words[i] = word.capitalize()
        content = ' '.join(words)
    
    # Add ellipsis if truncated
    if len(first_message.strip()) > len(content):
        content += "..."
    
    # Fallback if title is too short after cleaning
    if len(content.strip()) < 5:
        content = first_message[:30] + "..." if len(first_message) > 30 else first_message
    
    return content[:50].strip()  # Ensure max 50 characters


def save_message(role, content, sources=None):
    """Save message to database, creating session if needed"""
    if not st.session_state.authenticated:
        return
    
    db = ChatSessionLocal()

    
    try:
        current_sid = st.session_state.get('current_session_id')
        if not current_sid:
            # Generate a meaningful title from the first message
            title = generate_conversation_title(content)
            new_sess = ChatSession(
                user_id=st.session_state.user_id,
                title=title
            )
            db.add(new_sess)
            db.commit()
            db.refresh(new_sess)
            current_sid = new_sess.id
            st.session_state.current_session_id = current_sid
            # Enforce max 20 sessions per user
            enforce_max_sessions(st.session_state.user_id, max_sessions=20)
        
        # Serialize sources to JSON if provided
        sources_json_str = None
        if sources:
            try:
                import json
                sources_json_str = json.dumps(sources, default=str)
            except Exception:
                pass
        
        msg = Message(
            session_id=current_sid,
            role=role,
            content=str(content),
            sources_json=sources_json_str
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

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load sentence-transformer once per process, reused by ingestion."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"[APP] Embedding model load failed: {e}")
        return None


def _process_and_ingest(file_data: list) -> bool:
    """
    Hand uploaded file bytes to the ingestion layer.
    app.py does NOT touch pdf_loader / preprocess / embed_store directly.
    """
    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    def on_progress(fraction: float, msg: str):
        progress_bar.progress(fraction, text=f"{msg}")
        status_text.text(msg)

    # Important: ensure any cached Chroma/RAG resources are released before writing.
    # This prevents stale sqlite handles after CHROMA_DIR was cleared elsewhere.
    try:
        st.cache_resource.clear()
    except Exception:
        pass

    result: IngestResult = ingest_uploaded_files(
        files=file_data,
        doc_manager=st.session_state.doc_manager,
        embedding_model=load_embedding_model(),
        chunk_size=1800,
        overlap=80,
        max_workers=8,
        max_images=10 if st.session_state.get("process_images", False) else 0,
        progress_cb=on_progress,
    )

    progress_bar.progress(1.0, text="Done")
    status_text.empty()

    if result.failed:
        st.warning(f"Failed to process: {', '.join(result.failed)}")
    if result.skipped:
        st.info(f"Already processed (skipped): {', '.join(result.skipped)}")

    return result.success

def render_chat_message(qa: dict):
    with st.chat_message(qa["role"]):
        st.markdown(qa["content"])
        
        # Display any images referenced in the response
        if qa.get("images"):
            for img_info in qa["images"]:
                img_path = img_info.get("path", "")
                caption = img_info.get("caption", "Figure")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=caption, use_container_width=True)
        
        if st.session_state.show_sources and qa.get("sources"):
            with st.expander("Sources & Figures"):
                for i, source in enumerate(qa["sources"][:5]):
                    meta = source.get("metadata", {})
                    chunk_type = meta.get("chunk_type", "content")
                    
                    # Try to find a good title/ID
                    title = meta.get("title") or meta.get("paper_id") or source.get("id", "Unknown")
                    
                    # Display image if this source is an image chunk
                    if chunk_type == "image_description":
                        image_path = meta.get("image_path", "")
                        page = meta.get("page_start", "?")
                        st.markdown(f"**Figure (Page {page}):**")
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)
                        # Show the description below the image
                        preview = source.get("content_preview", "")
                        if preview:
                            st.caption(preview[:300])
                        relevance = source.get("similarity", "N/A")
                        st.markdown(f"- Relevance: {relevance}")
                    else:
                        st.markdown(f"**Source {i+1}:** {title}")
                        page_start = meta.get("page_start", "")
                        page_end = meta.get("page_end", "")
                        if page_start and page_end and page_start != page_end:
                            st.markdown(f"- Pages: {page_start}-{page_end}")
                        elif page_start:
                            st.markdown(f"- Page: {page_start}")
                        section = meta.get("section", "")
                        if section and section != "unknown":
                            st.markdown(f"- Section: {section.title()}")
                        relevance = source.get("similarity", "N/A")
                        st.markdown(f"- Relevance: {relevance}")
                    
                    if i < len(qa["sources"][:5]) - 1:
                        st.divider()

# ==================== TOP PROFILE BUTTON ====================
def render_top_profile():
    """Render the profile button in the top right corner"""
    if st.session_state.authenticated:
        # Wrap the popover in a container with a key for reliable CSS targeting
        with st.container(key="nav-profile-popover"):
            with st.popover("Profile"):
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

# Inject dynamic CSS variables for sidebar state
st.markdown(f"""
    <style>
    :root {{
        --sidebar-width: {sidebar_width} !important;
        --sidebar-text-align: {sidebar_align} !important;
        --sidebar-btn-padding: {btn_padding} !important;
        --sidebar-btn-content-align: {btn_align} !important;
        --sidebar-btn-justify: {btn_justify} !important;
    }}
    </style>
""", unsafe_allow_html=True)



with st.sidebar:
    # Custom sidebar toggle with panel icon (CSS provides the SVG)
    if st.button(".", key="toggle-sidebar", help="Toggle sidebar"):
        st.session_state.sidebar_expanded = not st.session_state.sidebar_expanded
        st.session_state._sidebar_toggling = True  # Flag to prevent session restore
        st.rerun()
    
    if st.session_state.sidebar_expanded:
        if st.button("Chat", key="nav-chat", help="Chat", use_container_width=True):
            st.session_state.current_tab = "chat"
            st.session_state.is_history_view = False
            st.rerun()
        
        if st.button("Documents", key="nav-docs", help="Documents", use_container_width=True):
            st.session_state.current_tab = "documents"
            st.rerun()
        
        if st.button("Compare", key="nav-compare", help="Compare", use_container_width=True):
            st.session_state.current_tab = "compare"
            st.rerun()
        
        if st.button("Export", key="nav-export", help="Export", use_container_width=True):
            st.session_state.current_tab = "export"
            st.rerun()
        

        
        # EXPORT TAB REMOVED AS REQUESTED
        
        if st.session_state.authenticated:
            if st.button("New Session", key="nav-new-session", help="Start New Session", use_container_width=True):
                # Clear session state
                st.session_state.conversation = []
                st.session_state.current_session_id = None
                st.session_state.doc_manager = DocumentManager()
                st.session_state.is_history_view = False  # Reset history view
                
                # Clear uploaded files cache
                st.session_state.pop('_uploaded_file_data', None)
                st.session_state.pop('_uploaded_file_objects', None)

                # Clear extracted images cache
                try:
                    import shutil
                    images_dir = os.path.join(os.path.dirname(__file__), "data", "images")
                    if os.path.exists(images_dir):
                        shutil.rmtree(images_dir)
                    os.makedirs(images_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to clear extracted images: {e}")
                
                st.session_state.current_tab = "chat"
                st.rerun()
            
            st.markdown("---")
            st.markdown('<p class="sidebar-section-label">HISTORY</p>', unsafe_allow_html=True)
            
            # Clear toggle flag — history clicks in THIS render cycle are safe
            is_toggling = st.session_state.pop('_sidebar_toggling', False)
            
            try:
                db = ChatSessionLocal()
                user_sessions = db.query(ChatSession).filter(
                    ChatSession.user_id == st.session_state.user_id
                ).order_by(ChatSession.last_updated.desc()).limit(20).all()
                
                for s in user_sessions:
                    label = s.title if s.title else "New Chat"
                    if len(label) > 15:
                        label = label[:13] + "..."
                    
                    if st.button(label, key=f"hist_{s.id}", use_container_width=True):
                        # Only load session if we're NOT in the middle of a toggle rerun
                        if not is_toggling:
                            load_chat_session(s.id)
                            st.session_state.current_tab = "chat"
                            st.session_state.is_history_view = True  # Mark as history view (read-only)
                            st.rerun()

                db.close()
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
            
            # Reset Knowledge Base button at the end of expanded sidebar
            if st.button("Reset Knowledge Base", key="nav-reset", help="Clear all papers & embeddings", use_container_width=True):
                
                # 1) Clear chat
                st.session_state.conversation = []
                st.session_state.current_session_id = None

                # 2) Reset document manager
                st.session_state.doc_manager = DocumentManager()
                st.session_state.documents_ready = False

                # 3) Clear Chroma embeddings
                try:
                    st.cache_resource.clear()
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR)
                        os.makedirs(CHROMA_DIR, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to clear ChromaDB: {e}")
                
                st.session_state.current_tab = "chat"
                st.rerun()

    else:
        if st.button(".", key="nav-chat-icon", help="Chat"):
            st.session_state.current_tab = "chat"
            st.session_state.is_history_view = False
            st.rerun()
        
        if st.button(".", key="nav-docs-icon", help="Documents"):
            st.session_state.current_tab = "documents"
            st.rerun()
        
        if st.button(".", key="nav-compare-icon", help="Compare"):
            st.session_state.current_tab = "compare"
            st.rerun()
        
        if st.button(".", key="nav-export-icon", help="Export"):
            st.session_state.current_tab = "export"
            st.rerun()
        
        if st.session_state.authenticated:
            if st.button(".", key="nav-new-session-icon", help="Start New Session"):
                # Clear session state
                st.session_state.conversation = []
                st.session_state.current_session_id = None
                st.session_state.doc_manager = DocumentManager()
                st.session_state.is_history_view = False  # Reset history view
                
                # Clear uploaded files cache
                st.session_state.pop('_uploaded_file_data', None)
                st.session_state.pop('_uploaded_file_objects', None)

                # Clear extracted images cache
                try:
                    import shutil
                    images_dir = os.path.join(os.path.dirname(__file__), "data", "images")
                    if os.path.exists(images_dir):
                        shutil.rmtree(images_dir)
                    os.makedirs(images_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to clear extracted images: {e}")
                
                # Clear ChromaDB to keep it fresh for the new session
                try:
                    import shutil
                    st.cache_resource.clear()
                    if os.path.exists(CHROMA_DIR):
                        shutil.rmtree(CHROMA_DIR)
                        os.makedirs(CHROMA_DIR, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed clearing Chroma: {e}")

                # 4) Clear research metadata DB
                try:
                    db = ResearchSessionLocal()
                    db.query(ResearchPaperMetadata).delete()
                    db.commit()
                    db.close()
                except Exception as e:
                    logger.error(f"Failed clearing metadata DB: {e}")

                st.success("Workspace cleared. Upload new research papers.")
                st.rerun()



# ==================== MAIN CONTENT AREA ====================
st.markdown("""
    <div class="main-header-container">
        <h1>AI Research Literature Assistant</h1>
        <p>Intelligent document analysis powered by RAG</p>
    </div>
""", unsafe_allow_html=True)

# Export button below header - only show if authenticated and has conversation
if st.session_state.authenticated and st.session_state.conversation:
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("Export", key="export-btn-header", help="Export chat history", use_container_width=True):
            st.session_state.current_tab = "export"
            st.rerun()

# ==================== CHAT TAB ====================
if st.session_state.current_tab == "chat":
    # Check if pipeline exists (persistent on disk) even if doc_manager (in-memory) is empty
    pipeline_status = get_rag_pipeline()
    
    # If we have a pipeline with documents, but doc_manager is empty (e.g. after reload),
    # we should restore the "chat" view instead of showing upload prompt.
    # However, we won't visually populate the "Document Library" tab without metadata logic,
    # but the chat will work.
    is_history = st.session_state.get('is_history_view', False)
    # has_documents = st.session_state.doc_manager.get_document_count() > 0 or (pipeline_status is not None) or is_history

    has_documents = (
        st.session_state.documents_ready
        or st.session_state.doc_manager.get_document_count() > 0
        or (pipeline_status is not None)
        or is_history
    )

    if not st.session_state.authenticated:
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            if st.button("Login\nto Upload Research PDFs", key="auth-btn-trigger", type="primary", use_container_width=True):
                auth_dialog()

    elif not has_documents:
        with st.container(key="authenticated-upload"):
            uploaded_files = st.file_uploader(
                "Upload Research PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key="uploader-component"
            )

            st.checkbox(
                "Process images (slower)",
                key="process_images",
                value=False,
            )
        
        # Cache uploaded file data in session state so it survives sidebar toggle reruns
        if uploaded_files:
            cached_files = []
            seen_names = set()
            duplicate_names = set()
            for f in uploaded_files:
                if f.name not in seen_names:
                    seen_names.add(f.name)
                    content = f.read()
                    cached_files.append({"name": f.name, "content": content, "size": len(content)})
                    f.seek(0)  # Reset for later use
                else:
                    duplicate_names.add(f.name)
            st.session_state._uploaded_file_data = cached_files
            st.session_state._uploaded_file_objects = uploaded_files
            if duplicate_names:
                already_reported = st.session_state.get("reported_duplicates", set())
                new_dupes = duplicate_names - already_reported
                if new_dupes:
                    st.session_state.reported_duplicates = already_reported | new_dupes
                    st.toast(f"{len(new_dupes)} duplicate file(s) removed from selection.")
            else:
                st.session_state.pop("reported_duplicates", None)
        
        # Display selected files from cache (survives sidebar toggle)
        cached = st.session_state.get('_uploaded_file_data', [])
        if cached:
            with st.container(key="selected-files-list-section"):
                st.markdown("---")
                st.markdown("### Selected Documents")
                
                for fd in cached:
                    size_mb = fd["size"] / 1024 / 1024
                    st.markdown(f"**{fd['name']}** ({size_mb:.2f} MB)")
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Process Documents", key="process-docs-btn"):
                    # Show full-screen processing overlay
                    st.markdown("""
                    <div class="processing-overlay">
                        <div class="custom-spinner">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 150">
                                <path fill="none" stroke="#6B7280" stroke-width="15" stroke-linecap="round" stroke-dasharray="300 385" stroke-dashoffset="0" d="M275 75c0 31-27 50-50 50-58 0-92-100-150-100-28 0-50 22-50 50s23 50 50 50c58 0 92-100 150-100 24 0 50 19 50 50Z">
                                    <animate attributeName="stroke-dashoffset" calcMode="spline" dur="2" values="685;-685" keySplines="0 0 1 1" repeatCount="indefinite"></animate>
                                </path>
                            </svg>
                        </div>
                        <div class="processing-text">Processing Documents...</div>
                        <div class="processing-subtext">Extracting text, analyzing images & building embeddings</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Hand off to ingestion layer — app.py stops here ──
                    success = _process_and_ingest(cached)
                    # if success:
                    #     st.session_state.pop('_uploaded_file_data', None)
                    #     st.session_state.pop('_uploaded_file_objects', None)
                    #     st.success("Documents processed successfully!")
                    #     get_rag_pipeline.clear()
                    #     st.rerun()
                    if success:
                        st.session_state.documents_ready = True
                        st.session_state.pop('_uploaded_file_data', None)
                        st.session_state.pop('_uploaded_file_objects', None)
                        st.success("Documents processed successfully! You can now chat with your papers.")
                        get_rag_pipeline.clear()
                        st.rerun()
                    else:
                        st.error("Processing failed or no new files to process.")
    else:
        if not st.session_state.get('is_history_view', False):
            st.markdown("### Search Settings")
            with st.container():
                search_mode, alpha = SearchModeSelector.show()
                st.session_state.search_mode = search_mode
                st.session_state.alpha = alpha
                show_sources = st.checkbox("Show source citations", value=st.session_state.show_sources)
                st.session_state.show_sources = show_sources
            
        st.markdown("<div style='margin-bottom: 3rem;'></div>", unsafe_allow_html=True)

        for qa in st.session_state.conversation:
            render_chat_message(qa)
            
        if st.session_state.get('is_history_view', False):
            st.info("This is a past conversation. Start a new session to ask more questions.")
            user_query = None
        else:
            user_query = st.chat_input("Ask a question about your documents...")
        
        if user_query:
            pipeline = get_rag_pipeline()
            
            if pipeline:
                # Custom SVG spinner for "Thinking" state
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown("""
                    <div class="thinking-spinner-container">
                         <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" class="thinking-spinner-svg">
                            <circle fill="#6B7280" stroke="#6B7280" stroke-width="15" r="15" cx="40" cy="65">
                                <animate attributeName="cy" calcMode="spline" dur="2" values="65;135;65;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="-.4"></animate>
                            </circle>
                            <circle fill="#6B7280" stroke="#6B7280" stroke-width="15" r="15" cx="100" cy="65">
                                <animate attributeName="cy" calcMode="spline" dur="2" values="65;135;65;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="-.2"></animate>
                            </circle>
                            <circle fill="#6B7280" stroke="#6B7280" stroke-width="15" r="15" cx="160" cy="65">
                                <animate attributeName="cy" calcMode="spline" dur="2" values="65;135;65;" keySplines=".5 0 .5 1;.5 0 .5 1" repeatCount="indefinite" begin="0"></animate>
                            </circle>
                        </svg>
                    </div>
                """, unsafe_allow_html=True)
                
                try:
                    result = pipeline.query(
                        user_query,
                        search_mode=st.session_state.search_mode,
                        alpha=st.session_state.alpha
                    )
                finally:
                    spinner_placeholder.empty()
                
                st.session_state.conversation.append({"role": "user", "content": user_query})
                if st.session_state.authenticated:
                    save_message("user", user_query)
                
                response_content = result.get('answer', result.get('response', ''))
                sources = result.get('sources', [])
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": response_content,
                    "sources": sources
                })
                if st.session_state.authenticated:
                    save_message("assistant", response_content, sources=sources)
                
                st.rerun()
            else:
                st.error("Pipeline not initialized. Please upload documents first.")

# ==================== DOCUMENTS TAB ====================
elif st.session_state.current_tab == "documents":
    st.subheader("Document Library")

    if st.session_state.doc_manager.get_document_count() > 0:
        # Convert DataFrame to list of dicts for iteration
        df_docs = st.session_state.doc_manager.get_all_documents()
        docs = df_docs.to_dict('records')
        
        for doc in docs:
            doc_name = doc.get('name', 'Unknown Document')
            with st.expander(doc_name):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Uploaded:**", doc.get('upload_time', 'Unknown'))
                    st.write("**Status:**", doc.get('status', 'Unknown'))
                with col2:
                    st.write("**Chunks:**", doc.get('num_chunks', 0))
                    st.write("**Size:**", f"{doc.get('size_mb', 0)} MB")
    else:
        st.info("No documents uploaded yet. Go to Chat tab to upload PDFs.")


# ==================== COMPARE TAB ====================
elif st.session_state.current_tab == "compare":
    st.subheader("Compare Research Papers")

    from db.models import ResearchPaperMetadata

    db = ResearchSessionLocal()


    try:
        papers = db.query(ResearchPaperMetadata)\
            .filter(ResearchPaperMetadata.title.isnot(None))\
            .all()

        if len(papers) < 2:
            st.info("Upload at least 2 research papers to enable comparison.")
        else:
            paper_titles = [p.title for p in papers if p.title]

            selected_titles = st.multiselect(
                "Select papers to compare:",
                paper_titles
            )

            selected_papers = []

            if selected_titles:
                selected_papers = (
                    db.query(ResearchPaperMetadata)
                    .filter(ResearchPaperMetadata.title.in_(selected_titles))
                    .all()
                )

                # Build comparison table
                comparison_data = []

                for p in selected_papers:
                    comparison_data.append({
                        "Title": p.title,
                        "Year": p.year,
                        "Domain": p.domain,
                        "Research Problem": p.research_problem,
                        "Methodology": p.methodology,
                        "Dataset": p.dataset,
                        "Evaluation Metrics": p.evaluation_metrics,
                        "Baseline Models": p.baseline_models,
                        "Key Results": p.key_results,
                        "Contributions": p.contributions,
                        "Limitations": p.limitations,
                        "Future Work": p.future_work,
                    })

                import pandas as pd
                df = pd.DataFrame(comparison_data)

                st.markdown("### Comparison Table")
                st.dataframe(df, use_container_width=True)

                # AI insight button OUTSIDE finally
                if len(selected_papers) >= 2:
                    if st.button("Generate AI Comparison Insight", type="primary", key="ai_comparison_btn"):
                        with st.spinner("Analyzing research papers..."):

                            papers_json = [
                                {
                                    "title": p.title,
                                    "year": p.year,
                                    "domain": p.domain,
                                    "research_problem": p.research_problem,
                                    "methodology": p.methodology,
                                    "dataset": p.dataset,
                                    "evaluation_metrics": p.evaluation_metrics,
                                    "baseline_models": p.baseline_models,
                                    "key_results": p.key_results,
                                    "contributions": p.contributions,
                                    "limitations": p.limitations,
                                    "future_work": p.future_work,
                                }
                                for p in selected_papers
                            ]

                            insight = generate_comparison_insight(papers_json)

                            st.markdown("AI Research Analysis")
                            st.markdown(insight)

    finally:
        db.close()



# ==================== EXPORT TAB ====================
elif st.session_state.current_tab == "export":
    st.header("Export Chat History")
    
    st.subheader("Export Document")

    if st.session_state.conversation:
        exporter = ChatExporter()
        
        format_type = st.selectbox("Select export format:", ["JSON", "CSV", "TXT"])
        
        try:
            # Generate the file content for the current format
            path = exporter.export_chat(
                st.session_state.conversation,
                format_type.lower(),
                st.session_state.session_id
            )
            
            with open(path, 'r', encoding='utf-8') as f:
                file_data = f.read()
            
            # Use the download button directly - it will have the emerald green color via CSS key
            st.download_button(
                label="Export Chat",
                data=file_data,
                file_name=f"chat_export_{st.session_state.session_id}.{format_type.lower()}",
                mime=f"text/{format_type.lower()}",
                key="export-chat-btn",
                type="primary",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Export could not be prepared: {e}")

    else:
        st.info("No chat history to export. Start a conversation first!")

# ==================== PIPELINE ====================