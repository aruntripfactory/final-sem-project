import streamlit as st
import os
from ingestion.pdf_loader import extract_text_from_pdf
from ingestion.preprocess import ResearchPaperChunker
from ingestion.embed_store import store_documents_enhanced
from rag.pipeline import RAGPipeline

st.set_page_config(
    page_title="AI Research Literature Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize RAG pipeline (only when needed)
@st.cache_resource(show_spinner=False)
def get_pipeline():
    try:
        return RAGPipeline()
    except Exception:
        # Collection doesn't exist yet - will be created on first upload
        return None

# Show loading state during initialization
if 'initialized' not in st.session_state:
    with st.spinner("🔄 Initializing system..."):
        pipeline = get_pipeline()
        st.session_state.initialized = True
else:
    pipeline = get_pipeline()

# Main UI
st.title("📚 AI Research Literature Assistant")

st.sidebar.header("📄 Upload Research Papers")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload research papers to analyze"
)

# Track processed files to avoid reprocessing
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

if uploaded_files:
    # Get list of new files that haven't been processed
    current_files = {f.name for f in uploaded_files}
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    
    if new_files:
        chunker = ResearchPaperChunker(chunk_size=1000, overlap=150)
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        percentage_text = st.sidebar.empty()
        
        total_files = len(new_files)
        
        for idx, uploaded in enumerate(new_files):
            current_file_num = idx + 1
            progress_percentage = int((current_file_num / total_files) * 100)
            
            # Update progress bar and percentage
            progress_bar.progress(current_file_num / total_files)
            percentage_text.markdown(f"**Progress: {progress_percentage}%** ({current_file_num}/{total_files} files)")
            status_text.text(f"📖 Processing {uploaded.name}...")
            
            try:
                # Step 1: Reading PDF
                status_text.text(f"📖 [1/4] Reading {uploaded.name}...")
                raw_bytes = uploaded.read()
                
                # Step 2: Extracting text
                status_text.text(f"📄 [2/4] Extracting text from {uploaded.name}...")
                raw_text = extract_text_from_pdf(raw_bytes)
                
                paper_id = uploaded.name.replace('.pdf', '')
                
                # Step 3: Extracting metadata
                status_text.text(f"🔍 [3/4] Analyzing metadata for {uploaded.name}...")
                metadata = chunker.extract_metadata(raw_text)
                extracted_title = metadata.title if metadata.title else "No title detected"
                
                chunks = chunker.chunk_text(raw_text, paper_id=paper_id)
                
                # Step 4: Storing in database
                status_text.text(f"💾 [4/4] Storing {uploaded.name} in database...")
                chunk_ids = [f"{paper_id}_chunk_{i:04d}" for i in range(len(chunks))]
                
                store_documents_enhanced(
                    chunks=chunks,
                    ids=chunk_ids,
                    collection_name="research_papers"
                )
                
                # Mark as processed and show extracted title
                st.session_state.processed_files.add(uploaded.name)
                st.sidebar.success(f"✅ {uploaded.name}")
                st.sidebar.info(f"📝 Title: {extracted_title[:100]}...")
            except Exception as e:
                st.sidebar.error(f"❌ {uploaded.name}: {str(e)[:50]}")
        
        # Final completion message
        percentage_text.markdown(f"**✨ Complete: 100%** ({total_files}/{total_files} files)")
        status_text.text("✨ All documents processed successfully!")
        progress_bar.progress(1.0)
        
        # Clear progress indicators after 2 seconds worth of display
        import time
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        percentage_text.empty()
        # Reinitialize pipeline after new documents
        st.cache_resource.clear()
        st.rerun()
    else:
        # Show already processed files
        st.sidebar.info(f"✅ {len(current_files)} file(s) already indexed")

st.subheader("💬 Ask Questions")

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Show status message if no documents indexed
if pipeline is None:
    st.info("👆 Upload PDF files in the sidebar to get started!")

# Display conversation history
for idx, qa in enumerate(st.session_state.conversation):
    st.markdown(f"### ❓ Question {idx + 1}")
    st.write(qa['question'])
    st.markdown(f"### 💡 Answer")
    st.write(qa['answer'])
    
    if qa.get('sources') and qa.get('show_sources'):
        with st.expander("📚 Source Citations"):
            for i, source in enumerate(qa['sources'], 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Source {i}:** {source['id']}")
                with col2:
                    st.caption(f"Similarity: {source['similarity']}")
                
                if source.get("metadata"):
                    meta = source["metadata"]
                    st.caption(f"📄 {meta.get('paper_id', 'unknown')} | "
                             f"Section: {meta.get('section', 'unknown')}")
                
                with st.container():
                    st.text(source['content_preview'])
                st.divider()
    
    st.markdown("---")

# Input for new question using form to enable Enter key submission
with st.form(key=f"query_form_{len(st.session_state.conversation)}", clear_on_submit=True):
    question = st.text_input("Enter your question", placeholder="e.g., What are the main findings?")
    
    # Add a checkbox for showing sources
    show_sources = st.checkbox("Show source citations", value=False)
    
    # Submit button - form will also submit on Enter key
    submit_button = st.form_submit_button("🔍 Get Answer", type="primary")

if submit_button and question:
    if pipeline is None:
        st.warning("⚠️ Please upload PDF files first.")
    else:
        with st.spinner("🤔 Analyzing your documents..."):
            try:
                response = pipeline.query(
                    question=question,
                    n_retrieve=5,
                    include_sources=show_sources,
                    generate_response=True
                )
                
                if response.get("success"):
                    # Add to conversation history
                    st.session_state.conversation.append({
                        'question': question,
                        'answer': response["answer"],
                        'sources': response.get("sources", []),
                        'show_sources': show_sources
                    })
                    st.rerun()
                else:
                    st.error(f"❌ {response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Add buttons for clearing conversation and database
col1, col2 = st.columns(2)

with col1:
    if st.session_state.conversation:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation = []
            st.rerun()

with col2:
    if st.button("🔄 Clear Database & Re-index"):
        try:
            # Clear the collection
            from ingestion.embed_store import create_client
            client = create_client()
            try:
                client.delete_collection("research_papers")
                st.success("✅ Database cleared! Please re-upload your PDFs.")
            except:
                st.info("No database to clear.")
            
            # Clear processed files
            st.session_state.processed_files = set()
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
