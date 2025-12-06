import streamlit as st
from rag import (
    process_inputs, 
    generate_answer, 
    initialize_components,
    get_vectorstore_stats
)

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# PAGE CONFIG
st.set_page_config(
    page_title=" Semiconductor Research Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


# SESSION STATE INITIALIZATION

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'llm' not in st.session_state:
    st.session_state.llm = None

if 'processed_sources' not in st.session_state:
    st.session_state.processed_sources = set()

if 'failed_sources' not in st.session_state:
    st.session_state.failed_sources = []

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Add a reset counter to force widget clearing
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0


# HEADER
st.title(" Semiconductor Research Assistant")
st.markdown("""
""")

# Display stats if vectorstore has content
stats = get_vectorstore_stats(st.session_state)
if 'error' not in stats and stats.get('total_chunks', 0) > 0:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(" Total Chunks", stats['total_chunks'])
    with col2:
        st.metric(" Sources Processed", stats['total_sources'])
    with col3:
        st.metric(" Failed Sources", stats['failed_sources'])
else:
    st.info(" **Welcome!** Upload PDFs or provide URLs and ask questions  about semiconductor manufacturing, technology nodes, materials, and industry trends. ")

#st.divider()


# SIDEBAR INPUTS

st.sidebar.header(" Source Inputs")

st.sidebar.markdown("### Upload PDFs")

pdf_files = st.sidebar.file_uploader(
    "Select PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload research papers, reports, or technical documents",
    key=f"pdf_uploader_{st.session_state.reset_counter}"
)
st.sidebar.info("""
**Recommended PDFs:** Technical whitepapers, Research papers (IEEE PDFs), Investor reports, Fab process documents
""")

st.sidebar.markdown("###  URLs")

url1 = st.sidebar.text_input("URL 1", placeholder="https://...", key=f"url1_{st.session_state.reset_counter}")
url2 = st.sidebar.text_input("URL 2", placeholder="https://...", key=f"url2_{st.session_state.reset_counter}")
url3 = st.sidebar.text_input("URL 3", placeholder="https://...", key=f"url3_{st.session_state.reset_counter}")
st.sidebar.info("""
**Usually work:** arXiv, news sites (CNBC, TechCrunch, etc.),blogs  
**May fail:** ACM, IEEE, Springer (require institutional access)  
**Solution:** Download PDFs manually and upload above
""")

st.sidebar.divider()

# Action buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    process_button = st.sidebar.button(" Process Sources", use_container_width=True, type="primary")

with col2:
    refresh_button = st.sidebar.button(" Start Fresh", use_container_width=True)

# Info section
with st.sidebar.expander(" Help & Tips"):
    st.markdown("""
    **Supported Sources:**
    -  News articles & blog posts
    -  PDF research papers
    -  IEEE stamp links
    -  Investor reports
    
    **Tips:**
    - Process sources before asking questions
    - Use specific, technical questions
    - Click "Start Fresh" to clear all data and begin again
    
    **Privacy:**
    - Data exists only in this browser session
    - No data is saved between sessions
    """)



# REFRESH/RESET FUNCTIONALITY

if refresh_button:
    # Clear the vectorstore properly
    if 'vector_store' in st.session_state and st.session_state.vector_store:
        try:
            # Delete the collection to clear all data
            st.session_state.vector_store.delete_collection()
        except:
            pass
    
    # Increment reset counter to force widget clearing
    if 'reset_counter' in st.session_state:
        st.session_state.reset_counter += 1
    
    # Clear all session state except reset_counter
    reset_count = st.session_state.reset_counter
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Restore reset counter
    st.session_state.reset_counter = reset_count
    
    st.success(" Session cleared! Starting fresh...")
    st.rerun()



# PROCESS INPUTS
if process_button:
    urls = [u for u in (url1, url2, url3) if u.strip()]
    
    if not urls and not pdf_files:
        st.warning(" Please provide at least one URL or PDF file.")
    else:
        status_container = st.container()
        
        with status_container:
            st.markdown("### Processing Sources")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            status_expander = st.expander(" Processing Log", expanded=True)
            
            with status_expander:
                log_container = st.container()
            
            messages = []
            total_steps = len(urls) + (len(pdf_files) if pdf_files else 0)
            current_step = 0
            
            try:
                for msg in process_inputs(urls, pdf_files, st.session_state):
                    messages.append(msg)
                    
                    # Check for completion indicators (works without emojis too)
                    if any(indicator in msg for indicator in ["✅", "❌", "⏭️", "Loaded", "Failed", "Skipping", "processed"]):
                        current_step += 1
                        progress = min(current_step / max(total_steps, 1), 1.0)
                        progress_bar.progress(progress)
                    
                    with log_container:
                        st.text(msg)
                    
                    progress_text.text(f"Processing: {current_step}/{total_steps}")
                
                progress_bar.progress(1.0)
                progress_text.text(" Processing complete!")
                
                st.session_state.processing_complete = True
                
            except Exception as e:
                st.error(f" Processing error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander(" Successfully Processed Sources", expanded=True):
                good_sources = sorted(st.session_state.processed_sources)
                if good_sources:
                    for src in good_sources:
                        st.success(f" {src}")
                else:
                    st.info("No sources processed yet.")
        
        with col2:
            with st.expander("Failed Sources", expanded=bool(st.session_state.failed_sources)):
                failed = st.session_state.failed_sources
                if failed:
                    for src, err in failed:
                        st.error(f" **{src}**\n\n{err}")
                else:
                    st.info("No failures.")



# QUESTION SECTION
st.divider()
st.subheader(" Ask Your Questions:")
query = st.text_area(
    label="Question",
    placeholder="Example: What are the latest developments in chiplet technology? How do they compare to monolithic designs?",
    help="Ask specific questions about semiconductor technology, manufacturing, or industry trends",
    height=100,
    label_visibility="collapsed",
    key=f"query_{st.session_state.reset_counter}"
)

with st.expander(" Example Questions"):
    st.markdown("""
    - What node sizes are discussed in the documents?
    - Explain the EUV lithography process mentioned in the sources
    - What are the main challenges in 3nm chip manufacturing?
    - Compare TSMC and Intel's manufacturing approaches
    - What materials are used for gate-all-around transistors?
    """)

with st.expander("Advanced Settings"):
    k_value = st.slider(
        "Number of source chunks to retrieve",
        min_value=3,
        max_value=15,
        value=8,
        help="Higher values provide more context but may include less relevant information"
    )
    
    show_metrics = st.checkbox("Show retrieval metrics", value=False, help="Display confidence scores and chunk details")

if query:
    if not st.session_state.processed_sources:
        st.warning(" Please process some sources first before asking questions.")
    else:
        with st.spinner(" Analyzing sources and generating answer..."):
            try:
                answer, sources = generate_answer(query, st.session_state, k=k_value)
                
                st.markdown("###  Answer")
                st.markdown(answer)
                
                st.markdown("###  Sources Referenced")
                if sources and sources != "No sources available":
                    # Parse and display sources in a nice format
                    source_lines = sources.split('\n')
                    for source_line in source_lines:
                        if source_line.strip():
                            st.markdown(f"- {source_line}")
                else:
                    st.warning(" No specific sources were retrieved for this query.")
                
                # Export functionality
                st.divider()
                export_content = f"""# Research Query & Answer

## Question:
{query}

## Answer:
{answer}

## Sources Referenced:
{sources}

---
Generated by Semiconductor Research Assistant
"""
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.download_button(
                        label="Download Q&A as Markdown",
                        data=export_content,
                        file_name="research_qa.md",
                        mime="text/markdown"
                    )
                
                # Show metrics if enabled
                if show_metrics:
                    st.divider()
                    st.markdown("###  Retrieval Metrics")
                    
                    # Get retriever and retrieve docs again for metrics
                    from rag import initialize_components
                    initialize_components(st.session_state)
                    retriever = st.session_state['vector_store'].as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k_value}
                    )
                    retrieved_docs = retriever.invoke(query)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks Retrieved", len(retrieved_docs))
                    with col2:
                        unique_sources = len(set(d.metadata.get('source', '') for d in retrieved_docs))
                        st.metric("Unique Sources", unique_sources)
                    with col3:
                        avg_chunk_size = sum(len(d.page_content) for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
                        st.metric("Avg Chunk Size", f"{int(avg_chunk_size)} chars")
                    
                    # Show chunk details
                    with st.expander("View Retrieved Chunks"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            st.markdown(f"**Chunk {i}** - Source: `{doc.metadata.get('source', 'Unknown')}`")
                            st.caption(f"Page: {doc.metadata.get('page', 'N/A')} | Type: {doc.metadata.get('source_type', 'N/A')}")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.divider()
                
                st.divider()
                col1, col2, col3, col4, col5 = st.columns([1,1,1,1,2])

            
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                import traceback
                st.code(traceback.format_exc())


st.divider()
st.caption(" Powered by LangChain + Groq + ChromaDB ")
