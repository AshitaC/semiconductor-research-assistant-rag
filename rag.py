from uuid import uuid4
import tempfile
import os
import requests
from typing import List, Iterable, Tuple, Optional
from urllib.parse import urlparse
import ipaddress

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma


try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

load_dotenv()

# CONFIG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
HTTP_TIMEOUT = 15
MAX_TOKENS = 1500

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}



# URL VALIDATION & SECURITY
def is_safe_url(url: str) -> Tuple[bool, str]:
    """Validate URL is safe (not internal/private IP)."""
    try:
        parsed = urlparse(url)
        
        if parsed.scheme not in ['http', 'https']:
            return False, "Only HTTP/HTTPS URLs allowed"
        
        if not parsed.hostname:
            return False, "Invalid URL format"
        
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                return False, "Private/internal IP addresses not allowed"
        except ValueError:
            pass
        
        if parsed.hostname.lower() in ['localhost', '127.0.0.1', '0.0.0.0']:
            return False, "Localhost URLs not allowed"
        
        return True, ""
    
    except Exception as e:
        return False, f"URL validation error: {str(e)}"


def normalize_url(url: str) -> str:
    """Normalize URL for consistent duplicate detection."""
    url = url.strip().rstrip('/')
    parsed = urlparse(url)
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized



# INITIALIZATION (EPHEMERAL - NO PERSISTENCE)
def initialize_components(session_state):
    """Initialize LLM and in-memory vectorstore."""
    
    if session_state.get('llm') is None:
        session_state['llm'] = ChatGroq(
            #model="llama-3.3-70b-versatile",
            #model="llama-3.1-8b-instruct",
            #model="llama-3.2-3b-instruct",
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=MAX_TOKENS,
        )

    if session_state.get('vector_store') is None:
        emb = HuggingFaceEmbeddings(
            #model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"trust_remote_code": True},
        )

        # NO persist_directory = ephemeral in-memory storage
        session_state['vector_store'] = Chroma(
            embedding_function=emb
        )
    
    if 'processed_sources' not in session_state:
        session_state['processed_sources'] = set()
    if 'failed_sources' not in session_state:
        session_state['failed_sources'] = []


# PDF Detection Utility (Optimized)
def check_content_type(url: str) -> Tuple[Optional[str], Optional[bytes]]:
    """
    Check URL content type with a single request.
    Returns (content_type, pdf_bytes) or (content_type, None) for non-PDFs.
    """
    try:
        r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=HTTP_TIMEOUT)
        ct = r.headers.get("Content-Type", "").lower()
        
        if "pdf" in ct or url.lower().endswith(".pdf"):
            r_get = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
            r_get.raise_for_status()
            return "pdf", r_get.content
        
        if "html" in ct or "text" in ct:
            return ct, None
        
        r_get = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        r_get.raise_for_status()
        ct = r_get.headers.get("Content-Type", "").lower()
        
        if "pdf" in ct:
            return "pdf", r_get.content
        
        return ct, None
    
    except requests.Timeout:
        raise TimeoutError(f"Request timed out after {HTTP_TIMEOUT}s")
    except requests.HTTPError as e:
        if e.response.status_code == 403:
            raise ConnectionError(f"Access forbidden (403). This site blocks automated access. Try downloading the PDF manually and uploading it instead.")
        elif e.response.status_code == 401:
            raise ConnectionError(f"Authentication required (401). This content requires institutional access. Download the PDF manually.")
        else:
            raise ConnectionError(f"HTTP error {e.response.status_code}: {str(e)}")
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to fetch URL: {str(e)}")


def is_valid_pdf(bytes_data: bytes) -> bool:
    """Check if bytes contain a valid PDF (basic validation)."""
    if not bytes_data or len(bytes_data) < 5:
        return False
    return bytes_data[:5] == b"%PDF-"


def extract_text_from_html(url: str) -> str:
    """
    Fallback method to extract text from HTML.
    """
    try:
        from bs4 import BeautifulSoup
        
        resp = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    except ImportError:
        resp = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    
    except Exception as e:
        raise Exception(f"HTML extraction failed: {str(e)}")



# PROCESS INPUTS
def process_inputs(
    urls: Optional[List[str]], 
    pdf_files: Optional[List],
    session_state
) -> Iterable[str]:
    """Process URLs and uploaded PDFs into in-memory vectorstore."""
    
    initialize_components(session_state)
    
    vector_store = session_state['vector_store']
    processed_sources = session_state['processed_sources']
    failed_sources = session_state['failed_sources']
    
    urls = urls or []
    pdf_files = pdf_files or []
    
    docs = []
    new_processed = []
    new_failed = []


    # URL Processing

    for url in urls:
        url = url.strip()
        if not url:
            continue
        
        normalized_url = normalize_url(url)
        
        if normalized_url in processed_sources:
            yield f"[SKIP] Duplicate: {url}"
            continue
        
        is_safe, error_msg = is_safe_url(url)
        if not is_safe:
            new_failed.append((url, error_msg))
            yield f"[ERROR] Unsafe URL rejected: {url} - {error_msg}"
            continue
        
        yield f"[INFO] Checking URL: {url}"
        
        temp_path = None
        
        try:
            content_type, pdf_bytes = check_content_type(url)
            
            if pdf_bytes is not None:
                yield f"[INFO] URL returns a PDF: {url}"
                
                if not is_valid_pdf(pdf_bytes):
                    new_failed.append((url, "Invalid PDF format"))
                    yield f"[ERROR] Invalid PDF content: {url}"
                    continue
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    temp_path = tmp.name
                
                try:
                    loader = PyPDFLoader(temp_path)
                    pdf_docs = loader.load()
                    
                    if not pdf_docs or not any(d.page_content.strip() for d in pdf_docs):
                        new_failed.append((url, "PDF contains no extractable text"))
                        yield f"[ERROR] Empty PDF: {url}"
                        continue
                    
                    for i, d in enumerate(pdf_docs):
                        d.metadata.update({
                            "source": url,
                            "source_type": "pdf_url",
                            "page": i + 1,
                            "total_pages": len(pdf_docs)
                        })
                    
                    docs.extend(pdf_docs)
                    new_processed.append(normalized_url)
                    yield f"[SUCCESS] Loaded PDF ({len(pdf_docs)} pages): {url}"
                
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                
                continue
            
            yield f"[INFO] Loading webpage: {url}"
            
            try:
                loader = UnstructuredURLLoader(urls=[url], headers=HEADERS)
                html_docs = loader.load()
                
                if not html_docs or not html_docs[0].page_content.strip() or len(html_docs[0].page_content.strip()) < 100:
                    yield f"[INFO] Using fallback extractor..."
                    text = extract_text_from_html(url)
                    
                    if len(text.strip()) < 100:
                        new_failed.append((url, "Insufficient text content"))
                        yield f"[ERROR] Insufficient content from {url}"
                        continue
                    
                    html_docs = [Document(
                        page_content=text,
                        metadata={"source": url, "source_type": "html_fallback"}
                    )]
                else:
                    for d in html_docs:
                        d.metadata.update({
                            "source": url,
                            "source_type": "html"
                        })
                
                docs.extend(html_docs)
                new_processed.append(normalized_url)
                yield f"[SUCCESS] Successfully processed webpage: {url}"
            
            except Exception as e:
                try:
                    yield f"[INFO] Primary loader failed, trying fallback..."
                    text = extract_text_from_html(url)
                    
                    if len(text.strip()) < 100:
                        new_failed.append((url, f"Insufficient content: {str(e)}"))
                        yield f"[ERROR] Could not extract content from {url}"
                        continue
                    
                    html_docs = [Document(
                        page_content=text,
                        metadata={"source": url, "source_type": "html_fallback"}
                    )]
                    
                    docs.extend(html_docs)
                    new_processed.append(normalized_url)
                    yield f"[SUCCESS] Loaded using fallback: {url}"
                
                except Exception as fallback_error:
                    new_failed.append((url, f"All methods failed: {str(fallback_error)}"))
                    yield f"[ERROR] Failed to load {url}: {fallback_error}"
        
        except TimeoutError as e:
            new_failed.append((url, str(e)))
            yield f"[ERROR] Timeout: {url}"
        
        except ConnectionError as e:
            new_failed.append((url, str(e)))
            yield f"[ERROR] Connection failed: {url}"
        
        except Exception as e:
            new_failed.append((url, f"Unexpected error: {str(e)}"))
            yield f"[ERROR] Error processing {url}: {e}"
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


    # Uploaded PDFs
    for pdf in pdf_files:
        temp_path = None
        
        if pdf.name in processed_sources:
            yield f"[SKIP] Duplicate file: {pdf.name}"
            continue
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                temp_path = tmp.name
            
            try:
                loader = PyPDFLoader(temp_path)
                pdf_docs = loader.load()
                
                if not pdf_docs or not any(d.page_content.strip() for d in pdf_docs):
                    new_failed.append((pdf.name, "PDF contains no extractable text"))
                    yield f"[ERROR] Empty PDF: {pdf.name}"
                    continue
                
                for i, d in enumerate(pdf_docs):
                    d.metadata.update({
                        "source": pdf.name,
                        "source_type": "uploaded_pdf",
                        "page": i + 1,
                        "total_pages": len(pdf_docs)
                    })
                
                docs.extend(pdf_docs)
                new_processed.append(pdf.name)
                yield f"[SUCCESS] Uploaded PDF processed ({len(pdf_docs)} pages): {pdf.name}"
            
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            new_failed.append((pdf.name, str(e)))
            yield f"[ERROR] Failed to process {pdf.name}: {e}"
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass


    # Validate & Filter Docs
    docs = [d for d in docs if d.page_content.strip()]
    
    if not docs:
        yield "[ERROR] No valid documents found."
        session_state['failed_sources'].extend(new_failed)
        return

    yield f"[INFO] Total documents loaded: {len(docs)}"


    # Chunking
    yield f"[INFO] Splitting documents into chunks..."
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]

    yield f"[INFO] Created {len(chunks)} chunks from {len(docs)} documents."


    # Store in In-Memory VectorDB
    try:
        ids = [str(uuid4()) for _ in chunks]
        vector_store.add_documents(chunks, ids=ids)
        
        session_state['processed_sources'].update(new_processed)
        session_state['failed_sources'].extend(new_failed)
        
        yield f"[SUCCESS] Successfully indexed {len(chunks)} chunks from {len(new_processed)} sources!"
    
    except Exception as e:
        yield f"[ERROR] Failed to index documents: {e}"
        session_state['failed_sources'].extend(new_failed)



# GENERATE ANSWER
def generate_answer(query: str, session_state, k: int = 5) -> Tuple[str, str]:
    """Generate answer using RAG pipeline."""
    
    initialize_components(session_state)
    
    vector_store = session_state['vector_store']
    llm = session_state['llm']
    
    # Check if vectorstore has documents
    try:
        collection = vector_store._collection
        if collection.count() == 0:
            return (
                " No documents in the vectorstore. Please process some sources first.",
                "No sources available"
            )
    except:
        pass
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    prompt = PromptTemplate.from_template("""
You are a Semiconductor Industry Research Analyst with deep expertise in chip manufacturing, materials science, and industry trends.

INSTRUCTIONS:
- Use ONLY the retrieved context below to answer the question
- Write clean, professional answers without referencing document IDs, metadata, or technical artifacts
- Be specific and cite information naturally (e.g., "According to the source material..." or "The documents indicate...")
- If the context doesn't contain enough information, say so clearly
- Do NOT hallucinate or make up information
- Do NOT include document IDs, UUIDs, or any technical metadata in your response
- Provide detailed, technical answers when appropriate in a readable format

Context:
{context}

Question: {question}

Answer:
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)

    # Get source documents
    retrieved_docs = retriever.invoke(query)
    sources_info = []
    seen_sources = set()
    
    for d in retrieved_docs:
        source = d.metadata.get("source", "Unknown")
        if source not in seen_sources:
            page = d.metadata.get("page")
            source_type = d.metadata.get("source_type", "unknown")
            
            if page:
                sources_info.append(f" {source} (page {page}, {source_type})")
            else:
                sources_info.append(f" {source} ({source_type})")
            
            seen_sources.add(source)
    
    sources_text = "\n".join(sources_info) if sources_info else "No sources found"

    return answer, sources_text



# UTILITY FUNCTIONS
def get_vectorstore_stats(session_state) -> dict:
    """Get statistics about the vectorstore."""
    try:
        initialize_components(session_state)
        
        vector_store = session_state.get('vector_store')
        if not vector_store:
            return {
                "total_chunks": 0,
                "total_sources": 0,
                "failed_sources": 0
            }
        
        collection = vector_store._collection
        count = collection.count()
        
        return {
            "total_chunks": count,
            "total_sources": len(session_state.get('processed_sources', set())),
            "failed_sources": len(session_state.get('failed_sources', []))
        }
    except Exception as e:
        return {
            "total_chunks": 0,
            "total_sources": 0,
            "failed_sources": 0,
            "error": str(e)
        }


