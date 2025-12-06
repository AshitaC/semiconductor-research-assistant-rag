#  Semiconductor Research Assistant

A RAG-based research tool for semiconductor industry analysis using LangChain, Groq, and ChromaDB.

##  Features

- **Multi-Source Processing**: Upload PDFs or provide URLs (arXiv, news articles, technical papers)
- **Intelligent Q&A**: Ask technical questions and get cited answers from your documents
- **Session-Based**: Each user gets isolated, private storage
- **Export Functionality**: Download Q&A sessions as Markdown
- **Retrieval Metrics**: View confidence scores and source attribution

##  Tech Stack

- **Framework**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Vector DB**: ChromaDB (in-memory)
- **Embeddings**: HuggingFace Sentence Transformers
- **Orchestration**: LangChain

##  Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.env`
4. Run: `streamlit run main.py`

##  Architecture
```
User Input (PDFs/URLs)
    ↓
Document Processing & Chunking
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
ChromaDB Vector Store (In-Memory)
    ↓
Query → Retrieval → LLM (Groq) → Answer
```

Embeddings — llama-3.2-11b-embed (Groq)
Used for vectorizing documents and queries.
This model provides high-quality semantic embeddings and significantly improves retrieval accuracy, especially for long or technical documents.

LLM — llama-3.1-8b-instant (Groq)
Used for generation.
Selected for its extremely low latency and strong instruction handling, making the demo fast and responsive.

##  Use Cases

- Competitive analysis (compare TSMC vs Intel approaches)
- Technical research (EUV lithography, GAA transistors)
- Market intelligence (industry trends, financial reports)
- Academic research (IEEE papers, arXiv preprints)

##  Privacy

- Session-isolated storage
- No data persistence between sessions
- No data shared between users

##  Future Enhancements

- [ ] Multi-turn conversations with memory
- [ ] Advanced RAG techniques (re-ranking, query expansion)
- [ ] Support for more document types (DOCX, HTML)
- [ ] Evaluation metrics (RAGAS framework)