
#  **Semiconductor Research Assistant**

**A RAG-powered AI tool for semiconductor industry research and technical analysis.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.3_70B-black)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-306AFF)
![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow)


##  **Overview**

The **Semiconductor Research Assistant** is an advanced **RAG-based (Retrieval-Augmented Generation)** application designed for **semiconductor industry research**, combining the speed of **Groq Llama 3.3 70B**, the flexibility of **LangChain**, and the efficiency of **ChromaDB**.

Upload PDFs, provide URLs, or load technical sources — then ask **high-level engineering, market, or fabrication-related questions** and get **precise, cited answers** instantly.

---


##  Features

- **Multi-Source Processing**: Upload PDFs or provide URLs (arXiv, news articles, technical papers)
- **Intelligent Q&A**: Ask technical questions and get cited answers from your documents
- **Session-Based**: Each user gets isolated, private storage
- **Export Functionality**: Download Q&A sessions as Markdown


---

## [Live App](https://financial-kpi-extractor.streamlit.app/)

<img width="1680" height="925" alt="image" src="https://github.com/user-attachments/assets/a0eec016-050e-4d30-bffe-1b2d2650b77d" />



----

* ##  Tech Stack

- **Framework**: Streamlit
- **LLM**: Groq (Llama 3.3 70B)
- **Vector DB**: ChromaDB (in-memory)
- **Embeddings**: HuggingFace Sentence Transformers
- **Orchestration**: LangChain

---

##  Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Groq API key to `.env`
4. Run: `streamlit run main.py`
---

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
---
##  Use Cases

- Competitive analysis (compare TSMC vs Intel approaches)
- Technical research (EUV lithography, GAA transistors)
- Market intelligence (industry trends, financial reports)
- Academic research (IEEE papers, arXiv preprints)


---


##  Privacy

- Session-isolated storage
- No data persistence between sessions
- No data shared between users

##  Future Enhancements

- [ ] Multi-turn conversations with memory
- [ ] Advanced RAG techniques (re-ranking, query expansion)
- [ ] Support for more document types (DOCX, HTML)
- [ ] Evaluation metrics (RAGAS framework)
---



Contact

Ashita C

[LinkedIn Profile](https://www.linkedin.com/in/ashita-chandnani/)








