# Document Question-Answering (QA) System

A production-quality, clean-architecture, completely **offline** Document Question-Answering (QA) System built using Python, Streamlit, FAISS, Sentence-Transformers, and Hugging Face Transformers.

This application allows you to index multiple PDF files locally, chunk and embed them using a sentence transformer model, build a local FAISS vector index, and query them using a local Large Language Model (LLM) — all running entirely on your machine without sending data to any external APIs or cloud services.

## Features

1. **Multiple PDF Upload**: Process one or multiple PDFs simultaneously via Streamlit.
2. **Deterministic Text Extraction**: Page-by-page text parsing with exact page-level metadata capturing.
3. **Semantic Text Chunking**: Recursive character splitting with customizable chunk sizes and overlap offsets.
4. **Local Embedding Generation**: Generates 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
5. **FAISS Vector DB**: In-memory similarity search with disk persistence.
6. **Local LLM Execution**: Offline Question Answering powered by `google/flan-t5-base`.
7. **Hallucination Protection**: Prompt templates designed to force the LLM to adhere strictly to the retrieved context.
8. **Modern Chat UI**: Responsive, dark-themed Streamlit chat interface showing clear source citations and match scores.

---

## Directory Architecture

The project is structured following clean-architecture principles, separating data extraction, chunking, representation, index storage, generation, and presentation logic:

```
Document-QA-System/
├── utils/
│   ├── pdf_loader.py       # Extract text & metadata from PDFs page-by-page
│   ├── text_splitter.py     # Custom semantic chunking with overlap tracking
│   ├── embedder.py          # Sentence-Transformers local model wrapper
│   ├── vector_store.py      # FAISS indexing, loading, saving, and querying
│   ├── llm.py               # Hugging Face local LLM runner and prompt templates
│   └── rag.py               # Orchestrator integrating all components
├── uploads/                 # Directory holding copies of uploaded files
├── vector_store/            # Stores the serialized FAISS index and metadata
├── models/                  # Caches the weights for offline embeddings and LLM
├── assets/                  # Folder for images, custom CSS, or styling assets
├── app.py                   # Main Streamlit web application
├── requirements.txt         # Package dependencies file
└── README.md                # System documentation
```

---

## Installation & Setup

### Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- C++ Build Tools (required by FAISS on some Windows setups; if not installed, Pip will download pre-built wheels).

### Step 1: Clone or Navigate to the Directory
```bash
cd c:\Users\HP\Desktop\Document-QA-System
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate     # On Windows PowerShell/Cmd
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Run the Web Interface
Execute the following command to start the Streamlit application:
```bash
streamlit run app.py
```

Streamlit will boot up and provide a local URL (usually `http://localhost:8501`). Open it in your web browser.

---

## Offline Deployment & Model Caching

During the **first run**, the application will connect to the internet to download the models from Hugging Face:
1. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (~90MB) -> Saved in `models/embeddings/`
2. **Language Model**: `google/flan-t5-base` (~990MB) -> Saved in `models/llm/`

After this initial run, the models are stored locally. **You can disconnect your network connection entirely.** The application will always load the cached models directly from disk and run fully offline.

To force offline execution programmatically:
- The system uses `local_files_only=True` to attempt loading from the cache first.
- Only if the models are not found does it revert to online retrieval to fetch them.

---

## Technical Component Details

- **`pdf_loader.py`**: Reads PDF documents and outputs page data dicts. Grabs metadata like original filename and page index. Includes checks for encrypted and blank pages.
- **`text_splitter.py`**: Chunks text pages using separators `\n\n`, `\n`, ` `, and `""`. Respects the `chunk_size` limit and includes a running backtracking system for `chunk_overlap`.
- **`embedder.py`**: Standardizes embedding generation. Handles CPU/GPU loading and normalization of output vectors.
- **`vector_store.py`**: Manages FAISS Indexing. Utilizes Inner Product indices (`IndexFlatIP`) matching Cosine Similarity. Contains pickle serialization to map index offsets back to source chunks.
- **`llm.py`**: Instantiates a PyTorch Text2Text generation pipeline. Employs instruction templates to ensure facts are sourced strictly from documents.
- **`rag.py`**: Integrates the pipeline. Lazily initializes components when they are needed rather than at startup, saving RAM/VRAM during uploads.
