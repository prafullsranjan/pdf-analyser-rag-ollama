# 🧠 Historical Labor Analytics RAG System

> Ask longitudinal, historically-anchored questions across Bureau of Labor Statistics Occupational Outlook Handbooks (or any temporal corpus) and get evidence-backed comparative insights. 
> Powered by **Gemma 3 (4B)** + **Ollama**, **Docling**, **ChromaDB**, and **BGE-Reranker**!

---

## 🚀 Key Upgrades & Features

1. **Intelligent Layout Parsing (Docling)** 
   Replaces primitive text extraction with `docling`, which uses AI to natively understand complex, historical, double-column PDF layouts and preserve data reading structures (like Job and Wage tables) perfectly.

2. **Temporal Anchoring** 
   Automatically extracts the year from PDF identifiers and explicitly bounds the text vectors (e.g. `Year: 1980 | Content: ...`). This anchors semantic memory meaning models won't accidentally conflate a 1950s skillset with a 2020s skillset.

3. **Hybrid & Metadata-Aware RAG** 
   Combines standard semantic similarity search with raw ChromaDB metadata filtering (`$in` operations). Users can explicitly constrain analysis down to specific eras (e.g., comparing 1980 vs. 2000).

4. **Two-Stage Reranking Pipeline** 
   Retrieves broad sets of chunks from ChromaDB and passes them through a powerful cross-encoder (`BAAI/bge-reranker-base`) to rigorously evaluate actual context-to-query correlation, ensuring the LLM only generates insights from absolute high-signal snippets.

5. **Expanded Context Analysis** 
   Pushed `num_ctx` to `8192` allowing Gemma 3 to perform large-scale longitudinal decade-over-decade labor comparisons in one pass without truncating context.

---

## 🔄 System Architecture Flow

```text
================================================================================
                           SYSTEM ARCHITECTURE FLOW
================================================================================

  [ 💻 app.py ] (Frontend)                      [ ⚙️ rag_pipeline.py ] (Backend)
         │                                                 │
         ▼                                                 ▼
   ┌───────────┐    User provides folder path        ┌─────────────┐
   │   Index   │ ──────────────────────────────────▶ │  Scan PDFs  │
   └───────────┘                                     └─────────────┘
         ▲                                                 │
         │                                                 ▼
         │  Streamlit real-time loading UI           ┌─────────────┐
         │                                           │   Docling   │ Layout parsing
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │                                           │  Temporal   │ Year:XXXX | prefix
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │ ◀ - - Stream indexing statistics - - - -  │   Chunking  │ Recursive splitting
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │                                           │  Embeddings │ nomic-embed-text
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                          [(  ChromaDB  )] Vector Data Store
         │                                                 
         │                                                 
   ┌───────────┐    User query + Year Filters        ┌─────────────┐
   │ Generate  │ ──────────────────────────────────▶ │   Retrieve  │
   └───────────┘                                     └─────────────┘
         ▲                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │                                           │ Sem Search  │ Embed query string
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │                                           │ Filter Meta │ Isolate $in Years
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                          [(  ChromaDB  )] Extract Top M chunks
         │                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │                                           │  Reranker   │ BGE cross-encoder
         │                                           └─────────────┘
         │                                                 │
         │                                                 ▼
         │                                           ┌─────────────┐
         │ ◀ - - Render Analytics & LLM Answer - - - │   Gemma3    │ Context Generation
         │                                           └─────────────┘
```

---

## 📁 Project Structure

```text
people_analytics_rag/
├── app.py              ← Streamlit frontend (Dynamic UI, filtering, stats tracker)
├── rag_pipeline.py     ← Core backend (Docling → Chunking → Embed → Rerank → Chroma → Gemma 3)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
├── chroma_store/       ← Auto-created: persistent local vector database
└── index_manifest.json ← Auto-created: incremental indexed corpus tracking map
```

---

## ⚙️ Setup & Installation

### Step 1 — Install Ollama
You need Ollama running to process embeddings and generate local LLM responses.
* macOS: `brew install ollama`
* Linux: `curl -fsSL https://ollama.com/install.sh | sh`
* Windows: [Download Installer](https://ollama.com/download/windows)

Start the server:
```bash
ollama serve
```

### Step 2 — Pull Required Models
```bash
# LLM for advanced analytical reasoning (~5 GB)
ollama pull gemma3:4b

# Embedding generation (~274 MB)
ollama pull nomic-embed-text
```

### Step 3 — Python Environment
```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows

# Install critical dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Launch the Front End**
```bash
streamlit run app.py
```

2. **Index your Corpus**
   - Provide an absolute path to your folder containing historical PDFs.
   - Files **must** have their publication year in the title (e.g. `OOH_1980.pdf`) for the Temporal Anchoring to work.
   - Click **⚡ Index**. 
   - A dynamic progress card in the sidebar will track PDF discovery, layout extraction via Docling, temporal chunk injection, and ChromaDB uploading.

3. **Interrogate History**
   - Head to the main query interface. 
   - Select specific decades in the **Sidebar Filter** to force hybrid metadata isolation.
   - Use the **Top K** slider to expand or constrain how much historic data the LLM correlates.
   - Choose output formatting (e.g. Markdown text, visual KPI analysis, visual mapping).
   - Generate insights!

4. **Maintenance**
   - Clicking **🔄 Refresh** runs an incremental diff over your provided repo file map.
   - Clicking **🗑️ Clear** wipes ChromaDB and safely destroys the tracking manifest so you can start entirely over.

---

## 🛠️ Modifying Constants

If you have exceptional hardware or want stricter routing constraints, override the top of `rag_pipeline.py`:

```python
CHUNK_SIZE        = 800     # Word/Token density per doc slice  
CHUNK_OVERLAP     = 150     # Bleed-over buffer ensuring context continuation
MAX_CHUNKS_RETURN = 10      # Max passages queried from Vector Store before reranking
LLM_MODEL         = "gemma3:4b"  # Deep analytical reasoning backbone
EMBED_MODEL       = "nomic-embed-text" 
```

---

## 📋 Example Analytic Inquiries

- *"How did definitions surrounding 'Automation' evolve between 1970 and 1990?"*
- *"Based on these job profiles, how did the minimum technical requirements for 'Accountant' change decade-over-decade?"*
- *"Track the transition of required communication skills across managerial roles."*
- *"Historically, when did computer terminology begin significantly infiltrating blue-collar job expectations?"*

---
*Built for Longitudinal Labor & Sociology analysis. Powered by Gemma 3, Docling, BGE-Reranker, ChromaDB, and Streamlit.*
