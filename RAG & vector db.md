### Pillar 5: MLOps & Production Architecture (Retrieval-Augmented Generation & Scalability)

Deploying AI models into production reveals two major flaws in raw Large Language Models (LLMs): they **hallucinate** facts when unsure, and their internal knowledge base is **frozen in time** from their last training date. Rather than spending millions of dollars re-training parameters, production engineering fixes this by implementing an open-book architecture called **RAG (Retrieval-Augmented Generation)**. 

### 🏛️ 1. The RAG Pipeline Architecture

RAG acts as a hybrid intelligence system. It uses a high-speed search index to find external text reference material and feeds those clean facts directly into an LLM's prompt window before generation. 

### The Operational Flow:

1. **User Prompt:** The client submits a query into the application.
2. **Retrieval Stage:** The query is vectorised and sent to a search engine to query the internal corporate database.
3. **Augmentation Stage:** The system fetches the most relevant text chunks and writes a structured system prompt wrapping the query around the fresh background text.
4. **Generation Stage:** The LLM reads the verified prompt window and synthesises a clean answer entirely anchored to the facts provided.

### 📊 2. High-Dimensional Indexing: Vector Databases

A **Vector Database** (e.g., Pinecone, ChromaDB, Milvus) is a specialized storage engine optimized to store, index, and query millions of high-dimensional document embedding vectors simultaneously. 

* **Ingestion Pipeline:** Long enterprise files (PDFs, Markdown text, text strings) are parsed, chopped into distinct fragments, turned into vector coordinates using an embedding engine, and indexed.
* **Similarity Retrieval:** When a query vector enters the database, the engine runs millions of optimized **Cosine Similarity** calculations concurrently to return documents whose geometric paths align closest to the user's intent.

### ⚙️ 3. Production Optimization: Chunking & Re-ranking

A basic vector search breaks down when handling long, complex enterprise files. Production pipelines maintain accuracy by using a two-stage data-handling strategy: 

### A. Sliding Window Chunking with Overlap

Splitting data text purely by word counts can clip a sentence right in half, breaking semantic context. Engineers use a **Sliding Window** strategy: documents are cut into uniform blocks (e.g., 500 words) while preserving a strict data **Overlap** (e.g., 100 words). This ensures boundary context is shared across adjacent vector chunks. 

### B. Two-Stage Retrieval with Re-ranking

Vector similarity search is incredibly fast but can struggle with precise phrase alignment or nuance. Production engines deploy a two-stage process: 

1. **Stage 1 (Retrieval):** The fast Vector DB queries millions of entries and returns the top 20 candidate chunks.
2. **Stage 2 (Re-ranking):** A heavy, precise deep-learning model called a **Cross-Encoder (Re-ranker)** re-evaluates the relationship between the prompt text and those 20 candidates, reorganising them to place the absolute best 3 context segments directly into the LLM prompt.

text

Database Chunks (Millions) ➔ [ Vector DB Search ] ➔ Candidates (Top 20) ➔ [ Re-ranker ] ➔ Final Context (Top 3)

Use code with caution.

### 🛠️ Directory Roadmap

* document_chunking.py — Iterative sliding window text parsing with structural lookback overlap.
* vector_db_mock.py — In-memory vector indexing and cosine distance querying loop.
* rag_prompt_pipeline.py — Context extraction, system instruction template generation, and LLM input binding.
* cross_encoder_rerank.py — Query text relevance scores recalculation filter.
