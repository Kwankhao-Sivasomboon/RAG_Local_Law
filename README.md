# Thai Law RAG (Retrieval-Augmented Generation) System

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Supported-green?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langchain)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange?logo=ollama&logoColor=white)](https://ollama.com)
[![VectorDB](https://img.shields.io/badge/ChromaDB-v1.5%2B-blue?logo=database&logoColor=white)](https://github.com/chroma-core/chroma)

A localized, high-precision retrieval-augmented generation (RAG) system for Thai legal question-answering. This repository is specifically optimized to perform hybrid semantic-lexical searches over core and recent Thai legislation datasets, run local LLM generation using **Ollama (`llama3.2`)**, and perform programmatic evaluations using LLM-as-a-judge methodologies on Hugging Face test sets.

---

## 🎯 Project Objective
The goal of this project is to provide strictly accurate, hallucination-free answers to Thai legal queries. It solves standard RAG challenges (like vocabulary mismatches, document fragmentation, and domain-specific terms) by implementing a custom **True Hybrid Retriever** with **Reciprocal Rank Fusion (RRF)**, query expansion, and structural metadata filters.

---

## 📂 Directory Structure

Below is the workspace layout. Every major python module has clickable links for easy navigation:

```
├── chroma_db/                  # Local SQLite and vector index files generated upon ingestion
├── datasets/                   # Storage for Parquet test/train datasets and raw JSONL laws
│   ├── test-00000-of-00001.parquet
│   └── train-00000-of-00001.parquet
├── src/
│   ├── config.py               # Central configuration module (paths, model names, retrieval parameters)
│   ├── evaluate.py             # Script to evaluate RAG performance using LLM-as-a-judge
│   ├── ingest_core.py          # Data ingestion pipeline for core Thai laws (Hugging Face / Krisdika)
│   ├── ingest_recent.py        # Data ingestion pipeline for recent Thai laws (IAPP 2025 format)
│   ├── llm_client.py           # Local Ollama client with structured Chain-of-Thought prompting
│   ├── main.py                 # Interactive console application for Q&A testing
│   ├── retriever.py            # High-performance Hybrid Retriever (Chroma + PyThaiNLP BM25 + RRF)
│   └── scratch/
│       └── download_dataset.py # Script to download evaluation datasets from Hugging Face
├── requirements.txt            # Python library dependencies
└── README.md                   # This overview file
```

### Module Contexts (For AI Agents and Developers)
- [src/config.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/config.py): Manages paths, vector store collection names (`core_law` & `recent_law`), chunk sizes (`1000` tokens, `200` overlap), and retrieval configurations.
- [src/ingest_core.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/ingest_core.py): Standardizes and ingest core Thai legislation from the Krisdika dataset. It extracts hierarchy levels (e.g. `มาตรา` or `ข้อ`), cleans section headers via regex, and uploads them to ChromaDB in batches of `500` to prevent database corruption.
- [src/ingest_recent.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/ingest_recent.py): Parses recent legislation documents (from OCR markdown outputs in the `iapp_2025` directory) and indexes them into a separate vector database collection.
- [src/retriever.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/retriever.py): Performs dynamic multi-stage hybrid retrieval:
  1. Combines semantic vector similarity search (dense) and PyThaiNLP-tokenized BM25 (sparse).
  2. Applies reciprocal rank fusion (RRF) to merge ranks.
  3. Uses hard metadata title filters (e.g., matching the specific Act in the query) and query expansion templates to filter out irrelevant contexts.
- [src/llm_client.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/llm_client.py): Sets up `ChatOllama` using the zero-temperature `llama3.2` model. Employs strict system prompts forcing Chain-of-Thought (ข้อเท็จจริงอ้างอิง -> สรุป: <ใช่/ไม่ใช่/ได้/ไม่ได้>) and returns `"ข้อมูลไม่เพียงพอ"` when contexts do not contain enough facts.
- [src/evaluate.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/evaluate.py): An automated benchmark suite. It runs retrieval and generation on test data, utilizes the LLM as a judge to evaluate answer correctness (calculating Accuracy, Precision, Recall, F1, False Positive Rate, and False Negative Rate), and outputs a confusion matrix.
- [src/main.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/main.py): A CLI loop allowing manual Q&A interactions with the RAG pipeline.
- [requirements.txt](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/requirements.txt): Lists all package dependencies with version constraints.

---

## 🌟 Core Features

### 1. True Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
To counter the vocabulary mismatch issue common in Thai legal scripts, the retriever retrieves candidates using:
- **Dense Vector Search:** Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings via ChromaDB.
- **Sparse BM25 Search:** Utilizes `pythainlp.tokenize.word_tokenize` to ensure morphologically correct word-level lexical matching for Thai words.
Both result ranks are merged using Reciprocal Rank Fusion (RRF):
$$\text{Score}(d) = \frac{1}{60 + \text{Rank}_{\text{vector}}(d)} + \frac{1}{60 + \text{Rank}_{\text{BM25}}(d)}$$

### 2. Strict Grounding and Hallucination Control
The `LLMClient` incorporates system constraints designed to eliminate LLM hallucinations:
- **No prior knowledge:** The model must rely solely on the provided context block.
- **Chain of Thought:** Forces the model to state legal references and quotes before making a final conclusion.
- **Strict Conclusion Format:** Answers must end with `สรุป: ได้`, `สรุป: ไม่ได้`, `สรุป: ใช่`, or `สรุป: ไม่ใช่`.
- **Fail-safe:** Out-of-context queries return `ข้อมูลไม่เพียงพอ`.

### 3. Comprehensive Metric Evaluations
The evaluator in [src/evaluate.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/evaluate.py) parses semantic polarities of answers (Positive, Negative, Other) and outputs:
- **Accuracy, Precision, Recall, and F1-Score** for the Positive class.
- **False Positive Rate (FPR):** Measures critical legal risks where the system incorrectly says something is permitted (`ได้`) when it is prohibited.
- **False Negative Rate (FNR):** Measures instances where valid permissions are incorrectly marked as prohibited or insufficient information.

---

## 🛠️ Tech Stack & Library Choices

| Library / Tool | Role in Project | Rationale |
| :--- | :--- | :--- |
| **Python 3.11+** | Core Programming Language | High stability with deep learning/embeddings and LangChain integration. |
| **langchain & langchain-chroma** | LLM & Database Orchestration | Simplifies retrieval-augmented setups and facilitates seamless database wrappers. |
| **langchain-ollama** | Local LLM Integration | Interfaces with Ollama's local server running `llama3.2`. |
| **chromadb** | Vector Database | Fast, lightweight, embeddable local vector database. |
| **sentence-transformers** | Embedding Generation | Generates high-quality dense vector embeddings using `paraphrase-multilingual-MiniLM-L12-v2`. |
| **pythainlp** | Thai Word Tokenizer | Critical for splitting Thai sentences without spaces into tokens before feeding them to BM25. |
| **pandas & pyarrow** | Parquet Data Handling | Efficiently loads and reads Hugging Face dataset files (`.parquet`). |
| **torch & torchvision** | Backend Computation | PyTorch runtime utilized as the computing backend for Hugging Face embeddings. |

---

## 🔑 Environment Configuration
The application does not require heavy environment variables for local operation, but you can manage overrides or settings by modifying [src/config.py](file:///c:/Users/yourh/Desktop/PersonalProject/RAG_Local_Law/src/config.py) directly.

If GPU acceleration is required on Windows, ensure that the PyTorch environment is configured correctly. A built-in DLL load fix is automated in `evaluate.py` to prevent multi-threading OpenMP crashes (`KMP_DUPLICATE_LIB_OK=TRUE`).

---

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python 3.11+ installed. Install project requirements:
```bash
pip install -r requirements.txt
```

### 2. Pull and Start the Ollama Model
Ensure [Ollama](https://ollama.com/) is installed and running, then pull the target LLM:
```bash
ollama pull llama3.2
```

### 3. Datasets Ingestion
Place your raw legislation datasets into the `datasets/` folder:
- Core Laws under `datasets/ocs-krisdika_manual/`
- Recent Laws under `datasets/iapp_2025/`

Then index the datasets to build the local ChromaDB database:
```bash
# Ingest Core Laws
python src/ingest_core.py

# Ingest Recent Laws
python src/ingest_recent.py
```

### 4. Run the Q&A Application
Launch the interactive terminal application to ask legal questions:
```bash
python src/main.py
```

### 5. Run Evaluations
Run the evaluation suite against the Hugging Face test sets to evaluate system performance:
```bash
python src/evaluate.py
```
