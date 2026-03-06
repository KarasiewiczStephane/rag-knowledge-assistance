# RAG Knowledge Assistant

> Retrieval-Augmented Generation chatbot that ingests documents, embeds them into ChromaDB, and answers questions with source citations.

## Overview

A production-ready RAG pipeline that ingests documents (PDF, DOCX, Markdown, TXT), chunks and embeds them into a local ChromaDB vector store, and answers questions with inline citations using LangChain and configurable LLM backends (Ollama, Anthropic Claude, OpenAI). Features conversation memory with sliding window, RAGAS-style evaluation metrics, and a Streamlit chat interface.

## Architecture

```
                    +------------------+
                    |   Streamlit UI   |
                    |  (Chat + Docs)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   RAG Pipeline   |
                    |  (Orchestrator)  |
                    +--------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
+---------v------+  +--------v-------+  +-------v--------+
|   Ingestion    |  |   Retrieval    |  |   Generation   |
| Parser/Chunker |  | Vector Search  |  | LLM + Prompts  |
|   Embedder     |  |   Re-ranker    |  |   Citations    |
+----------------+  +--------+-------+  +----------------+
                             |
                    +--------v---------+
                    |    ChromaDB      |
                    |  (Vector Store)  |
                    +------------------+
```

**Data flow:** Document Upload -> Parse -> Chunk (500 chars, 50 overlap) -> Embed (MiniLM-L6) -> Store in ChromaDB -> Query -> Retrieve top-K -> Prompt LLM -> Answer with Citations

## Quick Start

```bash
# 1. Clone and set up environment
git clone git@github.com:KarasiewiczStephane/rag-knowledge-assistance.git
cd rag-knowledge-assistance
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
make install

# 3. Configure LLM backend
#    Default: Ollama (free, local) -- install from https://ollama.com then pull a model:
ollama pull llama3

#    Or switch to a cloud provider by copying and editing the env file:
cp .env.example .env
#    Uncomment and set LLM_PROVIDER=anthropic (or openai) and the matching API key.

# 4. Launch the Streamlit dashboard
make dashboard
#    Opens at http://localhost:8501

# 5. Ingest documents
#    Use the sidebar file uploader in the dashboard to add PDF, DOCX, MD, or TXT files.
#    Five sample documents are included in data/sample_docs/ for quick testing.
```

## Features

- **Multi-format ingestion**: PDF, DOCX, Markdown, TXT with metadata extraction
- **Configurable chunking**: RecursiveCharacterTextSplitter with paragraph preservation
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **Vector search**: ChromaDB with cosine similarity and optional cross-encoder re-ranking
- **Triple LLM support**: Ollama (local/free), Anthropic Claude, and OpenAI GPT -- switchable via config
- **Citation tracking**: Source file, page, chunk index, relevance score per answer
- **Conversation memory**: Sliding window (5 exchanges), session save/load
- **RAGAS evaluation**: Answer relevancy, faithfulness, context precision/recall
- **Duplicate detection**: Hash-based deduplication on ingestion
- **Streamlit dashboard**: Chat interface, document management sidebar, source panel

## Tech Stack

- **Python 3.11+** with type hints
- **LangChain** for text splitting
- **ChromaDB** for vector storage
- **sentence-transformers** for embeddings
- **Ollama / Anthropic / OpenAI** for LLM generation
- **Streamlit** for the chat interface
- **pytest** with >80% code coverage
- **ruff** for linting and formatting
- **Docker** + **GitHub Actions CI**

## Usage

### Launch the Streamlit dashboard

```bash
make dashboard
# or directly:
streamlit run src/dashboard/app.py
```

Upload documents via the sidebar, then ask questions in the chat. The assistant returns answers with source citations and confidence indicators.

### CLI mode

```bash
make run
# or:
python -m src.main
```

Prints configuration info and a pointer to the dashboard.

### Docker

```bash
docker compose up --build
# Access at http://localhost:8501
```

The Docker image bundles the sample documents in `data/sample_docs/` and persists ChromaDB and session data via named volumes.

### Development commands

```bash
make install     # Install dependencies
make test        # Run tests with coverage
make lint        # Lint and format code
make clean       # Remove caches
make run         # Run CLI entry point
make dashboard   # Launch Streamlit dashboard
make docker      # Build and run via Docker Compose
```

### Run RAGAS evaluation

```python
from src.evaluation.ragas_eval import RAGASEvaluator
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
evaluator = RAGASEvaluator(pipeline=pipeline)
cases = evaluator.load_test_cases("data/test_qa_pairs.json")
result = evaluator.evaluate(cases)
print(evaluator.generate_report(result))
```

Twenty pre-built Q&A pairs are included in `data/test_qa_pairs.json`.

## Configuration

All settings live in `configs/config.yaml` with environment variable overrides (via `.env`):

| Setting | Config Key | Env Override | Default |
|---------|-----------|--------------|---------|
| LLM provider | `llm.provider` | `LLM_PROVIDER` | ollama |
| LLM model | `llm.model` | `LLM_MODEL` | llama3 |
| Ollama base URL | `llm.base_url` | -- | http://localhost:11434 |
| Chunk size | `ingestion.chunk_size` | `CHUNK_SIZE` | 500 |
| Chunk overlap | `ingestion.chunk_overlap` | `CHUNK_OVERLAP` | 50 |
| Top-K results | `retrieval.top_k` | `RETRIEVAL_TOP_K` | 5 |
| Similarity threshold | `retrieval.similarity_threshold` | `SIMILARITY_THRESHOLD` | 0.7 |
| Embedding model | `embeddings.model` | `EMBEDDING_MODEL` | all-MiniLM-L6-v2 |
| Memory window | `memory.window_size` | `MEMORY_WINDOW_SIZE` | 5 |

To use Anthropic or OpenAI instead of Ollama, set `LLM_PROVIDER` and the corresponding API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) in your `.env` file.

## Project Structure

```
rag-knowledge-assistance/
├── src/
│   ├── ingestion/        # Document parsing, chunking, embedding
│   │   ├── parser.py     # PDF, DOCX, MD, TXT parsers
│   │   ├── chunker.py    # RecursiveCharacterTextSplitter
│   │   └── embedder.py   # sentence-transformers embeddings
│   ├── retrieval/        # Vector search and re-ranking
│   │   ├── vector_store.py  # ChromaDB integration
│   │   ├── retriever.py     # Query pipeline
│   │   └── reranker.py      # Cross-encoder re-ranking
│   ├── generation/       # LLM clients, prompts, citations
│   │   ├── llm_client.py    # Ollama + Anthropic + OpenAI clients
│   │   ├── prompt_builder.py # Context-aware prompt construction
│   │   └── citation_tracker.py # Source reference linking
│   ├── memory/           # Conversation management
│   │   └── conversation.py   # Sliding window + session persistence
│   ├── evaluation/       # Quality metrics
│   │   └── ragas_eval.py     # RAGAS-style evaluation
│   ├── dashboard/        # Streamlit UI
│   │   └── app.py
│   ├── utils/            # Config and logging
│   ├── rag_pipeline.py   # Main orchestrator
│   └── main.py           # CLI entry point
├── tests/                # 160+ pytest tests (>80% coverage)
├── configs/config.yaml   # Application configuration
├── data/
│   ├── sample_docs/      # 5 sample documents for testing
│   ├── chromadb/         # ChromaDB persistent storage (gitignored)
│   ├── sessions/         # Saved conversation sessions (gitignored)
│   └── test_qa_pairs.json # 20 Q&A pairs for RAGAS evaluation
├── .env.example          # Template for API keys and overrides
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## License

MIT
