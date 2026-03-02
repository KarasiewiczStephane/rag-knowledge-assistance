# RAG Knowledge Assistant

> Retrieval-Augmented Generation chatbot that ingests documents, embeds them into ChromaDB, and answers questions with source citations.

## Overview

A production-ready RAG pipeline that ingests documents (PDF, DOCX, Markdown, TXT), chunks and embeds them into a local ChromaDB vector store, and answers questions with inline citations using LangChain and Anthropic Claude / OpenAI APIs. Features conversation memory with sliding window, RAGAS-style evaluation metrics, and a Streamlit chat interface.

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

## Features

- **Multi-format ingestion**: PDF, DOCX, Markdown, TXT with metadata extraction
- **Configurable chunking**: RecursiveCharacterTextSplitter with paragraph preservation
- **Embedding**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **Vector search**: ChromaDB with cosine similarity and optional cross-encoder re-ranking
- **Dual LLM support**: Anthropic Claude and OpenAI GPT, switchable via config
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
- **Anthropic / OpenAI** for LLM generation
- **Streamlit** for the chat interface
- **pytest** with >80% code coverage
- **ruff** for linting and formatting
- **Docker** + **GitHub Actions CI**

## Setup

```bash
# Clone
git clone git@github.com:KarasiewiczStephane/rag-knowledge-assistance.git
cd rag-knowledge-assistance

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and/or OPENAI_API_KEY
```

## Usage

### Launch the Streamlit UI

```bash
streamlit run src/dashboard/app.py
```

### CLI mode

```bash
python -m src.main
```

### Docker

```bash
docker compose up --build
# Access at http://localhost:8501
```

### Development commands

```bash
make install   # Install dependencies
make test      # Run tests with coverage
make lint      # Lint and format code
make clean     # Remove caches
make run       # Run main entry point
make docker    # Build and run via Docker Compose
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

## Configuration

All settings are in `configs/config.yaml` with environment variable overrides:

| Setting | Config Key | Env Override | Default |
|---------|-----------|--------------|---------|
| Chunk size | `ingestion.chunk_size` | `CHUNK_SIZE` | 500 |
| Chunk overlap | `ingestion.chunk_overlap` | `CHUNK_OVERLAP` | 50 |
| Top-K results | `retrieval.top_k` | `RETRIEVAL_TOP_K` | 5 |
| Similarity threshold | `retrieval.similarity_threshold` | `SIMILARITY_THRESHOLD` | 0.7 |
| LLM provider | `llm.provider` | `LLM_PROVIDER` | anthropic |
| LLM model | `llm.model` | `LLM_MODEL` | claude-sonnet-4-20250514 |
| Embedding model | `embeddings.model` | `EMBEDDING_MODEL` | all-MiniLM-L6-v2 |
| Memory window | `memory.window_size` | `MEMORY_WINDOW_SIZE` | 5 |

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
│   │   ├── llm_client.py    # Anthropic + OpenAI clients
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
├── tests/                # 150+ pytest tests (>80% coverage)
├── configs/config.yaml   # Application configuration
├── data/
│   ├── sample_docs/      # 5 sample documents for testing
│   └── test_qa_pairs.json # 20 Q&A pairs for evaluation
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## License

MIT
