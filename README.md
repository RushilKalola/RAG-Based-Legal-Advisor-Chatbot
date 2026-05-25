# ⚖️ RAG-Based Legal Advisor Chatbot

An AI-powered legal assistant for Indian law, built with a full Retrieval-Augmented Generation (RAG) pipeline. Ask natural-language questions about Indian legal acts, or get a structured side-by-side comparison of any two acts on a topic — all grounded in the actual document text, never fabricated.

---

## Features

- **Chat interface** — Ask questions about Indian law; get precise answers with cited source documents.
- **Act Comparison** — Compare how two Indian acts treat any legal topic (e.g. "punishment for theft" across BNS vs IPC).
- **RAG pipeline** — PDF ingestion → semantic chunking → vector embeddings → Qdrant retrieval → cross-encoder reranking → Mistral LLM answer generation.
- **Source transparency** — Every answer includes the names of the source PDF files used to generate it.
- **Concurrency-safe** — A global semaphore limits simultaneous Mistral API calls (max 3) to avoid rate-limit errors.
- **Streamlit UI** — Clean two-page frontend (Chat + Act Comparison) served alongside the FastAPI backend.
- **Docker-ready** — Single `docker compose up` starts both the API server and the Streamlit UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (port 8501)              │
│       💬 Chat Page          🔄 Act Comparison Page       │
└───────────────┬─────────────────────────┬───────────────┘
                │ POST /chat/             │ POST /compare/
┌───────────────▼─────────────────────────▼───────────────┐
│              FastAPI Backend (port 8000)                  │
│   /chat/    →  ChatTool  →  ChatService                  │
│   /compare/ →  ActComparisonTool  →  ComparisonService   │
│   /health/  →  Health check                              │
└───────────────┬─────────────────────────────────────────┘
                │
      ┌─────────▼─────────┐
      │   RetrievalService │
      │  1. Embed query    │  ← SentenceTransformer (all-mpnet-base-v2)
      │  2. Qdrant search  │  ← Qdrant Cloud (top-12, cosine similarity)
      │  3. Rerank results │  ← CrossEncoder (ms-marco-MiniLM-L-12-v2)
      │  4. Return top-5   │
      └─────────┬─────────┘
                │
      ┌─────────▼─────────┐
      │   Mistral LLM      │  ← mistral-large-latest (via LangChain)
      │   Generate answer  │
      └───────────────────┘
```

### Ingestion pipeline (run once before starting the server)

```
PDF files → PDFLoader (pypdf) → TextSplitter (SemanticChunker)
         → SentenceTransformer embeddings → Qdrant Cloud upsert
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral AI (`mistral-large-latest`) via LangChain |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| Vector DB | Qdrant Cloud |
| Text splitting | LangChain `SemanticChunker` |
| PDF parsing | pypdf |
| API server | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Evaluation | RAGAS, ROUGE, BERTScore |

---

## Legal Documents Included

The `data/raw 2/` folder ships with nine Indian legal PDFs:

- The Bharatiya Nyaya Sanhita, 2023
- The Code of Civil Procedure, 1908
- The Code of Criminal Procedure, 1973
- The Companies Act, 2013
- The Constitution of India
- The Consumer Protection Act, 2019
- The Indian Evidence Act, 1872
- The Information Technology Act, 2000
- The Motor Vehicles Act, 1988

---

## Project Structure

```
RAG-Based-Legal-Advisor-Chatbot/
├── app/
│   ├── api/
│   │   ├── main.py                  # FastAPI app, CORS, route registration
│   │   └── routes/
│   │       ├── chat.py              # POST /chat/
│   │       ├── compare.py           # POST /compare/
│   │       └── health.py            # GET /health/
│   ├── ingestion/
│   │   ├── ingest.py                # Full ingestion pipeline (run once)
│   │   ├── pdf_loader.py            # PDF → raw text via pypdf
│   │   └── text_splitter.py         # SemanticChunker via LangChain
│   ├── services/
│   │   ├── retrieval.py             # Qdrant search + cross-encoder rerank
│   │   ├── chat_services.py         # RAG answer generation via Mistral
│   │   └── comparison_service.py    # Parallel dual-act retrieval + comparison
│   ├── tools/
│   │   ├── chat_tool.py             # Thin wrapper around ChatService
│   │   └── act_comparison_tool.py   # Thin wrapper around ComparisonService
│   └── utils/
│       ├── config.py                # Settings loaded from .env
│       └── logger.py                # Loguru logger
├── data/
│   └── raw 2/                       # Indian legal PDFs (9 files)
├── eval_results/                    # RAGAS/ROUGE/BERTScore evaluation JSONs
├── streamlit_app.py                 # Streamlit two-page UI
├── eval.py                          # Evaluation script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env                             # (you create this — see below)
```

---

## Prerequisites

- Python 3.12+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier works)
- A [Mistral AI](https://console.mistral.ai/) API key
- Docker & Docker Compose (optional, for containerized run)

---

## Setup & How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/RAG-Based-Legal-Advisor-Chatbot.git
cd RAG-Based-Legal-Advisor-Chatbot
```

### 2. Create your `.env` file

Create a file named `.env` in the project root with the following contents:

```env
# Qdrant Cloud
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=legal_docs

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768

# Mistral AI
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key
MISTRAL_MODEL=mistral-large-latest

# FastAPI
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development
LOG_LEVEL=INFO

# Retrieval
TOP_K_RESULTS=12
SCORE_THRESHOLD=0.40
RERANK_TOP_K=5
```

---

### Option A — Run locally (without Docker)

#### Step 1: Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

#### Step 2: Ingest the PDF documents into Qdrant

This step only needs to be done once. It reads every PDF in `data/raw 2/`, chunks it semantically, embeds each chunk, and uploads the vectors to your Qdrant collection.

```bash
PYTHONPATH=. python app/ingestion/ingest.py
```

> The script will print progress for each file and confirm when ingestion is complete. This may take several minutes depending on your machine.

#### Step 3: Start the FastAPI backend

```bash
PYTHONPATH=. uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs`

#### Step 4: Start the Streamlit frontend

Open a second terminal (with the virtual environment activated):

```bash
PYTHONPATH=. streamlit run streamlit_app.py --server.port 8501
```

The UI will be available at `http://localhost:8501`.

> **Note:** When running locally, Streamlit points to `http://app:8000` by default (the Docker service name). Change the URLs at the top of `streamlit_app.py` to `http://localhost:8000` before running locally:
> ```python
> CHAT_API_URL    = "http://localhost:8000/chat/"
> COMPARE_API_URL = "http://localhost:8000/compare/"
> ```

---

### Option B — Run with Docker Compose (recommended)

#### Step 1: Ingest documents first (still required, run once locally)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python app/ingestion/ingest.py
```

#### Step 2: Start both services

```bash
docker compose up --build
```

This starts:
- **FastAPI backend** on `http://localhost:8000` (4 Uvicorn workers)
- **Streamlit UI** on `http://localhost:8501`

To stop:

```bash
docker compose down
```

---

## API Reference

### `GET /` — Root health check
```json
{ "message": "RAG Legal Advisor Chatbot is running 🚀", "environment": "development" }
```

### `POST /chat/` — Ask a legal question

**Request:**
```json
{ "query": "What are the fundamental rights under the Constitution of India?" }
```

**Response:**
```json
{
  "answer": "The Constitution of India guarantees six fundamental rights...",
  "sources": ["THE CONSTITUTION OF INDIA.pdf"]
}
```

### `POST /compare/` — Compare two acts on a topic

**Request:**
```json
{
  "topic": "punishment for theft",
  "act_a": "Bharatiya Nyaya Sanhita",
  "act_b": "Code of Criminal Procedure"
}
```

**Response:**
```json
{
  "topic": "punishment for theft",
  "act_a": "Bharatiya Nyaya Sanhita",
  "act_b": "Code of Criminal Procedure",
  "comparison": "### 1. Bharatiya Nyaya Sanhita — Key Provisions\n...",
  "sources_a": ["THE BHARATIYA NYAYA SANHITA, 2023.pdf"],
  "sources_b": ["THE CODE OF CRIMINAL PROCEDURE, 1973.pdf"]
}
```

### `GET /health/` — Health check

---

## Evaluation

The project includes a comprehensive evaluation script (`eval.py`) using RAGAS, ROUGE, and BERTScore. Results are saved as timestamped JSON files in `eval_results/`.

Latest evaluation scores (from `eval_results/2026-04-24_09-22-31_full.json`):

| Metric | Chat | Act Comparison |
|---|---|---|
| Faithfulness | 0.933 | 0.846 |
| Answer Relevancy | 0.908 | 0.866 |
| Context Recall | 1.000 | 0.722 |
| Context Precision | 0.840 | 0.167 |

To run evaluation:
```bash
PYTHONPATH=. python eval.py
```

---

## How the RAG Pipeline Works

1. **Ingestion** — Each PDF is loaded with `pypdf`, semantically chunked using LangChain's `SemanticChunker` (powered by `all-mpnet-base-v2`), and each chunk is embedded and upserted into Qdrant Cloud.

2. **Retrieval** — At query time, the user's question is embedded with the same model. Qdrant returns the top-12 most similar chunks (cosine similarity, HNSW index). A `CrossEncoder` (`ms-marco-MiniLM-L-12-v2`) then reranks all 12 candidates and only the top 5 are passed forward.

3. **Generation** — The top-5 chunks are assembled into a prompt along with a strict instruction to answer only from the provided context. Mistral LLM generates the final answer. If the answer isn't in the context, it says so explicitly.

4. **Act Comparison** — Two parallel Qdrant searches run concurrently (one per act), results are filtered by source filename, and a structured prompt asks Mistral to produce a four-section comparison (key provisions A, key provisions B, similarities, differences).

---

## Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `QDRANT_URL` | Your Qdrant Cloud cluster URL | — |
| `QDRANT_API_KEY` | Qdrant Cloud API key | — |
| `QDRANT_COLLECTION_NAME` | Collection name in Qdrant | — |
| `EMBEDDING_MODEL` | HuggingFace embedding model name | — |
| `EMBEDDING_DIMENSION` | Vector size (768 for mpnet) | `768` |
| `MISTRAL_API_KEY` | Mistral AI API key | — |
| `MISTRAL_MODEL` | Mistral model name | — |
| `APP_HOST` | FastAPI host | `0.0.0.0` |
| `APP_PORT` | FastAPI port | `8000` |
| `TOP_K_RESULTS` | Number of Qdrant results to fetch | `12` |
| `SCORE_THRESHOLD` | Minimum cosine similarity score | `0.40` |
| `RERANK_TOP_K` | Final top-K after cross-encoder reranking | `5` |

---

## Troubleshooting

**`ModuleNotFoundError`** — Make sure `PYTHONPATH=.` is set when running any script, or run from the project root.

**Streamlit can't reach the API** — If running locally, update `CHAT_API_URL` and `COMPARE_API_URL` in `streamlit_app.py` from `http://app:8000` to `http://localhost:8000`.

**Qdrant collection already exists** — The ingestion script skips collection creation if it already exists; re-ingestion will add duplicate points. Delete the collection from your Qdrant Cloud dashboard before re-ingesting.

**Mistral rate limit (429)** — The app retries automatically with exponential backoff (up to 6 retries via LangChain). The UI shows a friendly "service busy" message if all retries fail.

**Ingestion is slow** — SemanticChunker calls the embedding model for every split decision. This is expected; a 2MB PDF may take 2–5 minutes. Run ingestion once and it persists in Qdrant.
