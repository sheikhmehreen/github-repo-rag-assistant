# GitHub Repo RAG Assistant

A work-in-progress Retrieval-Augmented Generation (RAG) application for asking questions about public GitHub repositories. The planned experience is to return answers grounded in repository files, with citations.

## Current project layout

```text
github-repo-rag-assistant/
├── backend/
│   ├── app/                    # Python FastAPI application
│   │   ├── api/                # HTTP routes and error responses
│   │   ├── core/               # Settings, exceptions, and logging
│   │   ├── database/           # Reserved for database code
│   │   ├── generation/         # Reserved for LLM response code
│   │   ├── ingestion/          # Repository input and ingestion code
│   │   ├── retrieval/          # Reserved for search and context retrieval
│   │   └── main.py             # FastAPI application setup
│   └── tests/                  # Tests grouped to match backend/app
│       ├── api/
│       ├── core/
│       └── ingestion/
├── docs/                       # Project documentation
├── frontend/                   # Web user interface (planned)
├── .github/workflows/          # CI workflows (planned)
├── .env.example                # Safe template for local configuration
├── docker-compose.yml          # Container orchestration configuration
├── Dockerfile                  # Container build configuration
└── README.md
```

## Where to put files

- Put Python application code in `backend/app/`, inside the folder for its responsibility. For example, GitHub URL validation belongs in `backend/app/ingestion/`.
- Put backend tests in `backend/tests/`, mirroring the application folders. For example, ingestion tests belong in `backend/tests/ingestion/`.
- Put browser UI code in `frontend/` and project guides in `docs/`.
- Keep real credentials and local settings in the root `.env` file. It is ignored by Git. Update `.env.example` with variable names and safe example values only.
- Cache folders, virtual environments, and generated data are local artifacts; do not place application source code in them.

## Planned stack

- Backend: Python 3.12 and FastAPI
- Retrieval: Sentence Transformers and ChromaDB
- Storage: SQLite
- LLM: Ollama, with a provider-agnostic design
- Frontend: Next.js and TypeScript
- Tooling: Docker and GitHub Actions

## Status

Stage 2 implements GitHub repository URL validation, metadata lookup, commit-pinned bounded archive download, safe extraction, and filtering and reading of repository text files. Epic 3 adds language-aware chunking and a local Chroma index. Epic 4 adds normalized embeddings, a concurrency-safe vector-store interface, SQLite repository deduplication, and a failure-safe ingestion pipeline. The answer endpoint and frontend remain in progress.

### Epic 3 (chunking and vector indexing)

Install the CPU build of PyTorch first, then the dependencies in the root `requirements.txt`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Use Python 3.12 for this environment, as required by the planned backend stack.

The embedding model downloads from Hugging Face on first use. Chunk settings and the Chroma persistence path can be adjusted with `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL_NAME`, `RETRIEVAL_TOP_K`, `CHROMA_PERSIST_DIR`, and `MAX_CHUNKS_PER_REPO` in `.env`. To index a local folder and inspect results, run from the repository root:

```powershell
python -m scripts.smoke_index . "where are the application settings loaded"
```

### Epic 4 (repository ingestion pipeline)

The pipeline validates a public repository URL, checks metadata and size limits, claims a deduplicated SQLite job, then downloads and safely extracts the commit archive. It filters and reads files, chunks and embeds them in batches, writes a per-repository cosine index, and records success or failure. Temporary files are cleaned up on every run. Paths and limits are configurable with `EMBEDDING_BATCH_SIZE`, `SQLITE_PATH`, `INGESTION_TIMEOUT_SECONDS`, `INGESTION_MAX_REPO_SIZE_KB`, `INGESTION_MAX_CHUNKS`, and `INGESTION_TEMP_DIR`.

Run the real-repository retrieval milestone from the repository root (it contacts GitHub and downloads the configured public repositories):

```powershell
python -m scripts.milestone1 | Tee-Object -FilePath milestone1-output.txt
```

Review the top five results for each question and record counts, timing, and one good and one weak result per repository in `docs/milestone1-notes.md`.

## Repository ingestion (Stage 2)

The MVP accepts repository-root URLs in these forms:

- `https://github.com/<owner>/<repo>`
- `https://github.com/<owner>/<repo>/`
- `https://github.com/<owner>/<repo>.git`

Branch and tree URLs, non-HTTPS URLs, other hosts, query strings, and fragments are rejected. Repository downloads are pinned to the commit SHA returned by GitHub.

Default ingestion limits are 51,200 KB per repository, 52,428,800 archive bytes, 209,715,200 extracted bytes, 2,000 regular files, 1,048,576 bytes per file, and a 100:1 compression ratio. API requests time out after 15 seconds and archive downloads have a 60-second overall deadline. These values are configurable through the matching `Settings` fields and environment variables.

The filter prunes `.git`, `node_modules`, virtual environments, build output, and other generated directories. It excludes lockfiles, minified or generated files, unsupported extensions, and sensitive files such as `.env`, private keys, certificates, credential JSON, and `secrets.*`. `.env.example` and `.env.sample` are allowed. Binary, non-UTF-8, empty, unreadable, and oversized files are skipped rather than added to the indexed file list.

The default pytest run excludes network tests. To opt in to the live test against the small public `octocat/Hello-World` repository, run this from `backend/`:

```powershell
python -m pytest -m integration tests/ingestion/test_live_integration.py
```

This test requires internet access and uses GitHub's unauthenticated API. Keep it excluded from CI and the default test run.
