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

Stage 2 implements GitHub repository URL validation, metadata lookup, commit-pinned bounded archive download, safe extraction, and filtering and reading of repository text files. Its offline ingestion tests use mocked GitHub responses. Later stages for chunking, indexing, retrieval, answer generation, and the frontend remain in progress.

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
