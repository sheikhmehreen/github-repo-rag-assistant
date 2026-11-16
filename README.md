# GitHub Repo RAG Assistant

A Retrieval-Augmented Generation (RAG) application that helps developers understand public GitHub repositories.

## Overview
Submit a public GitHub repository URL, ask questions about its codebase, and receive answers grounded in the actual code, with file and line citations.

## Planned Features
- Public GitHub repository ingestion
- Code and documentation chunking
- Semantic search with embeddings
- Context-grounded answers from an LLM
- File and line citations

## Planned Technology Stack
Subject to change as decisions are finalized:
- Backend: Python 3.12, FastAPI
- Retrieval: Sentence Transformers, ChromaDB
- Storage: SQLite
- LLM: Ollama (local), provider-agnostic design
- Frontend: Next.js, TypeScript
- Tooling: Docker, GitHub Actions

## Project Structure
Here is how the project folders are organized:

├── .github/
│   └── workflows/        # Automated tests and checks
├── backend/
│   ├── app/
│   │   ├── api/          # Web endpoints and routing
│   │   ├── core/         # Settings and configurations
│   │   ├── database/     # Database models and sessions
│   │   ├── generation/   # LLM response logic
│   │   ├── ingestion/    # Code for fetching GitHub repos
│   │   └── retrieval/    # Vector search and context matching
│   └── tests/            # Backend unit and integration tests
├── docs/                 # Documentation and guides
└── frontend/             # Next.js and TypeScript user interface


## Status
Under active development. Not yet runnable.

## Setup
Instructions will be added as development progresses.