# AGENTS.md

This file gives coding agents (and human contributors) the context needed to work on this repository correctly and consistently. Read this before making changes.

## 1. Project Overview

**AOA — LLM Answer or Abstain** is a document-based Retrieval-Augmented Generation (RAG) system. Users upload documents, ask natural-language questions, and receive answers that are verified for faithfulness to the source content before being returned. If verification fails, the system either regenerates the answer or falls back to web search, and explicitly abstains (returns "insufficient information") when no reliable answer can be produced.

Core pipeline: **ingest → retrieve → generate → verify (Lynx) → classify failure → remediate → respond**.

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Package/env manager | `uv` |
| Backend framework | FastAPI |
| Orchestration | LangGraph |
| Retrieval/indexing | LlamaIndex |
| Vector DB | ChromaDB (dev), Qdrant/pgvector (prod) |
| Generation LLM | Groq API |
| Hallucination verification | Patronus Lynx API |
| Web verification | Tavily / SerpAPI |
| Frontend | React / Next.js |
| Testing | Pytest, pytest-asyncio, pytest-cov |
| Linting/formatting | Ruff, Black, mypy |

## 3. Environment Setup

This project uses **`uv`** for dependency and environment management. Do not use `pip install` directly or commit a `requirements.txt` as the source of truth — `pyproject.toml` + `uv.lock` are authoritative.

```bash
# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and sync the virtual environment from pyproject.toml / uv.lock
uv sync

# Activate the environment (or prefix commands with `uv run`)
source .venv/bin/activate
```

**Rules for agents:**
- Always add new dependencies via `uv add <package>` (or `uv add --dev <package>` for dev-only tools) — never hand-edit `pyproject.toml` dependency lists without running `uv lock` afterward.
- Never commit `.venv/` — it is git-ignored.
- Run all commands through `uv run <command>` unless the venv is already activated in the current shell, to guarantee the correct interpreter and dependency versions are used.
- Required environment variables (never hardcode or commit actual values — use `.env`, which is git-ignored, and update `.env.example` when adding new ones):
  - `GROQ_API_KEY`
  - `PATRONUS_API_KEY`
  - `SERPAPI_API_KEY` (or `WEB_SEARCH_API_KEY`)
  - `WEB_SEARCH_API_KEY`
  - `VECTOR_DB_URL` (if not using local Chroma)

## 4. Project Structure

```
aoa/
├── backend/
│   ├── ingestion/          # Document parsing, chunking, embedding
│   ├── retrieval/          # Vector search against the index
│   ├── generation/         # Prompt construction + Groq calls
│   ├── verification/       # Lynx integration
│   ├── remediation/        # Web search fallback + regeneration
│   ├── orchestrator/       # LangGraph state graph definition
│   ├── api/                # FastAPI routes
│   └── models/             # Shared Pydantic/TypedDict schemas (e.g. RAGState)
├── frontend/                # React/Next.js chat UI
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/            # Sample documents, mock API responses
├── pyproject.toml
├── uv.lock
├── .env.example
└── AGENTS.md
```

Each backend module should expose a clear, typed function/class interface and avoid reaching into another module's internals directly — communicate through the shared `RAGState` schema and defined function signatures, not ad hoc dict access.

## 5. Code Quality Requirements

Agents must ensure the following before considering any change complete:

1. **Type hints everywhere.** All function signatures must have full type annotations. Run `mypy backend/` and resolve errors before finishing a task.
2. **Formatting and linting.** Run `uv run ruff check --fix .` and `uv run black .` before finishing any change. No code should be submitted with lint errors.
3. **No bare `except:` blocks.** Catch specific exceptions (e.g. `httpx.TimeoutException`, `ValueError`), log the error with context, and handle it explicitly per the fallback rules in the project's LLD (Section 7 — insufficient info, API failure resilience, etc.).
4. **No secrets in code or commits.** API keys must only ever be read from environment variables via a config module (e.g. `backend/config.py`), never inlined.
5. **Docstrings on public functions/classes.** Every function exposed outside its own module needs a docstring describing inputs, outputs, and side effects.
6. **Small, single-responsibility functions.** If a function is doing ingestion *and* validation *and* logging, split it up. Each pipeline stage (retrieve, generate, verify, classify, remediate) should remain a separate, independently testable unit — do not collapse stages together for convenience.
7. **State schema changes require updating `backend/models/`.** If a pipeline node needs a new field in `RAGState`, add it to the shared schema definition, not as an ad hoc key injected at runtime.

## 6. Testing Requirements

**Framework:** Pytest, with `pytest-asyncio` for async endpoints and `pytest-cov` for coverage reporting.

### 6.1 What must be tested

- **Unit tests** for each pipeline module in isolation (ingestion, retrieval, generation, verification, classification, remediation) using mocked external API calls (Groq, Lynx, web search) — no real network calls in unit tests.
- **Integration tests** for the full LangGraph pipeline end-to-end, using a small fixture document set and known question/answer pairs, still with mocked external APIs.
- **Failure-path tests** are mandatory, not optional — every remediation branch must have a test:
  - Lynx pass → direct return
  - Lynx fail + low retrieval score → web search branch triggered
  - Lynx fail + healthy retrieval score → regeneration branch triggered
  - Regeneration still fails after retry → fallback/abstain response returned
  - External API failure/timeout (Groq, Lynx, web search) → safe fallback, not a crash or silent pass
- **Error handling tests** for malformed uploads, empty documents, and unsupported file types.

### 6.2 How to run tests

```bash
# Run the full test suite
uv run pytest

# Run with coverage report
uv run pytest --cov=backend --cov-report=term-missing

# Run only unit tests
uv run pytest tests/unit

# Run only integration tests
uv run pytest tests/integration

# Run a specific test file
uv run pytest tests/unit/test_classifier.py -v
```

### 6.3 Coverage expectations

- Minimum 80% line coverage on `backend/` before a change is considered complete.
- 100% of conditional branches in `orchestrator/` (the LangGraph routing logic) must be covered — this is the core correctness-critical logic of the project (Section 7.2 of the LLD).
- New code must include new tests in the same change; do not defer test-writing to a follow-up task.

## 7. Workflow Checklist for Agents

Before marking any task complete, confirm:

- [ ] `uv sync` runs cleanly with no dependency errors
- [ ] `uv run ruff check .` and `uv run black --check .` pass
- [ ] `uv run mypy backend/` passes
- [ ] `uv run pytest --cov=backend` passes with no regressions and coverage stays ≥80%
- [ ] New/changed pipeline nodes have corresponding unit tests, including failure paths
- [ ] No secrets, API keys, or `.env` values are committed
- [ ] Any new environment variable is added to `.env.example`
- [ ] Public functions/classes have docstrings and full type hints
- [ ] Changes to `RAGState` are reflected in `backend/models/`

## 8. What Agents Should NOT Do

- Do not install packages with raw `pip` — always use `uv add`.
- Do not make real calls to Groq, Patronus, or web search APIs inside unit tests — use mocks/fixtures.
- Do not merge generation and verification logic into a single function — they must remain separately testable stages per the LLD architecture.
- Do not silently swallow exceptions from external API calls — every failure path must resolve to a defined state (`retrieval_gap`, `bad_generation`, or an explicit fallback/abstain response), never an unhandled crash or a silently returned unverified answer.
- Do not commit `.venv/`, `.env`, or any generated vector DB data files.