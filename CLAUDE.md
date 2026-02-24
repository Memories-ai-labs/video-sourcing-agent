# CLAUDE.md

## Build & Dev

```bash
# Install (editable)
pip install -e ".[dev]"

# Run tests
pytest                              # all tests
pytest tests/test_sorting.py        # single file
pytest tests/test_web_api.py::TestSSEEvents  # single class
pytest tests/test_retry.py -v       # verbose
pytest -x                           # stop on first failure

# Lint & type-check
ruff check src/ tests/              # lint (rules: E, F, I, N, W, UP)
ruff format src/ tests/             # format
mypy src/                           # strict mode, pydantic plugin

# Dev server
uvicorn video_sourcing_agent.web.main:app --reload --port 8000

# Docker
docker compose up                   # dev with hot reload via volume mount
docker build -t vexel .             # production image
```

## Architecture

Agentic loop pattern: Gemini orchestrates tool calls in a step-bounded loop (`max_agent_steps`, default 10).

```
User query
  -> QueryParser (LLM slot extraction via Gemini, regex fallback)
  -> ClarificationManager (checks if query is too ambiguous)
  -> VideoSourcingAgent.query() agentic loop:
       Gemini decides which tools to call
       -> ToolRegistry dispatches to BaseTool subclasses
       -> RetryExecutor (exponential backoff + fallback chains)
       -> Results filtered by TimeFrame before Gemini sees them
       -> Loop until Gemini produces final text answer or max steps
  -> AgentResponse (answer + VideoReferences + UsageMetrics)
```

**Web layer**: FastAPI + SSE streaming (`sse-starlette`). `StreamingAgentWrapper` yields typed `SSEEvent` subclasses (`started`, `progress`, `tool_call`, `tool_result`, `clarification_needed`, `complete`, `error`). Single endpoint: `POST /api/v1/queries/stream`.

**Tool system**: `BaseTool` ABC with `name`, `description`, `input_schema`, `execute()`, `health_check()`. `ToolRegistry` manages registration/dispatch. `ToolResult` has `.ok()`, `.no_results()`, `.fail()` constructors.

**Available tools**: YouTube (search/channel), Exa (search/similar/content/research), TikTok/Instagram/Twitter (Apify-based search + creator), Memories.ai v2 (metadata/transcript/MAI transcript/VLM analysis), VideoSearch (unified Exa+Apify).

**OpenClaw integration**: `openclaw_runner.py` streams agent events as NDJSON for CLI consumers.

## Key Conventions

- **Rename in progress**: `video_sourcing_agent` -> `vexel`. Files are staged as renames but on-disk the package is still `src/video_sourcing_agent/`. Imports use `video_sourcing_agent.*`.
- **Python 3.11+**, async-first (`async def execute`, `pytest-asyncio` with `asyncio_mode = "auto"`)
- **Pydantic v2** for all models and settings (`BaseModel`, `BaseSettings` from `pydantic-settings`)
- **Settings**: env vars loaded via `pydantic-settings` with `validation_alias` for `UPPER_CASE` env names. `.env` file supported.
- **Ruff**: line-length 100, target py311, rules `E F I N W UP`
- **mypy**: strict mode with `pydantic.mypy` plugin (`init_forbid_extra`, `init_typed`, `warn_required_dynamic_aliases`)
- **Tests**: `tests/` dir, class-based grouping (e.g. `TestSSEEvents`), `pytest-asyncio` for async tests
- **Cost tracking**: `UsageMetrics` tracks Gemini token costs + per-tool invocation costs (flat-rate or token-based for VLM)
