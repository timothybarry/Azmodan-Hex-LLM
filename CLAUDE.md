# CLAUDE.md

Context for Claude Code sessions working on this repo.

## What this is
`prompt-rag`: a self-improving, iterative RAG system that turns a rough request into an
optimized LLM prompt. Python 3.10+, Anthropic SDK, SQLite. See README.md for the architecture.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests (offline, fake LLM, no API key needed): `python -m pytest -q`
- Run: `ANTHROPIC_API_KEY=... prompt-rag optimize "request" --verbose`
- Other: `prompt-rag feedback <id> <0-10> "comment"`, `prompt-rag reflect`, `prompt-rag stats`

## Design decisions (keep unless the user changes them)
- **Claude for everything.** The user wants their Claude API credit to cover both generation
  and retrieval. Claude has no embeddings endpoint, so "embedding" is done by Claude-written
  semantic profiles (indexed with BM25) plus Claude reranking of the BM25 shortlist.
  Don't add a third-party embedding provider without asking.
- **Default model `claude-opus-5-5`** for drafting, judging and retrieval;
  `--retrieval-model` lets the user pick a cheaper model for profiling and reranking.
- **Iterative RAG:** each revision round re-retrieves lessons/examples using the judge's
  weaknesses (`engine._expand_context`).
- **Human check-ins are core, not optional extras** (`review.py`): confirm intent → drift gate
  (fidelity < 7 is quarantined from few-shot unless a human vouches) → accept/edit/reject review
  → human approval of reflection's lesson adds/retirements. Rationale: a loop scored only by its
  own judge drifts toward pleasing the judge and stores those drifted prompts as "good" examples.
- **No 8-10 parallel pipelines.** The user asked about running 8-10. We recommended against it
  (cost, and it amplifies judge-gaming). If multi-candidate is wanted, 2-3 candidates max.
- Request settings in `llm.py`: adaptive thinking, `output_config.format` JSON schema, and
  server-side `fallbacks: "default"` only for models in `_FALLBACK_MODELS`.

## Status / next steps
- Not yet run against the live API (the cloud session had no key). First thing to do on a
  machine with `ANTHROPIC_API_KEY`: run a few real `optimize` calls, read the outputs and
  judge critiques, and tune the prompts in `prompts.py`.
- Measure real cost per run from `response.usage`. The README's $0.25-0.75 figure is an estimate.
- Ideas the user has shown interest in: optional 2-3 candidate generation, a web UI `Reviewer`.
