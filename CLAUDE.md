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

## Next build: consistency scoring + targeted clarifying question (approved by the user)
The user approved building these two ideas together as an **experimental mode** that runs
alongside the current judge, so the two can be compared on real requests. They are untested
hypotheses, not established methods. Measure before making either the default.

### Idea 1: score prompts by output consistency (semantic entropy)
Problem it solves: today a Claude judge reads the prompt and scores it. A model judging its
own style is gameable and self-preferring, which is the weakest part of the loop.
- For a candidate prompt, run it N times (config `n_samples`, default 6, range 5-8) to get
  N actual outputs.
- One Claude call groups the N outputs into meaning clusters (same substance = same
  cluster, regardless of wording). Structured output: clusters of sample indices, plus a
  one-line description of each cluster.
- Semantic entropy H = -sum(p_c * log p_c) over cluster proportions p_c. Normalize to 0-1 by
  dividing by log(N). Low entropy = the prompt is unambiguous.
- Also check that the dominant cluster matches the confirmed intent (one Claude call,
  yes/no + reason). Agreeing on the wrong thing must not score well.
- Consistency score (0-10) = 10 * (1 - normalized_H), set to 0 if the dominant cluster
  fails the intent check. Blend it with the judge's score through a config weight; the judge
  stays in as a secondary signal.
- Redundancy is used to measure a prompt, not to pick a winner. This replaces the idea of
  8-10 parallel pipelines.

### Idea 2: turn disagreement into one targeted question
- When normalized entropy is above a threshold (config, e.g. 0.4), one Claude call reads the
  cluster descriptions and names the axis the outputs split along (audience, tone, length,
  scope...). It then writes the single question for the user whose answer would most reduce
  that split, e.g. "Is this for a recruiter or a hiring manager?"
- Ask it through the `Reviewer` protocol in `review.py`: add `ask_clarifying(question) -> str`,
  with `AutoReviewer` returning "" and `ConsoleReviewer` reading stdin. The answer is
  appended to the confirmed intent, and the next revision round uses the updated intent.
- At most one clarifying question per run by default, so check-ins stay rare but sharp.

### Integration points
- `prompts.py`: new schemas and system prompts for cluster, intent-match and question.
- `engine.py`: new `_consistency(task, prompt, intent)`; call it in `optimize` next to
  `_critique`; add `Config` fields (`consistency_mode`, `n_samples`, `entropy_threshold`,
  `consistency_weight`, `sample_effort`); record entropy and clusters in `Result`.
- `llm.py`: running the candidate prompt needs a plain text call (no JSON schema); add a
  `complete(prompt, effort)` method to `ClaudeLLM` and to the `LLM` protocol.
- `cli.py`: add a `--consistency` flag and a `--sample-model` option (default: same as
  `--model`). Using `claude-haiku-5-5` for the N samples is the suggested way to keep cost
  down, but only if the user chooses it.
- Tests: extend the `FakeLLM` in `tests/test_engine.py`, and cover the entropy math, the
  intent-mismatch zeroing, question triggering, and the intent update.

### Cost and evaluation
- Each scored candidate costs N extra generations plus about 2 calls. That is several times
  more per round. Report the cost per run from `response.usage`.
- Add a side-by-side comparison: run the same requests in judge-only mode and in consistency
  mode, and compare human ratings. That comparison decides whether consistency mode becomes
  the default.

## Status / next steps
- **Next:** build the consistency scoring + clarifying question mode described above.
- Not yet run against the live API (the cloud session had no key). First thing to do on a
  machine with `ANTHROPIC_API_KEY`: run a few real `optimize` calls, read the outputs and
  judge critiques, and tune the prompts in `prompts.py`.
- Measure real cost per run from `response.usage`. The README's $0.25-0.75 figure is an estimate.
- Ideas the user has shown interest in: optional 2-3 candidate generation, a web UI `Reviewer`.
