# prompt-rag: a self-improving, iterative RAG prompt optimizer

Turn a rough request ("help me write a cover letter for a data job") into the prompt you
should actually send to an LLM. Every run is remembered, scored, and reused, so results
improve the more you use it. Human check-ins keep it anchored to what you actually meant.

Claude does all the model work: drafting, judging, and the retrieval jobs an
embedding model would normally do.

## How it works

```
request ──► [Claude] semantic profile ──► CHECK-IN 1: confirm/correct intent
                                                   │
          ┌────────────────────────────────────────┘
          ▼
  retrieve lessons + past examples  (BM25 over request+profile ─► [Claude] rerank ─► quality boost)
          │
          ▼
  ┌─► [Claude] draft ──► [Claude] judge (clarity, specificity, context, output spec, FIDELITY)
  │        │
  │   score < target? ── no ──► drift gate (fidelity < 7 → flagged) ──► CHECK-IN 2: accept / edit / reject + rating
  │        │ yes                                                                │
  │        ▼                                                                    ▼
  └── ITERATIVE RAG: re-retrieve using the judge's weaknesses          memory updated:
                                                                        example stored with score,
                                                                        lessons credited
```

### Why Claude does the retrieval instead of an embedding model

The Claude API has no embeddings endpoint, so Claude covers that job in two ways:

* **Semantic profiles.** Each request is indexed with a Claude-written profile (task type,
  domain, intent, and keywords including synonyms), so "resume" finds "CV".
* **Reranking.** BM25 picks a shortlist and Claude reranks it by real usefulness,
  dropping candidates that only share wording.

### Three self-improvement loops

1. **Inner loop (iterative RAG).** Draft → judge → *retrieve again for the judge's complaints*
   → revise, until the score reaches the target or the round limit.
2. **Example memory.** Finished prompts are stored with their score. Retrieval rank is
   relevance × quality, so strong prompts become few-shot examples and weak ones never do.
3. **Lesson memory.** Lessons the drafter actually applied are credited with the run's
   score. Lessons with low utility are retired, and `reflect` mines recent runs and your
   feedback for new ones.

### Human check-ins (drift and hallucination guard)

A loop that only listens to its own judge can drift: it learns to please the judge rather
than you, and then saves those drifted prompts as "good examples". Four checkpoints prevent this:

| Checkpoint | When | Effect |
|---|---|---|
| Confirm intent | before drafting | Your corrected goal outranks Claude's reading, for both the drafter and the judge |
| Drift gate | after judging | Fidelity < 7 is flagged; if nobody vouches for it, it's kept out of the few-shot pool |
| Review | end of every run | Accept / edit / reject + 0-10 rating. Your rating overrides the judge, and your edits become 10/10 examples |
| Approve memory changes | during `reflect` | No lesson is added or retired without your yes |

## Setup

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
prompt-rag optimize "write a cover letter for a junior data analyst role" --verbose
prompt-rag optimize "..." --review flagged   # only stop when drift is detected
prompt-rag feedback 12 9 "great, but trim the intro"   # rate later
prompt-rag reflect                           # propose new lessons (you approve each)
prompt-rag stats                             # is it improving? first-half vs second-half scores
```

Options: `--model` (default `claude-opus-5-5`), `--retrieval-model` (e.g. `claude-haiku-5-5`
to make profiling and reranking cheaper), `--no-semantic` (BM25 only), `--no-iterative`,
`--target`, `--rounds`, `--db`.

As a library:

```python
from prompt_rag import ClaudeLLM, PromptOptimizer, Store
from prompt_rag.review import ConsoleReviewer

engine = PromptOptimizer(ClaudeLLM(), Store("prompt_rag.db"))
result = engine.optimize("summarize this contract for a non-lawyer", reviewer=ConsoleReviewer())
print(result.prompt, result.score, result.flagged, result.retrieval_trace)
```

Write your own `Reviewer` (see `prompt_rag/review.py`) to put the check-ins in a web UI or Slack.

## Cost

A run makes 1 profiling call, up to 1 rerank call, and 1–3 draft/judge rounds (2 calls per
round). On Opus 5.5 ($4 / $20 per million input/output tokens) that is very roughly
$0.25–$0.75 per run. This is an estimate, not a measurement. System prompts are cached
across calls. `--retrieval-model claude-haiku-5-5` and `--rounds 2` cut cost further.

## Layout

```
prompt_rag/
  engine.py     optimizer: retrieval, iterative loop, drift gate, learning
  review.py     human checkpoints (AutoReviewer, ConsoleReviewer)
  prompts.py    system prompts + structured-output schemas
  llm.py        Claude client (adaptive thinking, JSON schema output, refusal fallbacks)
  retrieval.py  BM25 + Jaccard dedupe
  store.py      SQLite memory (examples, lessons)
  seeds/lessons.json  starting prompt-engineering lessons
tests/          offline tests with a scripted fake LLM
```
