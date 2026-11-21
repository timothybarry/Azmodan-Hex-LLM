"""Command-line interface.

    prompt-rag optimize "write a cover letter for a data role"
    prompt-rag feedback 12 9 "great, but too long"
    prompt-rag reflect
    prompt-rag stats
"""

from __future__ import annotations

import argparse
import json
import sys

from .engine import Config, PromptOptimizer
from .llm import DEFAULT_MODEL, ClaudeLLM
from .review import AutoReviewer, ConsoleReviewer
from .store import Store


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="prompt-rag", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--db", default="prompt_rag.db", help="SQLite memory file (default: prompt_rag.db)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model for drafting and judging")
    p.add_argument("--retrieval-model", default=None, help="Model for profiling/reranking (default: --model)")
    p.add_argument("--no-semantic", action="store_true", help="Plain BM25 only, no Claude retrieval calls")
    p.add_argument(
        "--review",
        choices=["all", "flagged", "none"],
        default=None,
        help="Human check-ins: all (default in a terminal), flagged (only drift warnings), none",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("optimize", help="Turn a rough request into an optimized prompt")
    o.add_argument("task", nargs="?", help="The rough request (reads stdin if omitted)")
    o.add_argument("--context", default="", help="Extra context to consider")
    o.add_argument("--target", type=float, default=8.5, help="Stop refining at this judge score")
    o.add_argument("--rounds", type=int, default=3, help="Max draft/critique rounds")
    o.add_argument("--no-iterative", action="store_true", help="Retrieve once instead of every round")
    o.add_argument("--verbose", action="store_true", help="Show every round's score, weaknesses and retrievals")

    f = sub.add_parser("feedback", help="Rate a past result 0-10 (teaches the system)")
    f.add_argument("id", type=int)
    f.add_argument("score", type=float)
    f.add_argument("comment", nargs="?", default="")

    sub.add_parser("reflect", help="Distill new lessons from recent runs and feedback")
    sub.add_parser("stats", help="Show learning statistics")

    args = p.parse_args(argv)
    store = Store(args.db)
    cfg = Config(
        target_score=getattr(args, "target", 8.5),
        max_rounds=getattr(args, "rounds", 3),
        semantic=not args.no_semantic,
        rerank=not args.no_semantic,
        iterative=not getattr(args, "no_iterative", False),
    )
    mode = args.review or ("all" if sys.stdin.isatty() else "none")
    reviewer = AutoReviewer() if mode == "none" else ConsoleReviewer(review_only_flagged=(mode == "flagged"))

    if args.cmd == "stats":
        print(json.dumps(PromptOptimizer(None, store, cfg).stats(), indent=2))  # type: ignore[arg-type]
        return 0

    llm = ClaudeLLM(model=args.model)
    retrieval_llm = ClaudeLLM(model=args.retrieval_model, client=llm.client) if args.retrieval_model else llm
    engine = PromptOptimizer(llm, store, cfg, retrieval_llm=retrieval_llm)

    if args.cmd == "optimize":
        task = args.task or sys.stdin.read().strip()
        if not task:
            p.error("optimize needs a task argument or stdin input")
        r = engine.optimize(task, args.context, reviewer=reviewer)
        if args.verbose:
            for i, (_, c) in enumerate(r.history, 1):
                t = r.retrieval_trace[i - 1] if i - 1 < len(r.retrieval_trace) else {}
                print(
                    f"--- round {i}: {c.overall:.1f}/10 (fidelity {c.fidelity}) | retrieved lessons "
                    f"{t.get('lessons', [])} examples {t.get('examples', [])}",
                    file=sys.stderr,
                )
                for w in c.weaknesses:
                    print(f"    - {w}", file=sys.stderr)
        if r.flagged:
            print("WARNING: low fidelity to your request; review before use.", file=sys.stderr)
        print(r.prompt)
        print(
            f"\n[id {r.example_id} | score {r.score:.1f}/10 | rounds {r.rounds} | "
            f"lessons {r.lessons_used} | examples {r.examples_used}]",
            file=sys.stderr,
        )
        if r.assumptions:
            print("Assumptions: " + "; ".join(r.assumptions), file=sys.stderr)
        print(f"Rate it: prompt-rag feedback {r.example_id} <0-10> \"comment\"", file=sys.stderr)
    elif args.cmd == "feedback":
        engine.feedback(args.id, args.score, args.comment)
        print(f"Recorded feedback for #{args.id}.")
    elif args.cmd == "reflect":
        refl = engine.reflect(reviewer=reviewer)
        print(json.dumps(refl.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
