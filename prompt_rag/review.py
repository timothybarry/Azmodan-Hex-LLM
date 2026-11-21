"""Human-in-the-loop checkpoints.

A self-improving loop that only listens to its own judge can drift: it learns
to please the judge rather than the person. These checkpoints anchor it to
human intent at the four places drift enters:

1. ``confirm_intent``  - before drafting: correct the system's reading of the request.
2. ``review``          - after drafting: accept / edit / reject, with a 0-10 rating.
3. ``approve_lessons`` - before reflection adds lessons to long-term memory.
4. ``approve_retirements`` - before reflection removes lessons.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from .engine import Result
    from .store import Lesson


@dataclass
class Review:
    action: Literal["accept", "edit", "reject"]
    score: float | None = None  # 0-10 human rating; None = trust the judge
    comment: str = ""
    edited_prompt: str | None = None


class Reviewer(Protocol):
    def confirm_intent(self, task: str, intent: str) -> str: ...
    def review(self, result: "Result") -> Review: ...
    def approve_lessons(self, lessons: list[str]) -> list[str]: ...
    def approve_retirements(self, lessons: list["Lesson"]) -> list[int]: ...


class AutoReviewer:
    """No human present: accept everything. Flagged results are still quarantined by the engine."""

    def confirm_intent(self, task, intent):
        return intent

    def review(self, result):
        return Review("accept")

    def approve_lessons(self, lessons):
        return lessons

    def approve_retirements(self, lessons):
        return [l.id for l in lessons]


class ConsoleReviewer:
    """Interactive terminal checkpoints. Prompts go to stderr so stdout stays clean."""

    def __init__(self, review_only_flagged: bool = False):
        self.review_only_flagged = review_only_flagged

    @staticmethod
    def _ask(msg: str) -> str:
        print(msg, end="", file=sys.stderr, flush=True)
        return sys.stdin.readline().strip()

    def confirm_intent(self, task, intent):
        if not intent:
            return intent
        print(f"\nI read your goal as:\n  {intent}", file=sys.stderr)
        fix = self._ask("Enter to confirm, or type a corrected goal: ")
        return fix or intent

    def review(self, result):
        if self.review_only_flagged and not result.flagged:
            return Review("accept")
        bar = "=" * 60
        print(f"\n{bar}\n{result.prompt}\n{bar}", file=sys.stderr)
        print(f"Judge: {result.score:.1f}/10, fidelity {result.fidelity}/10", file=sys.stderr)
        if result.flagged:
            print("WARNING: low fidelity - this may have drifted from your request:", file=sys.stderr)
            for w in result.history[-1][1].weaknesses:
                print(f"  - {w}", file=sys.stderr)
        if result.assumptions:
            print("Assumptions: " + "; ".join(result.assumptions), file=sys.stderr)
        choice = self._ask("[a]ccept / [e]dit / [r]eject? ").lower()[:1] or "a"
        edited = None
        if choice == "e":
            print("Paste the corrected prompt, end with a line containing only '.':", file=sys.stderr)
            lines = []
            for line in sys.stdin:
                if line.rstrip("\n") == ".":
                    break
                lines.append(line)
            edited = "".join(lines).strip() or None
        raw = self._ask("Rating 0-10 (Enter to skip): ")
        score = float(raw) if raw else None
        comment = self._ask("Comment (optional): ")
        action = {"e": "edit", "r": "reject"}.get(choice, "accept")
        return Review(action, score, comment, edited)

    def approve_lessons(self, lessons):
        kept = []
        for text in lessons:
            if self._ask(f"\nAdd lesson? \"{text}\" [y/N] ").lower().startswith("y"):
                kept.append(text)
        return kept

    def approve_retirements(self, lessons):
        return [
            l.id for l in lessons
            if self._ask(f"\nRetire lesson [{l.id}] \"{l.text}\" (utility {l.utility():.1f})? [y/N] ")
            .lower().startswith("y")
        ]
