"""SQLite-backed memory: past optimization runs (examples) and learned lessons.

Both tables carry quality statistics that the engine updates over time; those
statistics are what make retrieval self-improving.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task          TEXT NOT NULL,          -- the user's raw request
    prompt        TEXT NOT NULL,          -- the optimized prompt produced
    judge_score   REAL NOT NULL,          -- 0-10 automatic score
    user_score    REAL,                   -- 0-10 human score (overrides judge when set)
    feedback      TEXT NOT NULL DEFAULT '',
    lesson_ids    TEXT NOT NULL DEFAULT '[]',
    example_ids   TEXT NOT NULL DEFAULT '[]',
    rounds        INTEGER NOT NULL DEFAULT 1,
    retrieved     INTEGER NOT NULL DEFAULT 0,   -- times used as a few-shot example
    profile       TEXT NOT NULL DEFAULT '',     -- Claude-written semantic profile used for retrieval
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS lessons (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    text          TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT 'seed',  -- seed | reflection
    uses          INTEGER NOT NULL DEFAULT 0,
    score_sum     REAL NOT NULL DEFAULT 0,       -- sum of final scores of runs that used it
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL
);
"""


@dataclass
class Example:
    id: int
    task: str
    prompt: str
    judge_score: float
    user_score: float | None
    feedback: str
    lesson_ids: list[int] = field(default_factory=list)
    example_ids: list[int] = field(default_factory=list)
    rounds: int = 1
    retrieved: int = 0
    profile: str = ""

    @property
    def score(self) -> float:
        """Effective quality: human rating wins over the automatic judge."""
        return self.user_score if self.user_score is not None else self.judge_score


@dataclass
class Lesson:
    id: int
    text: str
    source: str
    uses: int
    score_sum: float
    active: bool

    def utility(self, prior_mean: float = 7.0, prior_weight: float = 3.0) -> float:
        """Bayesian-smoothed mean score of runs that applied this lesson (0-10)."""
        return (self.score_sum + prior_mean * prior_weight) / (self.uses + prior_weight)


class Store:
    def __init__(self, path: str | Path = "prompt_rag.db"):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- examples ----------
    def add_example(
        self,
        task: str,
        prompt: str,
        judge_score: float,
        lesson_ids: list[int],
        example_ids: list[int],
        rounds: int,
        profile: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO examples (task, prompt, judge_score, lesson_ids, example_ids, rounds, profile, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task, prompt, judge_score, json.dumps(lesson_ids), json.dumps(example_ids), rounds, profile, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_example(self, example_id: int) -> Example | None:
        row = self.conn.execute("SELECT * FROM examples WHERE id = ?", (example_id,)).fetchone()
        return self._example(row) if row else None

    def all_examples(self) -> list[Example]:
        return [self._example(r) for r in self.conn.execute("SELECT * FROM examples ORDER BY id")]

    def recent_examples(self, limit: int) -> list[Example]:
        rows = self.conn.execute("SELECT * FROM examples ORDER BY id DESC LIMIT ?", (limit,))
        return [self._example(r) for r in rows]

    def set_user_feedback(self, example_id: int, score: float, feedback: str) -> None:
        self.conn.execute(
            "UPDATE examples SET user_score = ?, feedback = ? WHERE id = ?", (score, feedback, example_id)
        )
        self.conn.commit()

    def mark_retrieved(self, ids: list[int]) -> None:
        self.conn.executemany("UPDATE examples SET retrieved = retrieved + 1 WHERE id = ?", [(i,) for i in ids])
        self.conn.commit()

    @staticmethod
    def _example(r: sqlite3.Row) -> Example:
        return Example(
            id=r["id"],
            task=r["task"],
            prompt=r["prompt"],
            judge_score=r["judge_score"],
            user_score=r["user_score"],
            feedback=r["feedback"],
            lesson_ids=json.loads(r["lesson_ids"]),
            example_ids=json.loads(r["example_ids"]),
            rounds=r["rounds"],
            retrieved=r["retrieved"],
            profile=r["profile"],
        )

    # ---------- lessons ----------
    def add_lesson(self, text: str, source: str = "seed") -> int:
        cur = self.conn.execute(
            "INSERT INTO lessons (text, source, created_at) VALUES (?, ?, ?)", (text, source, time.time())
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def lessons(self, active_only: bool = True) -> list[Lesson]:
        q = "SELECT * FROM lessons" + (" WHERE active = 1" if active_only else "") + " ORDER BY id"
        return [
            Lesson(r["id"], r["text"], r["source"], r["uses"], r["score_sum"], bool(r["active"]))
            for r in self.conn.execute(q)
        ]

    def credit_lessons(self, ids: list[int], score: float) -> None:
        self.conn.executemany(
            "UPDATE lessons SET uses = uses + 1, score_sum = score_sum + ? WHERE id = ?",
            [(score, i) for i in ids],
        )
        self.conn.commit()

    def adjust_lesson_credit(self, ids: list[int], delta: float) -> None:
        """Re-attribute a score change (e.g. human rating replacing judge score) to lessons."""
        self.conn.executemany(
            "UPDATE lessons SET score_sum = score_sum + ? WHERE id = ?", [(delta, i) for i in ids]
        )
        self.conn.commit()

    def retire_lesson(self, lesson_id: int) -> None:
        self.conn.execute("UPDATE lessons SET active = 0 WHERE id = ?", (lesson_id,))
        self.conn.commit()

    def lesson_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0])
