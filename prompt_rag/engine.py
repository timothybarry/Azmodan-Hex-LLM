"""The self-improving prompt optimizer.

Three feedback loops make it improve with use:

1. **Inner loop (per request):** draft -> critique -> revise until the judge
   score reaches ``target_score`` or ``max_rounds`` is hit.
2. **Example memory:** every finished prompt is stored with its score. Retrieval
   ranks past examples by relevance x quality, so good prompts get reused as
   few-shot examples and bad ones fade. Human ratings override judge scores.
3. **Lesson memory:** lessons the drafter applied are credited with the run's
   final score; low-utility lessons are retired. ``reflect()`` mines recent
   runs and human feedback for new lessons.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import resources

from .llm import LLM
from .prompts import (
    CRITIQUE_SYSTEM,
    DRAFT_SYSTEM,
    PROFILE_SYSTEM,
    REFLECT_SYSTEM,
    RERANK_SYSTEM,
    Critique,
    Draft,
    Profile,
    Reflection,
    Rerank,
    critique_user,
    draft_user,
    profile_text,
    reflect_user,
    rerank_user,
    revise_user,
)
from .retrieval import bm25_scores, jaccard
from .review import AutoReviewer, Reviewer
from .store import Example, Lesson, Store


@dataclass
class Config:
    k_examples: int = 3
    k_lessons: int = 6
    target_score: float = 8.5
    max_rounds: int = 3
    min_example_score: float = 6.0  # never few-shot a prompt scored below this
    quality_weight: float = 1.0  # how much stored quality boosts retrieval rank
    exploration: float = 0.3  # UCB bonus for rarely-used lessons
    retire_after_uses: int = 5
    retire_below_utility: float = 5.5
    dedupe_threshold: float = 0.6  # jaccard similarity above which a new lesson is a duplicate
    draft_effort: str = "high"
    judge_effort: str = "medium"
    # Claude-powered semantic retrieval (stands in for an embedding model)
    semantic: bool = True  # profile requests with Claude and index the profiles
    rerank: bool = True  # let Claude rerank the BM25 shortlist
    rerank_pool: int = 12  # shortlist size sent to the reranker
    retrieval_effort: str = "low"
    # Iterative RAG: each revision round re-retrieves using the critique
    iterative: bool = True
    iter_lessons: int = 3  # extra lessons per round
    iter_examples: int = 1  # extra examples per round
    # Drift guard: below this judge fidelity, a result is flagged for human review
    min_fidelity: int = 7


@dataclass
class Result:
    example_id: int
    prompt: str
    score: float
    rounds: int
    rationale: str
    assumptions: list[str]
    lessons_used: list[int]
    examples_used: list[int]
    history: list[tuple[str, Critique]] = field(default_factory=list)
    intent: str = ""
    fidelity: int = 10
    flagged: bool = False
    retrieval_trace: list[dict] = field(default_factory=list)


class PromptOptimizer:
    def __init__(
        self,
        llm: LLM,
        store: Store,
        config: Config | None = None,
        seed: bool = True,
        retrieval_llm: LLM | None = None,
    ):
        self.llm = llm
        self.retrieval_llm = retrieval_llm or llm
        self.store = store
        self.cfg = config or Config()
        if seed and self.store.lesson_count() == 0:
            seeds = json.loads(resources.files("prompt_rag").joinpath("seeds/lessons.json").read_text())
            for text in seeds:
                self.store.add_lesson(text, source="seed")

    # ------------------------------------------------------------------ retrieval
    def _profile(self, task: str) -> Profile | None:
        if not self.cfg.semantic:
            return None
        return self.retrieval_llm.structured(PROFILE_SYSTEM, task, Profile, self.cfg.retrieval_effort)

    def profile(self, task: str) -> str:
        """Claude-written semantic profile of a request, as indexable text ("" when off)."""
        p = self._profile(task)
        return profile_text(p) if p else ""

    def retrieve_examples(self, task: str, query_profile: str = "", rerank: bool | None = None) -> list[Example]:
        pool = [e for e in self.store.all_examples() if e.score >= self.cfg.min_example_score]
        if not pool:
            return []
        # Stage 1: lexical shortlist over request text + Claude's semantic profile.
        query = f"{task} {query_profile}"
        rel = bm25_scores(query, [f"{e.task} {e.profile}" for e in pool])
        top = max(rel) or 1.0
        scored = [(r / top, e) for r, e in zip(rel, pool) if r > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        shortlist = scored[: self.cfg.rerank_pool]

        # Stage 2: Claude reranks the shortlist by real usefulness.
        do_rerank = self.cfg.rerank if rerank is None else rerank
        if do_rerank and len(shortlist) > self.cfg.k_examples:
            by_id = {e.id: e for _, e in shortlist}
            order = self.retrieval_llm.structured(
                RERANK_SYSTEM, rerank_user(task, by_id.values()), Rerank, self.cfg.retrieval_effort
            ).relevant_ids
            order = [i for i in dict.fromkeys(order) if i in by_id]
            n = len(order) or 1
            # Rank position replaces lexical relevance; candidates Claude dropped are excluded.
            shortlist = [(1 - pos / n, by_id[i]) for pos, i in enumerate(order)]

        # Stage 3: boost by stored quality, so better prompts are reused more.
        ranked = [(rel_ * (1 + self.cfg.quality_weight * e.score / 10.0), e) for rel_, e in shortlist]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in ranked[: self.cfg.k_examples]]

    def retrieve_lessons(self, task: str, query_profile: str = "") -> list[Lesson]:
        lessons = self.store.lessons()
        if not lessons:
            return []
        rel = bm25_scores(f"{task} {query_profile}", [l.text for l in lessons])
        top = max(rel) or 1.0
        total_uses = sum(l.uses for l in lessons) + 1
        ranked = []
        for l, r in zip(lessons, rel):
            utility = l.utility() / 10.0
            # Upper-confidence bonus keeps new lessons from being starved by established ones.
            bonus = self.cfg.exploration * math.sqrt(math.log(total_uses + 1) / (l.uses + 1))
            ranked.append((0.5 * (r / top) + utility + bonus, l))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [l for _, l in ranked[: self.cfg.k_lessons]]

    # ------------------------------------------------------------------ main loop
    def optimize(self, task: str, extra_context: str = "", reviewer: Reviewer | None = None) -> Result:
        reviewer = reviewer or AutoReviewer()

        # Checkpoint 1: anchor on a human-confirmed reading of the request.
        profile = self._profile(task)
        intent = reviewer.confirm_intent(task, profile.intent if profile else "")
        query_profile = profile_text(profile, intent) if profile else ""

        lessons = self.retrieve_lessons(task, query_profile)
        examples = self.retrieve_examples(task, query_profile)
        trace = [{"round": 1, "lessons": [l.id for l in lessons], "examples": [e.id for e in examples]}]

        draft = self.llm.structured(
            DRAFT_SYSTEM, draft_user(task, lessons, examples, extra_context, intent), Draft, self.cfg.draft_effort
        )
        best_draft, best_critique = draft, self._critique(task, draft.prompt, intent)
        history = [(draft.prompt, best_critique)]
        applied = set(draft.lessons_applied) & {l.id for l in lessons}

        rounds = 1
        while best_critique.overall < self.cfg.target_score and rounds < self.cfg.max_rounds:
            rounds += 1
            # Iterative RAG: re-query memory with the judge's complaints about the current draft.
            if self.cfg.iterative:
                lessons, examples, added = self._expand_context(task, query_profile, best_critique, lessons, examples)
                trace.append({"round": rounds, **added})
            draft = self.llm.structured(
                DRAFT_SYSTEM,
                revise_user(task, lessons, examples, best_draft.prompt, best_critique, extra_context, intent),
                Draft,
                self.cfg.draft_effort,
            )
            critique = self._critique(task, draft.prompt, intent)
            history.append((draft.prompt, critique))
            applied |= set(draft.lessons_applied) & {l.id for l in lessons}
            if critique.overall > best_critique.overall:
                best_draft, best_critique = draft, critique

        lesson_ids = sorted(applied)
        example_ids = [e.id for e in examples]
        judge_score = float(best_critique.overall)
        result = Result(
            example_id=-1,
            prompt=best_draft.prompt,
            score=judge_score,
            rounds=rounds,
            rationale=best_draft.rationale,
            assumptions=best_draft.assumptions,
            lessons_used=lesson_ids,
            examples_used=example_ids,
            history=history,
            intent=intent,
            fidelity=best_critique.fidelity,
            flagged=best_critique.fidelity < self.cfg.min_fidelity,
            retrieval_trace=trace,
        )

        # Checkpoint 2: human review. The human's verdict outranks the judge.
        review = reviewer.review(result)
        stored_score = judge_score
        if result.flagged and review.score is None:
            # Drift gate: an unvouched low-fidelity prompt is kept out of the few-shot pool.
            stored_score = min(judge_score, self.cfg.min_example_score - 0.1)
        if review.action == "edit" and review.edited_prompt:
            result.prompt = review.edited_prompt
            stored_score = 10.0 if review.score is None else review.score
        elif review.action == "reject" and review.score is None:
            review.score = 0.0

        result.example_id = self.store.add_example(
            task, result.prompt, stored_score, lesson_ids, example_ids, rounds, profile=query_profile
        )
        self.store.credit_lessons(lesson_ids, stored_score)
        if review.score is not None or review.comment:
            score = review.score if review.score is not None else stored_score
            self.feedback(result.example_id, score, review.comment)
            result.score = score
        self.store.mark_retrieved(example_ids)
        self._retire_weak_lessons()
        return result

    def _expand_context(self, task, query_profile, critique: Critique, lessons, examples):
        """Retrieve more lessons/examples targeted at the current draft's weaknesses."""
        need = " ".join(critique.weaknesses + critique.suggestions)
        have_l = {l.id for l in lessons}
        have_e = {e.id for e in examples}
        new_l = [l for l in self.retrieve_lessons(need) if l.id not in have_l][: self.cfg.iter_lessons]
        new_e = [
            e for e in self.retrieve_examples(f"{task} {need}", query_profile, rerank=False)
            if e.id not in have_e
        ][: self.cfg.iter_examples]
        return lessons + new_l, examples + new_e, {
            "lessons": [l.id for l in new_l], "examples": [e.id for e in new_e]
        }

    def _critique(self, task: str, prompt: str, intent: str = "") -> Critique:
        c = self.llm.structured(
            CRITIQUE_SYSTEM, critique_user(task, prompt, intent), Critique, self.cfg.judge_effort
        )
        c.overall = max(0.0, min(10.0, float(c.overall)))
        return c

    # ------------------------------------------------------------------ learning
    def feedback(self, example_id: int, score: float, comment: str = "") -> None:
        """Record a human 0-10 rating; it replaces the judge score everywhere downstream."""
        ex = self.store.get_example(example_id)
        if ex is None:
            raise KeyError(f"No example with id {example_id}")
        score = max(0.0, min(10.0, score))
        delta = score - ex.score
        self.store.set_user_feedback(example_id, score, comment)
        self.store.adjust_lesson_credit(ex.lesson_ids, delta)
        self._retire_weak_lessons()

    def reflect(self, window: int = 20, reviewer: Reviewer | None = None) -> Reflection:
        """Distill new lessons from recent runs and feedback; retire harmful ones.

        Checkpoints 3-4: with a reviewer, nothing enters or leaves long-term memory without approval.
        """
        reviewer = reviewer or AutoReviewer()
        runs = self.store.recent_examples(window)
        lessons = self.store.lessons()
        if len(runs) < 2:
            return Reflection(new_lessons=[], retire_lesson_ids=[])
        refl = self.llm.structured(REFLECT_SYSTEM, reflect_user(lessons, runs), Reflection, "high")

        by_id = {l.id: l for l in lessons}
        proposed = [by_id[i] for i in dict.fromkeys(refl.retire_lesson_ids) if i in by_id]
        approved_retire = reviewer.approve_retirements(proposed) if proposed else []
        for lid in approved_retire:
            self.store.retire_lesson(lid)
        refl.retire_lesson_ids = approved_retire

        existing = [l.text for l in self.store.lessons(active_only=False)]
        candidates = []
        for text in refl.new_lessons:
            text = text.strip()
            if text and all(jaccard(text, e) < self.cfg.dedupe_threshold for e in existing + candidates):
                candidates.append(text)
        added = reviewer.approve_lessons(candidates) if candidates else []
        for text in added:
            self.store.add_lesson(text, source="reflection")
        refl.new_lessons = added
        return refl

    def _retire_weak_lessons(self) -> None:
        for l in self.store.lessons():
            if l.uses >= self.cfg.retire_after_uses and l.utility() < self.cfg.retire_below_utility:
                self.store.retire_lesson(l.id)

    def stats(self) -> dict:
        examples = self.store.all_examples()
        lessons = self.store.lessons(active_only=False)
        scores = [e.score for e in examples]
        half = len(scores) // 2
        return {
            "runs": len(examples),
            "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
            "mean_score_first_half": round(sum(scores[:half]) / half, 2) if half else None,
            "mean_score_second_half": round(sum(scores[half:]) / (len(scores) - half), 2) if half else None,
            "human_rated": sum(e.user_score is not None for e in examples),
            "lessons_active": sum(l.active for l in lessons),
            "lessons_retired": sum(not l.active for l in lessons),
            "top_lessons": [
                {"id": l.id, "utility": round(l.utility(), 2), "uses": l.uses, "text": l.text}
                for l in sorted((l for l in lessons if l.active), key=lambda l: l.utility(), reverse=True)[:5]
            ],
        }
