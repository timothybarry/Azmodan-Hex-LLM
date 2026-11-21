"""Offline tests: a scripted fake LLM stands in for Claude."""

from __future__ import annotations

import pytest

from prompt_rag import Config, PromptOptimizer, Store
from prompt_rag.llm import strict_json_schema
from prompt_rag.prompts import Critique, Draft, Profile, Reflection, Rerank
from prompt_rag.retrieval import bm25_scores, jaccard


class FakeLLM:
    """Returns canned outputs per schema; critique scores come from a queue."""

    def __init__(self, scores=(9.0,), lessons_applied=(1, 2), rerank=None, new_lessons=(), fidelity=9,
                 weaknesses=("vague",), retire=()):
        self.fidelity = fidelity
        self.weaknesses = list(weaknesses)
        self.retire = list(retire)
        self.scores = list(scores)
        self.lessons_applied = list(lessons_applied)
        self.rerank = rerank
        self.new_lessons = list(new_lessons)
        self.calls: list[tuple[str, str]] = []

    def structured(self, system, user, schema, effort="medium"):
        self.calls.append((schema.__name__, user))
        if schema is Profile:
            return Profile(task_type="writing", domain="careers", intent="get a job", keywords=["resume", "cv"])
        if schema is Draft:
            n = sum(1 for c in self.calls if c[0] == "Draft")
            return Draft(prompt=f"draft {n}", rationale="r", lessons_applied=self.lessons_applied, assumptions=[])
        if schema is Critique:
            s = self.scores.pop(0) if self.scores else 9.0
            return Critique(clarity=8, specificity=8, context=8, output_spec=8, fidelity=self.fidelity,
                            overall=s, weaknesses=self.weaknesses, suggestions=["be specific"])
        if schema is Rerank:
            return Rerank(relevant_ids=self.rerank or [])
        if schema is Reflection:
            return Reflection(new_lessons=self.new_lessons, retire_lesson_ids=self.retire)
        raise AssertionError(schema)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db")


def count(llm, name):
    return sum(1 for c in llm.calls if c[0] == name)


def test_seeds_lessons(store):
    PromptOptimizer(FakeLLM(), store)
    assert store.lesson_count() >= 10


def test_stops_when_target_reached(store):
    llm = FakeLLM(scores=[9.0])
    r = PromptOptimizer(llm, store).optimize("write a resume")
    assert r.rounds == 1 and r.score == 9.0
    assert count(llm, "Profile") == 1 and count(llm, "Draft") == 1


def test_refines_and_keeps_best(store):
    llm = FakeLLM(scores=[5.0, 8.0, 7.0])
    r = PromptOptimizer(llm, store, Config(target_score=9.5, max_rounds=3)).optimize("write a resume")
    assert r.rounds == 3
    assert r.score == 8.0 and r.prompt == "draft 2"
    # The revision request carries the critique back to the drafter.
    revise = [u for name, u in llm.calls if name == "Draft"][1]
    assert "vague" in revise and "be specific" in revise


def test_stores_profile_and_credits_only_retrieved_lessons(store):
    llm = FakeLLM(scores=[9.0], lessons_applied=[1, 9999])
    r = PromptOptimizer(llm, store).optimize("write a resume")
    ex = store.get_example(r.example_id)
    assert "resume" in ex.profile
    assert 9999 not in r.lessons_used
    used = {l.id: l for l in store.lessons()}
    for lid in r.lessons_used:
        assert used[lid].uses == 1 and used[lid].score_sum == 9.0


def test_semantic_profile_matches_paraphrase(store):
    # Stored example says "CV"; new query says "resume". The Claude profile bridges them.
    store.add_example("polish my CV", "with-profile", 9.0, [], [], 1, profile="writing careers resume cv job")
    store.add_example("polish my curriculum vitae", "no-profile", 9.0, [], [], 1)
    eng = PromptOptimizer(FakeLLM(), store, Config(rerank=False))
    got = eng.retrieve_examples("improve my resume", eng.profile("improve my resume"))
    assert [e.prompt for e in got] == ["with-profile"]  # only the profiled example bridges CV <-> resume


def test_rerank_orders_and_filters(store):
    ids = [store.add_example(f"resume task {i}", "P", 8.0, [], [], 1, profile="resume") for i in range(5)]
    llm = FakeLLM(rerank=[ids[4], ids[2], 12345])
    eng = PromptOptimizer(llm, store, Config(k_examples=3))
    got = eng.retrieve_examples("resume", "resume")
    assert [e.id for e in got] == [ids[4], ids[2]]  # unknown id ignored, dropped ones excluded


def test_low_quality_examples_never_retrieved(store):
    store.add_example("write a resume", "bad", 3.0, [], [], 1)
    eng = PromptOptimizer(FakeLLM(), store, Config(semantic=False))
    assert eng.retrieve_examples("write a resume") == []


def test_quality_boost_prefers_better_example(store):
    store.add_example("write a resume", "ok", 6.5, [], [], 1)
    store.add_example("write a resume", "great", 9.5, [], [], 1)
    eng = PromptOptimizer(FakeLLM(), store, Config(semantic=False, rerank=False, k_examples=1))
    assert eng.retrieve_examples("write a resume")[0].prompt == "great"


def test_human_feedback_overrides_and_reattributes(store):
    llm = FakeLLM(scores=[9.0], lessons_applied=[1])
    eng = PromptOptimizer(llm, store)
    r = eng.optimize("write a resume")
    eng.feedback(r.example_id, 2.0, "way too long")
    ex = store.get_example(r.example_id)
    assert ex.score == 2.0 and ex.feedback == "way too long"
    lesson = next(l for l in store.lessons(active_only=False) if l.id == 1)
    assert lesson.score_sum == 2.0
    # A human-rated bad prompt drops out of the few-shot pool.
    assert eng.retrieve_examples("write a resume", "resume") == []


def test_bad_lessons_get_retired(store):
    eng = PromptOptimizer(FakeLLM(), store, Config(retire_after_uses=3))
    store.credit_lessons([1, 1, 1, 1], 1.0)
    eng._retire_weak_lessons()
    assert 1 not in {l.id for l in store.lessons()}


def test_reflect_adds_deduped_lessons(store):
    llm = FakeLLM(scores=[9.0, 9.0], new_lessons=[
        "Name the audience and their expertise level so tone and depth match the reader.",  # duplicate seed
        "Ask for a short summary at the top of long outputs.",
    ])
    eng = PromptOptimizer(llm, store)
    eng.optimize("a"); eng.optimize("b")
    before = store.lesson_count()
    refl = eng.reflect()
    assert refl.new_lessons == ["Ask for a short summary at the top of long outputs."]
    assert store.lesson_count() == before + 1


def test_strict_schema():
    s = strict_json_schema(Critique)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(Critique.model_fields)


def test_bm25_and_jaccard():
    s = bm25_scores("python sql", ["python code", "sql query in python", "cooking"])
    assert s[1] > s[0] > s[2] == 0
    assert jaccard("a b c", "a b c") == 0  # stopword/short tokens only
    assert jaccard("resume writing", "resume writing") == 1


# ---------------------------------------------------------------- iterative RAG + human checkpoints
from prompt_rag.review import AutoReviewer, Review  # noqa: E402


class ScriptedReviewer(AutoReviewer):
    def __init__(self, intent=None, review=None, keep_lessons=None, retire=None):
        self.intent, self._review, self.keep, self.retire = intent, review, keep_lessons, retire
        self.seen = None

    def confirm_intent(self, task, intent):
        return self.intent or intent

    def review(self, result):
        self.seen = result
        return self._review or Review("accept")

    def approve_lessons(self, lessons):
        return [l for l in lessons if self.keep is None or l in self.keep]

    def approve_retirements(self, lessons):
        return [l.id for l in lessons if self.retire is None or l.id in self.retire]


def test_iterative_rag_retrieves_for_weaknesses(store):
    # Weakness about output format should pull in the output-format lesson on round 2.
    llm = FakeLLM(scores=[5.0, 9.0], weaknesses=["no output format, length, or json structure specified"])
    eng = PromptOptimizer(llm, store, Config(k_lessons=1, target_score=8.5))
    r = eng.optimize("write a cover letter")
    assert len(r.retrieval_trace) == 2 and r.retrieval_trace[1]["lessons"]
    fmt = next(l.id for l in store.lessons() if "output format" in l.text)
    assert fmt in r.retrieval_trace[0]["lessons"] + r.retrieval_trace[1]["lessons"]
    round2 = [u for n, u in llm.calls if n == "Draft"][1]
    assert "output format explicitly" in round2


def test_iterative_off(store):
    r = PromptOptimizer(FakeLLM(scores=[5.0, 9.0]), store, Config(iterative=False)).optimize("x")
    assert len(r.retrieval_trace) == 1


def test_confirmed_intent_reaches_drafter_and_judge(store):
    llm = FakeLLM()
    rev = ScriptedReviewer(intent="Only a 3-line summary for executives")
    r = PromptOptimizer(llm, store).optimize("summarize this report", reviewer=rev)
    assert r.intent == "Only a 3-line summary for executives"
    for name in ("Draft", "Critique"):
        assert any("3-line summary" in u for n, u in llm.calls if n == name)
    assert "3-line summary" in store.get_example(r.example_id).profile


def test_drift_gate_quarantines_low_fidelity(store):
    eng = PromptOptimizer(FakeLLM(scores=[9.0], fidelity=4), store)
    r = eng.optimize("write a resume")
    assert r.flagged
    assert store.get_example(r.example_id).score < eng.cfg.min_example_score
    assert eng.retrieve_examples("write a resume", "resume") == []


def test_human_can_vouch_for_flagged(store):
    eng = PromptOptimizer(FakeLLM(scores=[9.0], fidelity=4), store)
    r = eng.optimize("write a resume", reviewer=ScriptedReviewer(review=Review("accept", score=8.0)))
    assert store.get_example(r.example_id).score == 8.0


def test_human_edit_becomes_gold_example(store):
    eng = PromptOptimizer(FakeLLM(), store)
    r = eng.optimize("write a resume", reviewer=ScriptedReviewer(review=Review("edit", edited_prompt="HUMAN")))
    ex = store.get_example(r.example_id)
    assert r.prompt == ex.prompt == "HUMAN" and ex.score == 10.0


def test_reject_excludes_from_pool(store):
    eng = PromptOptimizer(FakeLLM(), store)
    r = eng.optimize("write a resume", reviewer=ScriptedReviewer(review=Review("reject", comment="off target")))
    ex = store.get_example(r.example_id)
    assert ex.score == 0.0 and ex.feedback == "off target"


def test_reflection_requires_approval(store):
    llm = FakeLLM(scores=[9.0, 9.0], new_lessons=["Lesson about alpha tests.", "Lesson about beta docs."],
                  retire=[1, 2])
    eng = PromptOptimizer(llm, store)
    eng.optimize("a"); eng.optimize("b")
    refl = eng.reflect(reviewer=ScriptedReviewer(keep_lessons=["Lesson about beta docs."], retire=[2]))
    assert refl.new_lessons == ["Lesson about beta docs."] and refl.retire_lesson_ids == [2]
    active = {l.id for l in store.lessons()}
    assert 1 in active and 2 not in active
