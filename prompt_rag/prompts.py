"""System prompts and structured-output schemas for each pipeline stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Draft(BaseModel):
    prompt: str = Field(description="The complete, ready-to-use prompt.")
    rationale: str = Field(description="One or two sentences on the key choices made.")
    lessons_applied: list[int] = Field(description="IDs of the provided lessons actually applied.")
    assumptions: list[str] = Field(description="Assumptions made where the request was ambiguous.")


class Critique(BaseModel):
    clarity: int = Field(description="0-10")
    specificity: int = Field(description="0-10")
    context: int = Field(description="0-10")
    output_spec: int = Field(description="0-10")
    fidelity: int = Field(description="0-10: preserves the user's actual intent without inventing scope")
    overall: float = Field(description="0-10 overall quality")
    weaknesses: list[str]
    suggestions: list[str]


class Reflection(BaseModel):
    new_lessons: list[str] = Field(description="New, general, reusable prompt-writing lessons.")
    retire_lesson_ids: list[int] = Field(description="IDs of lessons that the evidence shows are harmful.")


DRAFT_SYSTEM = """You are an expert prompt engineer. You turn a person's rough request into the \
prompt they should actually send to a capable LLM.

A strong prompt states the goal and why it matters, gives the context the model needs, names \
the audience, defines what a good output looks like (format, length, structure), and lists the \
real constraints. It reads like a clear brief to a smart colleague: direct, specific, no filler, \
no role-play preamble unless it adds something.

Stay faithful to the request. When a confirmed_intent is given, the person has verified it: \
it outranks your own reading of the request. Do not invent requirements the person did not imply; where the \
request is ambiguous, pick the most plausible reading, record it under assumptions, and, if a \
detail is genuinely needed, leave a clearly marked placeholder like [AUDIENCE] in the prompt.

You will be given lessons learned from past runs and high-scoring past examples retrieved for \
similar requests. Use them where they fit; ignore them where they don't. Report the IDs of the \
lessons you actually applied."""

CRITIQUE_SYSTEM = """You are a strict evaluator of LLM prompts. Score the candidate prompt for \
how well it would get a capable LLM to produce what the person asked for.

Score each dimension 0-10. Reserve 9-10 for prompts with nothing material left to improve. \
Penalize fidelity heavily if the prompt adds scope, invents facts or requirements, or drops \
something the person asked for. When a confirmed_intent is given, the person verified it; judge \
fidelity against it. Name any invented or drifted content explicitly in weaknesses. \
List concrete weaknesses and specific, actionable suggestions; if there are none, return empty lists."""

REFLECT_SYSTEM = """You maintain a library of prompt-writing lessons used by an automatic prompt \
optimizer. You are shown the current lessons with their measured utility, plus recent runs with \
their scores and any human feedback.

Find patterns that separate high-scoring prompts from low-scoring ones, and patterns in human \
feedback. Propose at most 3 new lessons that are general (apply across many requests), \
actionable, one sentence each, and not duplicates of existing lessons. Propose retiring a lesson \
only if the evidence shows it hurts results. Returning nothing is fine when the evidence is thin."""


def format_lessons(lessons) -> str:
    if not lessons:
        return "(none)"
    return "\n".join(f"[{l.id}] {l.text}" for l in lessons)


def format_examples(examples) -> str:
    if not examples:
        return "(none)"
    parts = []
    for e in examples:
        fb = f"\nHuman feedback: {e.feedback}" if e.feedback else ""
        parts.append(
            f'<example score="{e.score:.1f}">\n<request>{e.task}</request>\n'
            f"<optimized_prompt>\n{e.prompt}\n</optimized_prompt>{fb}\n</example>"
        )
    return "\n\n".join(parts)


def draft_user(task: str, lessons, examples, extra_context: str = "", intent: str = "") -> str:
    ctx = f"\n\n<extra_context>\n{extra_context}\n</extra_context>" if extra_context else ""
    if intent:
        ctx += f"\n\n<confirmed_intent>\n{intent}\n</confirmed_intent>"
    return (
        f"<lessons>\n{format_lessons(lessons)}\n</lessons>\n\n"
        f"<past_examples>\n{format_examples(examples)}\n</past_examples>{ctx}\n\n"
        f"<request>\n{task}\n</request>\n\nWrite the optimized prompt for this request."
    )


def revise_user(
    task: str, lessons, examples, previous: str, critique: Critique, extra_context: str = "", intent: str = ""
) -> str:
    issues = "\n".join(f"- {w}" for w in critique.weaknesses) or "- (none listed)"
    fixes = "\n".join(f"- {s}" for s in critique.suggestions) or "- (none listed)"
    return (
        draft_user(task, lessons, examples, extra_context, intent)
        + f"\n\nA previous draft scored {critique.overall:.1f}/10.\n"
        f"<previous_draft>\n{previous}\n</previous_draft>\n"
        f"<weaknesses>\n{issues}\n</weaknesses>\n<suggestions>\n{fixes}\n</suggestions>\n"
        "Write an improved version that fixes these issues while staying faithful to the request."
    )


def critique_user(task: str, prompt: str, intent: str = "") -> str:
    ci = f"<confirmed_intent>\n{intent}\n</confirmed_intent>\n\n" if intent else ""
    return f"<request>\n{task}\n</request>\n\n{ci}<candidate_prompt>\n{prompt}\n</candidate_prompt>"


def reflect_user(lessons, runs) -> str:
    lesson_lines = "\n".join(
        f"[{l.id}] (utility {l.utility():.2f}, uses {l.uses}) {l.text}" for l in lessons
    ) or "(none)"
    run_lines = []
    for e in runs:
        fb = f" | human feedback: {e.feedback}" if e.feedback else ""
        run_lines.append(
            f'<run id="{e.id}" score="{e.score:.1f}" lessons="{e.lesson_ids}">\n'
            f"<request>{e.task}</request>\n<prompt>\n{e.prompt}\n</prompt>{fb}\n</run>"
        )
    return f"<lessons>\n{lesson_lines}\n</lessons>\n\n<recent_runs>\n" + "\n".join(run_lines) + "\n</recent_runs>"


# ---------------------------------------------------------------- semantic retrieval
# Claude has no embeddings endpoint, so Claude does the "embedding" job in two ways:
# it writes a keyword-rich semantic profile of each request (indexed with BM25, so
# synonyms and paraphrases match), and it reranks the shortlisted candidates.


class Profile(BaseModel):
    task_type: str = Field(description="e.g. writing, coding, analysis, extraction, brainstorming, planning")
    domain: str = Field(description="Subject area, e.g. marketing, data engineering, legal")
    intent: str = Field(description="One sentence: what the person ultimately wants to achieve")
    keywords: list[str] = Field(description="8-15 search terms incl. synonyms and related concepts")


class Rerank(BaseModel):
    relevant_ids: list[int] = Field(
        description="IDs of candidates useful as references, most relevant first. Omit irrelevant ones."
    )


PROFILE_SYSTEM = """You index requests for a semantic search system. Given a request someone \
wants turned into an LLM prompt, describe it so that similar requests worded differently will \
match: classify it and list search keywords, including synonyms, related concepts, and the \
deliverable type."""

RERANK_SYSTEM = """You rank past prompt-optimization examples by how useful each would be as a \
reference when writing a prompt for a new request. Useful means a similar task type, deliverable, \
or domain, such that its structure or techniques transfer. Surface wording overlap alone is not \
enough. Return only the IDs of useful candidates, most useful first."""


def profile_text(p: Profile, intent: str = "") -> str:
    """Indexable text; a human-corrected intent replaces Claude's guess."""
    return f"{p.task_type} | {p.domain} | {intent or p.intent} | {' '.join(p.keywords)}"


def rerank_user(task: str, candidates) -> str:
    items = "\n".join(
        f'<candidate id="{e.id}">\n<request>{e.task}</request>\n<profile>{e.profile}</profile>\n</candidate>'
        for e in candidates
    )
    return f"<new_request>\n{task}\n</new_request>\n\n<candidates>\n{items}\n</candidates>"
