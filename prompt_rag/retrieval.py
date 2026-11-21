"""Dependency-free lexical retrieval (Okapi BM25).

The Claude API has no embeddings endpoint, so retrieval is local. BM25 is a
strong baseline for short prompt/task texts; swap in an embedding retriever by
implementing the same ``score(query, docs)`` interface.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have i in is it its me my of on or "
    "please that the this to was were will with you your can could would should "
    "want need some".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Return a BM25 score for each doc against ``query`` (same order as ``docs``)."""
    if not docs:
        return []
    tokenized = [tokenize(d) for d in docs]
    n = len(tokenized)
    avgdl = sum(len(t) for t in tokenized) / n or 1.0
    df: Counter[str] = Counter()
    for toks in tokenized:
        df.update(set(toks))
    q_terms = set(tokenize(query))
    scores = []
    for toks in tokenized:
        tf = Counter(toks)
        dl = len(toks) or 1
        s = 0.0
        for term in q_terms:
            f = tf.get(term)
            if not f:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
