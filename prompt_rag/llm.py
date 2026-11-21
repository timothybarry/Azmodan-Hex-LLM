"""Claude client wrapper returning validated structured outputs."""

from __future__ import annotations

import copy
from typing import Protocol, TypeVar

import anthropic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"
# Models that accept server-side refusal fallbacks (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class LLM(Protocol):
    def structured(self, system: str, user: str, schema: type[T], effort: str = "medium") -> T: ...


class RefusalError(RuntimeError):
    pass


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> structured-output schema (no extra keys, all fields required)."""
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node: object) -> None:
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


class ClaudeLLM:
    """Calls Claude with adaptive thinking, JSON-schema output, and refusal fallbacks."""

    def __init__(self, model: str = DEFAULT_MODEL, client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def structured(self, system: str, user: str, schema: type[T], effort: str = "medium") -> T:
        extra = {}
        if self.model in _FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            **extra,
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": strict_json_schema(schema)},
            },
            # Stable system prompt first so it can be cached across calls.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if response.stop_reason == "refusal":
            raise RefusalError(f"Model declined the request: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Response truncated at max_tokens")
        text = "".join(b.text for b in response.content if b.type == "text")
        return schema.model_validate_json(text)
