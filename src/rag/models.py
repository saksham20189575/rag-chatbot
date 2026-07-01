"""Shared response types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass
class ChatResponse:
    type: Literal["answer", "refusal"]
    text: str
    citation_url: str
    last_updated: str
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    query_class: Literal["factual", "advisory", "performance", "pii", "out_of_scope"]
    scheme_id: str | None

    @property
    def class_name(self) -> str:
        return self.query_class


@dataclass
class DraftAnswer:
    text: str
    citation_url: str
    last_updated: str


@dataclass
class ValidationResult:
    ok: bool
    response: ChatResponse | None = None
    reason: str | None = None
