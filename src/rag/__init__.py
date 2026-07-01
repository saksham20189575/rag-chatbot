"""Retrieval, classification, generation, and validation."""

from src.rag.models import ChatResponse

__all__ = ["ChatResponse", "answer"]


def __getattr__(name: str):
    if name == "answer":
        from src.rag.pipeline import answer

        return answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
