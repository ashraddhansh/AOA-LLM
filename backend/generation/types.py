"""Typed contracts for grounded answer generation."""

from dataclasses import dataclass
from typing import Protocol


class GenerationError(Exception):
    """Raised when a grounded answer cannot be generated."""


@dataclass(frozen=True)
class GenerationResult:
    """Raw answer generated from the supplied question and context."""

    query: str
    answer: str


class TextGenerator(Protocol):
    """Protocol implemented by LLM providers."""

    def generate(self, prompt: str) -> str:
        """Generate raw text for a complete prompt."""
