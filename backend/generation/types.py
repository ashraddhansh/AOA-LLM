"""Typed contracts for grounded answer generation."""

from collections.abc import Sequence
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


class GenerationPipeline(Protocol):
    """Interface required by remediation orchestration."""

    def generate(self, query: str, retrieved_chunks: Sequence[str]) -> GenerationResult:
        """Generate an answer from context."""

    def regenerate(
        self, query: str, retrieved_chunks: Sequence[str]
    ) -> GenerationResult:
        """Generate a stricter retry from the same context."""

    def generate_from_web(
        self, query: str, search_results: Sequence[str]
    ) -> GenerationResult:
        """Generate from live web search context."""
