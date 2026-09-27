"""Typed contracts for answer verification."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class VerificationError(Exception):
    """Raised when a verifier cannot complete an evaluation."""


@dataclass(frozen=True)
class VerificationResult:
    """Faithfulness result for one generated answer."""

    passed: bool
    verified: bool
    score: float | None
    explanation: str | None


class Verifier(Protocol):
    """Protocol for answer-faithfulness evaluators."""

    def verify(
        self,
        query: str,
        answer: str,
        retrieved_chunks: Sequence[str],
    ) -> VerificationResult:
        """Evaluate answer faithfulness against the exact retrieved context."""
