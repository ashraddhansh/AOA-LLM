"""Classify failed verification results before remediation."""

from enum import StrEnum


class FailureType(StrEnum):
    """Supported root causes for an unverified answer."""

    RETRIEVAL_GAP = "retrieval_gap"
    BAD_GENERATION = "bad_generation"


def classify_failure(top_score: float | None, threshold: float) -> FailureType:
    """Classify an unverified answer from its strongest retrieval score."""
    if top_score is None or top_score < threshold:
        return FailureType.RETRIEVAL_GAP
    return FailureType.BAD_GENERATION
