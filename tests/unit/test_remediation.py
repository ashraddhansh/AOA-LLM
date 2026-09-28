"""Tests for failure classification and answer remediation."""

from collections.abc import Sequence

from backend.classification.service import FailureType, classify_failure
from backend.generation.types import GenerationResult
from backend.remediation.service import RemediationService
from backend.verification.types import VerificationResult


class FakeSearch:
    """Return deterministic live-web context."""

    def search(self, query: str) -> tuple[str, ...]:
        return ("web result",)


def failed() -> VerificationResult:
    return VerificationResult(False, True, 0.1, "Unsupported claim")


def passed() -> VerificationResult:
    return VerificationResult(True, True, 0.95, "Supported")


def test_classifies_retrieval_gap_and_bad_generation() -> None:
    assert classify_failure(0.2, 0.5) is FailureType.RETRIEVAL_GAP
    assert classify_failure(0.8, 0.5) is FailureType.BAD_GENERATION


def test_retrieval_gap_uses_web_search_and_labels_answer() -> None:
    service = RemediationService(FakeSearch(), retrieval_threshold=0.5)

    result = service.remediate(
        "question",
        "bad",
        ("document context",),
        0.2,
        failed(),
        lambda query, chunks: GenerationResult(query, "web answer"),
        lambda query, chunks: GenerationResult(query, "retry"),
        lambda query, answer, chunks: passed(),
    )

    assert result.answer == "web answer"
    assert result.remediation == "web_search"
    assert result.source_label == "not from your documents"
    assert result.source_chunks == ("web result",)


def test_bad_generation_regenerates_against_same_chunks() -> None:
    service = RemediationService(FakeSearch(), retrieval_threshold=0.5)
    seen: list[Sequence[str]] = []

    def regenerate(query: str, chunks: Sequence[str]) -> GenerationResult:
        seen.append(chunks)
        return GenerationResult(query, "strict answer")

    result = service.remediate(
        "question",
        "bad",
        ("document context",),
        0.8,
        failed(),
        lambda query, chunks: GenerationResult(query, "web"),
        regenerate,
        lambda query, answer, chunks: passed(),
    )

    assert result.answer == "strict answer"
    assert result.remediation == "strict_regeneration"
    assert seen == [("document context",)]


def test_failed_remediation_abstains() -> None:
    service = RemediationService(FakeSearch(), retrieval_threshold=0.5)

    result = service.remediate(
        "question",
        "bad",
        ("document context",),
        0.8,
        failed(),
        lambda query, chunks: GenerationResult(query, "web"),
        lambda query, chunks: GenerationResult(query, "retry"),
        lambda query, answer, chunks: failed(),
    )

    assert result.answer == "insufficient reliable information"
    assert result.remediation == "abstain"
    assert result.verification.verified is False
