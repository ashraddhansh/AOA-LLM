"""Remediation orchestration for unverified answers."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from backend.classification.service import FailureType
from backend.generation.types import GenerationError, GenerationResult
from backend.remediation.serpapi import SearchError, SerpApiSearch
from backend.verification.types import VerificationError, VerificationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemediationResult:
    """Final answer and verification state after remediation."""

    answer: str
    verification: VerificationResult
    failure_type: FailureType | None
    remediation: str | None
    source_label: str
    source_chunks: tuple[str, ...]
    error: str | None = None


class RemediationService:
    """Run strict regeneration or web-search remediation and abstain safely."""

    def __init__(
        self,
        search: SerpApiSearch,
        retrieval_threshold: float = 0.5,
    ) -> None:
        if not 0 <= retrieval_threshold <= 1:
            raise ValueError("retrieval_threshold must be between zero and one")
        self._search = search
        self._threshold = retrieval_threshold

    def remediate(
        self,
        query: str,
        answer: str,
        chunks: Sequence[str],
        top_score: float | None,
        verification: VerificationResult,
        generate: Callable[[str, Sequence[str]], GenerationResult],
        regenerate: Callable[[str, Sequence[str]], GenerationResult],
        verify: Callable[[str, str, Sequence[str]], VerificationResult],
        failure_type: FailureType | None = None,
    ) -> RemediationResult:
        """Remediate one failed answer, returning abstention if needed."""
        from backend.classification.service import classify_failure

        failure_type = failure_type or classify_failure(top_score, self._threshold)
        try:
            if failure_type is FailureType.BAD_GENERATION:
                retry = regenerate(query, chunks)
                retry_verification = verify(query, retry.answer, chunks)
                if retry_verification.passed:
                    return RemediationResult(
                        retry.answer,
                        retry_verification,
                        failure_type,
                        "strict_regeneration",
                        "uploaded documents",
                        tuple(chunks),
                    )
            else:
                web_chunks = self._search.search(query)
                web_answer = generate(query, web_chunks)
                web_verification = verify(query, web_answer.answer, web_chunks)
                if web_verification.passed:
                    return RemediationResult(
                        web_answer.answer,
                        web_verification,
                        failure_type,
                        "web_search",
                        "not from your documents",
                        tuple(web_chunks),
                    )
        except (GenerationError, SearchError, VerificationError) as exc:
            logger.exception("RAG remediation failed for query %r", query)
            return RemediationResult(
                "insufficient reliable information",
                VerificationResult(
                    False,
                    False,
                    None,
                    "Remediation failed, so the answer was withheld.",
                ),
                failure_type,
                "abstain",
                (
                    "not from your documents"
                    if failure_type is FailureType.RETRIEVAL_GAP
                    else "uploaded documents"
                ),
                tuple(chunks),
                str(exc),
            )
        return RemediationResult(
            "insufficient reliable information",
            VerificationResult(
                False, False, None, "Remediation did not produce a verified answer."
            ),
            failure_type,
            "abstain",
            (
                "not from your documents"
                if failure_type is FailureType.RETRIEVAL_GAP
                else "uploaded documents"
            ),
            tuple(chunks),
            "The remediation answer did not pass Lynx verification.",
        )
