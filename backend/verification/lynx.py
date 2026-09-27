"""Patronus Lynx faithfulness evaluator."""

import os
from collections.abc import Sequence
from typing import Any

from patronus.evals import RemoteEvaluator

from .types import VerificationError, VerificationResult


class LynxVerifier:
    """Verify generated answers with Patronus's Lynx hallucination evaluator."""

    def __init__(self, evaluator: Any | None = None) -> None:
        if evaluator is None and not os.getenv("PATRONUS_API_KEY"):
            raise VerificationError("PATRONUS_API_KEY is not configured.")
        self._evaluator = evaluator or RemoteEvaluator(
            "lynx",
            "patronus:hallucination",
            explain_strategy="always",
        )

    def verify(
        self,
        query: str,
        answer: str,
        retrieved_chunks: Sequence[str],
    ) -> VerificationResult:
        """Send exact generation inputs to Lynx and normalize its result."""
        if not query.strip() or not answer.strip():
            raise VerificationError("Query and answer are required for verification.")
        try:
            result = self._evaluator.evaluate(
                task_input=query,
                task_output=answer,
                task_context=list(retrieved_chunks),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise VerificationError(f"Lynx evaluation failed: {exc}") from exc

        passed = getattr(result, "pass_", None)
        if not isinstance(passed, bool):
            raise VerificationError("Lynx returned no pass/fail result.")
        score = getattr(result, "score", None)
        return VerificationResult(
            passed=passed,
            verified=True,
            score=float(score) if score is not None else None,
            explanation=getattr(result, "explanation", None),
        )
