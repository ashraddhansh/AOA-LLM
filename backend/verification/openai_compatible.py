"""Model-as-judge verification adapter for OpenAI-compatible chat models."""

import json
from collections.abc import Sequence
from typing import Any

from backend.llm.openai_compatible import ChatCompletionClient, ChatCompletionError

from .types import VerificationError, VerificationResult


class OpenAICompatibleVerifier:
    """Ask a configured chat model whether an answer is supported by evidence."""

    _SYSTEM_PROMPT = """You are a strict answer-faithfulness evaluator.
Judge only whether the answer's factual claims are supported by the supplied evidence.
Do not use outside knowledge. Treat a clear abstention as supported.
Return exactly one JSON object with keys: passed (boolean), score (number from 0 to 1),
and explanation (short string). Set passed=false if any material claim is unsupported."""

    def __init__(self, client: ChatCompletionClient) -> None:
        self._client = client

    def verify(
        self,
        query: str,
        answer: str,
        retrieved_chunks: Sequence[str],
    ) -> VerificationResult:
        """Evaluate the answer against the exact context used to generate it."""
        if not query.strip() or not answer.strip():
            raise VerificationError("Query and answer are required for verification.")
        payload = json.dumps(
            {
                "question": query,
                "answer": answer,
                "evidence": list(retrieved_chunks),
            },
            ensure_ascii=False,
        )
        try:
            raw = self._client.complete(
                [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": payload},
                ]
            )
        except ChatCompletionError as exc:
            raise VerificationError(f"Model verification failed: {exc}") from exc
        result = self._parse_result(raw)
        return VerificationResult(
            passed=result["passed"],
            verified=True,
            score=result["score"],
            explanation=result["explanation"],
        )

    @staticmethod
    def _parse_result(raw: str) -> dict[str, Any]:
        """Parse and validate the judge's JSON verdict, failing closed."""
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VerificationError("Verifier returned invalid JSON.") from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("passed"), bool):
            raise VerificationError("Verifier JSON must contain a boolean 'passed'.")
        score = decoded.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise VerificationError("Verifier score must be between 0 and 1.")
        explanation = decoded.get("explanation")
        if explanation is not None and not isinstance(explanation, str):
            raise VerificationError("Verifier explanation must be text or null.")
        return {
            "passed": decoded["passed"],
            "score": float(score) if score is not None else None,
            "explanation": explanation,
        }
