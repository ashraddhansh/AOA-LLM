"""Unit tests for Patronus Lynx verification."""

from typing import Any

import pytest

from backend.verification import LynxVerifier, VerificationError


class FakeResult:
    """Minimal Patronus result shape."""

    pass_ = True
    score = 0.93
    explanation = "The answer is supported by the context."


class FakeEvaluator:
    """Capture the exact fields sent to Lynx."""

    def __init__(self) -> None:
        self.arguments: dict[str, Any] = {}

    def evaluate(self, **kwargs: Any) -> FakeResult:
        self.arguments = kwargs
        return FakeResult()


def test_lynx_verifier_sends_exact_generation_inputs() -> None:
    evaluator = FakeEvaluator()

    result = LynxVerifier(evaluator=evaluator).verify(
        "What is AOA?",
        "AOA verifies answers.",
        ["AOA is a RAG system."],
    )

    assert result.passed is True
    assert result.verified is True
    assert result.score == 0.93
    assert evaluator.arguments == {
        "task_input": "What is AOA?",
        "task_output": "AOA verifies answers.",
        "task_context": ["AOA is a RAG system."],
    }


def test_lynx_missing_pass_result_is_explicit_failure() -> None:
    class IncompleteEvaluator:
        def evaluate(self, **kwargs: Any) -> object:
            return object()

    with pytest.raises(VerificationError, match="no pass/fail"):
        LynxVerifier(evaluator=IncompleteEvaluator()).verify(
            "question", "answer", ["context"]
        )


def test_lynx_empty_inputs_are_rejected() -> None:
    with pytest.raises(VerificationError, match="required"):
        LynxVerifier(evaluator=FakeEvaluator()).verify("", "answer", ["context"])


def test_lynx_initializes_sdk_before_creating_remote_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, str]] = []
    monkeypatch.setenv("PATRONUS_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.verification.lynx.patronus.init",
        lambda **kwargs: calls.append(kwargs),
    )

    verifier = LynxVerifier()

    assert verifier is not None
    assert calls == [{"project_name": "AOA", "app": "rag-api", "api_key": "test-key"}]
