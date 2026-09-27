"""Unit tests for grounded answer generation."""

from typing import Any

import pytest

from backend.generation import (
    GenerationError,
    GenerationService,
    GroqGenerator,
    build_grounded_prompt,
)


class FakeGenerator:
    """Capture prompts and return a deterministic answer."""

    def __init__(self, answer: str = "The answer is in the document.") -> None:
        self.prompt = ""
        self.answer = answer

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.answer


def test_prompt_contains_question_context_and_abstention_instruction() -> None:
    prompt = build_grounded_prompt("What is AOA?", ["AOA verifies answers."])

    assert "What is AOA?" in prompt
    assert "AOA verifies answers." in prompt
    assert 'say exactly "I don\'t know"' in prompt
    assert "only the provided context" in prompt


def test_generation_returns_raw_answer_and_uses_retrieved_chunks() -> None:
    generator = FakeGenerator("  grounded answer  ")

    result = GenerationService(generator).generate(
        "What is AOA?", ["AOA verifies answers."]
    )

    assert result.answer == "grounded answer"
    assert "AOA verifies answers." in generator.prompt
    assert "What is AOA?" in generator.prompt


def test_empty_question_is_rejected() -> None:
    with pytest.raises(GenerationError, match="Question must contain"):
        GenerationService(FakeGenerator()).generate(" ", ["context"])


def test_empty_model_response_is_rejected() -> None:
    with pytest.raises(GenerationError, match="empty text"):
        GenerationService(FakeGenerator(" ")).generate("question", ["context"])


def test_groq_adapter_calls_configured_model() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "Groq answer"}}]}

    class Client:
        def __init__(self) -> None:
            self.arguments: dict[str, Any] = {}

        def post(self, path: str, **kwargs: Any) -> Response:
            self.arguments = {"path": path, **kwargs}
            return Response()

    client = Client()
    result = GroqGenerator(client=client, model="groq-test").generate("prompt")

    assert result == "Groq answer"
    assert client.arguments["path"] == "/chat/completions"
    assert client.arguments["json"]["model"] == "groq-test"
    assert client.arguments["json"]["messages"][1]["content"] == "prompt"
