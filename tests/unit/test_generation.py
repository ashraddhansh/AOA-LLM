"""Unit tests for grounded answer generation."""

from typing import Any

import pytest

from backend.generation import (
    GeminiGenerator,
    GenerationError,
    GenerationService,
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


def test_gemini_adapter_calls_configured_model() -> None:
    class Response:
        text = "Gemini answer"

    class Models:
        def __init__(self) -> None:
            self.arguments: dict[str, Any] = {}

        def generate_content(self, **kwargs: Any) -> Response:
            self.arguments = kwargs
            return Response()

    class Client:
        def __init__(self) -> None:
            self.models = Models()

    client = Client()
    result = GeminiGenerator(client=client, model="gemini-test").generate("prompt")

    assert result == "Gemini answer"
    assert client.models.arguments == {
        "model": "gemini-test",
        "contents": "prompt",
    }
