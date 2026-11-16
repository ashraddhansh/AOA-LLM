"""Provider-neutral generation adapter for OpenAI-compatible chat models."""

from backend.llm.openai_compatible import ChatCompletionClient, ChatCompletionError

from .types import GenerationError


class OpenAICompatibleGenerator:
    """Generate text through a configured OpenAI-compatible model endpoint."""

    def __init__(self, client: ChatCompletionClient) -> None:
        self._client = client

    def generate(self, prompt: str) -> str:
        """Generate raw text for a complete prompt."""
        try:
            return self._client.complete([{"role": "user", "content": prompt}])
        except ChatCompletionError as exc:
            raise GenerationError(f"Answer generation failed: {exc}") from exc
