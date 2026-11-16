"""Provider-neutral chat-model integrations."""

from .openai_compatible import ChatCompletionClient, ChatCompletionError

__all__ = ["ChatCompletionClient", "ChatCompletionError"]
