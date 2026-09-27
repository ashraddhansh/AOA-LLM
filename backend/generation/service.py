"""Application service for document-grounded answer generation."""

from collections.abc import Sequence

from .types import GenerationError, GenerationResult, TextGenerator


def build_grounded_prompt(query: str, retrieved_chunks: Sequence[str]) -> str:
    """Build the LLD-specified prompt from a question and retrieved context."""
    if not query.strip():
        raise GenerationError(
            "Question must contain at least one non-whitespace character."
        )
    context = "\n\n".join(
        f"[Context {index}]\n{chunk.strip()}"
        for index, chunk in enumerate(retrieved_chunks, start=1)
        if chunk.strip()
    )
    if not context:
        context = "[No retrieved context]"
    return (
        "Answer the question using only the provided context. "
        'If the answer is not present in the context, say exactly "I don\'t know". '
        "Do not use outside knowledge or invent facts.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query.strip()}\n\n"
        "Answer:"
    )


class GenerationService:
    """Generate a raw, context-grounded answer without verification."""

    def __init__(self, generator: TextGenerator) -> None:
        self._generator = generator

    def generate(self, query: str, retrieved_chunks: Sequence[str]) -> GenerationResult:
        """Generate an answer from exactly the chunks supplied by retrieval."""
        prompt = build_grounded_prompt(query, retrieved_chunks)
        try:
            answer = self._generator.generate(prompt).strip()
        except (OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(f"Answer generation failed: {exc}") from exc
        if not answer:
            raise GenerationError("Answer generator returned empty text.")
        return GenerationResult(query=query, answer=answer)
