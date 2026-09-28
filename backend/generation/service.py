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

    def __init__(
        self, generator: TextGenerator, web_generator: TextGenerator | None = None
    ) -> None:
        self._generator = generator
        self._web_generator = web_generator or generator

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

    def regenerate(
        self, query: str, retrieved_chunks: Sequence[str]
    ) -> GenerationResult:
        """Regenerate using direct-quotation and abstention instructions."""
        prompt = build_grounded_prompt(query, retrieved_chunks)
        strict_prompt = (
            "This is a verification retry. Use the same context only. "
            "Quote or closely paraphrase the relevant context, add no outside facts, "
            'and say exactly "I don\'t know" if support is missing.\n\n'
            f"{prompt}"
        )
        try:
            answer = self._generator.generate(strict_prompt).strip()
        except (OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(f"Strict answer regeneration failed: {exc}") from exc
        if not answer:
            raise GenerationError("Strict answer regeneration returned empty text.")
        return GenerationResult(query=query, answer=answer)

    def generate_from_web(
        self, query: str, search_results: Sequence[str]
    ) -> GenerationResult:
        """Generate an answer explicitly grounded in live web search results."""
        context = "\n\n".join(
            f"[Web result {index}]\n{result}"
            for index, result in enumerate(search_results, start=1)
        )
        prompt = (
            "Answer the question using only these live web search results. "
            "Do not claim this came from the user's uploaded documents. "
            'If unsupported, say exactly "I don\'t know".\n\n'
            f"Web results:\n{context}\n\nQuestion:\n{query.strip()}\n\nAnswer:"
        )
        try:
            answer = self._web_generator.generate(prompt).strip()
        except (OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(f"Web answer generation failed: {exc}") from exc
        if not answer:
            raise GenerationError("Web answer generation returned empty text.")
        return GenerationResult(query=query, answer=answer)
