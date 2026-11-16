"""LangGraph workflow for retrieval, answer verification, and remediation."""

from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from backend.classification.service import FailureType, classify_failure, is_abstention
from backend.generation.types import GenerationError, GenerationPipeline
from backend.remediation.serpapi import SearchError
from backend.retrieval.types import RetrievalError, RetrievalResult
from backend.verification.types import VerificationError, VerificationResult, Verifier

ABSTENTION_ANSWER = "I could not find reliable information to answer this."


class RetrievalPipeline(Protocol):
    """Retrieval service required by the question graph."""

    def retrieve(
        self, query: str, filters: dict[str, str | int] | None = None
    ) -> RetrievalResult:
        """Retrieve context and similarity scores."""


class WebSearch(Protocol):
    """Search operation required by optional web remediation."""

    def search(self, query: str) -> Sequence[str]:
        """Return web snippets suitable as answer context."""


class QuestionState(TypedDict, total=False):
    """Values passed between nodes for one question request."""

    query: str
    chat_id: str
    web_search: bool
    retrieved_chunks: Sequence[str]
    retrieval_scores: Sequence[float]
    draft_answer: str
    web_chunks: Sequence[str]
    verification: VerificationResult
    failure_type: FailureType
    answer: str
    source_chunks: Sequence[str]
    source_scores: Sequence[float]
    source_label: str
    remediation: str | None
    remediation_error: str | None


class QuestionGraphRunner(Protocol):
    """Compiled graph interface consumed by the API layer."""

    def invoke(self, input: QuestionState) -> QuestionState:
        """Run the question graph to its final state."""


def build_question_graph(
    retrieval: RetrievalPipeline,
    generation: GenerationPipeline,
    verification: Verifier,
    web_search: WebSearch | None,
    retrieval_threshold: float,
) -> Any:
    """Compile the single-question workflow around the existing services."""

    def retrieve(state: QuestionState) -> dict[str, object]:
        result = retrieval.retrieve(
            state["query"], filters={"chat_id": state["chat_id"]}
        )
        return {
            "retrieved_chunks": result.retrieved_chunks,
            "retrieval_scores": result.retrieval_scores,
        }

    def generate(state: QuestionState) -> dict[str, object]:
        result = generation.generate(state["query"], state["retrieved_chunks"])
        return {"draft_answer": result.answer}

    def verify(state: QuestionState) -> dict[str, object]:
        result = verification.verify(
            state["query"], state["draft_answer"], state["retrieved_chunks"]
        )
        return {"verification": result}

    def classify(state: QuestionState) -> dict[str, object]:
        draft = state["draft_answer"]
        top_score = (
            state["retrieval_scores"][0] if state["retrieval_scores"] else None
        )
        failure_type = (
            FailureType.RETRIEVAL_GAP
            if is_abstention(draft)
            else classify_failure(top_score, retrieval_threshold)
        )
        return {"failure_type": failure_type}

    def regenerate(state: QuestionState) -> dict[str, object]:
        try:
            result = generation.regenerate(
                state["query"], state["retrieved_chunks"]
            )
        except GenerationError as exc:
            return {"remediation_error": str(exc)}
        return {"draft_answer": result.answer}

    def verify_regenerated(state: QuestionState) -> dict[str, object]:
        try:
            verdict = verification.verify(
                state["query"], state["draft_answer"], state["retrieved_chunks"]
            )
        except VerificationError as exc:
            return {"remediation_error": str(exc)}
        return {"verification": verdict}

    def search_web(state: QuestionState) -> dict[str, object]:
        if web_search is None:
            return {"remediation_error": "Web search is not configured."}
        try:
            return {"web_chunks": tuple(web_search.search(state["query"]))}
        except SearchError as exc:
            return {"remediation_error": str(exc)}

    def generate_from_web(state: QuestionState) -> dict[str, object]:
        try:
            result = generation.generate_from_web(
                state["query"], state["web_chunks"]
            )
        except GenerationError as exc:
            return {"remediation_error": str(exc)}
        return {"draft_answer": result.answer}

    def verify_web(state: QuestionState) -> dict[str, object]:
        try:
            verdict = verification.verify(
                state["query"], state["draft_answer"], state["web_chunks"]
            )
        except VerificationError as exc:
            return {"remediation_error": str(exc)}
        return {"verification": verdict}

    def answer_from_documents(state: QuestionState) -> dict[str, object]:
        return {
            "answer": state["draft_answer"],
            "source_chunks": state["retrieved_chunks"],
            "source_scores": state["retrieval_scores"],
            "source_label": "document-grounded",
            "remediation": None,
        }

    def answer_from_web(state: QuestionState) -> dict[str, object]:
        return {
            "answer": state["draft_answer"],
            "source_chunks": state["web_chunks"],
            "source_scores": tuple(0.0 for _ in state["web_chunks"]),
            "source_label": "not found in your documents",
            "remediation": "web_search",
        }

    def answer_after_regeneration(state: QuestionState) -> dict[str, object]:
        return {
            **answer_from_documents(state),
            "remediation": "strict_regeneration",
        }

    def abstain(state: QuestionState) -> dict[str, object]:
        previous = state.get("verification")
        web_chunks = state.get("web_chunks", ())
        use_web_context = bool(web_chunks)
        return {
            "answer": ABSTENTION_ANSWER,
            "verification": VerificationResult(
                passed=False,
                verified=False,
                score=previous.score if previous else None,
                explanation="No generated answer passed faithfulness verification.",
            ),
            "source_chunks": (
                web_chunks if use_web_context else state.get("retrieved_chunks", ())
            ),
            "source_scores": (
                tuple(0.0 for _ in web_chunks)
                if use_web_context
                else state.get("retrieval_scores", ())
            ),
            "source_label": "not found in your documents"
            if state.get("failure_type") is FailureType.RETRIEVAL_GAP
            else "document-grounded",
            "remediation": "abstain",
            "remediation_error": state.get("remediation_error"),
        }

    def after_verification(state: QuestionState) -> str:
        result = state["verification"]
        if result.passed and not (
            state["web_search"] and is_abstention(state["draft_answer"])
        ):
            return "answer"
        return "classify"

    def after_classification(state: QuestionState) -> str:
        failure_type = state["failure_type"]
        if failure_type is FailureType.BAD_GENERATION:
            return "regenerate"
        if state["web_search"]:
            return "search_web"
        return "abstain"

    def after_regeneration(state: QuestionState) -> str:
        result = state["verification"]
        if not state.get("remediation_error") and result.passed:
            return "answer"
        return "abstain"

    def after_web_generation(state: QuestionState) -> str:
        result = state.get("verification")
        if (
            not state.get("remediation_error")
            and result is not None
            and result.passed
            and not is_abstention(state["draft_answer"])
        ):
            return "answer"
        return "abstain"

    builder = StateGraph(QuestionState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_node("verify", verify)
    builder.add_node("classify", classify)
    builder.add_node("regenerate", regenerate)
    builder.add_node("verify_regenerated", verify_regenerated)
    builder.add_node("search_web", search_web)
    builder.add_node("generate_from_web", generate_from_web)
    builder.add_node("verify_web", verify_web)
    builder.add_node("answer_from_documents", answer_from_documents)
    builder.add_node("answer_after_regeneration", answer_after_regeneration)
    builder.add_node("answer_from_web", answer_from_web)
    builder.add_node("abstain", abstain)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "verify")
    builder.add_conditional_edges(
        "verify",
        after_verification,
        {"answer": "answer_from_documents", "classify": "classify"},
    )
    builder.add_conditional_edges(
        "classify",
        after_classification,
        {
            "regenerate": "regenerate",
            "search_web": "search_web",
            "abstain": "abstain",
        },
    )
    builder.add_conditional_edges(
        "regenerate",
        lambda state: "abstain" if state.get("remediation_error") else "verify",
        {"verify": "verify_regenerated", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "verify_regenerated",
        after_regeneration,
        {"answer": "answer_after_regeneration", "abstain": "abstain"},
    )
    def after_search(state: QuestionState) -> str:
        if state.get("remediation_error") or not state.get("web_chunks"):
            return "abstain"
        return "generate"

    builder.add_conditional_edges(
        "search_web",
        after_search,
        {"generate": "generate_from_web", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "generate_from_web",
        lambda state: "verify" if not state.get("remediation_error") else "abstain",
        {"verify": "verify_web", "abstain": "abstain"},
    )
    builder.add_conditional_edges(
        "verify_web",
        after_web_generation,
        {"answer": "answer_from_web", "abstain": "abstain"},
    )
    builder.add_edge("answer_from_documents", END)
    builder.add_edge("answer_after_regeneration", END)
    builder.add_edge("answer_from_web", END)
    builder.add_edge("abstain", END)
    return builder.compile()
