# Question graph and remediation

The question endpoint runs a compiled LangGraph workflow defined in
`backend/orchestration/question_graph.py`. Each request retrieves chat-scoped
Chroma context, generates an answer, and checks the answer against that exact
evidence.

If verification passes, the graph returns the answer and document sources. If
it fails, the graph classifies the failure using the strongest retrieval score.
A `bad_generation` can trigger one stricter regeneration against the same
chunks. A `retrieval_gap` can trigger SerpAPI search when
`SERPAPI_API_KEY` (or `WEB_SEARCH_API_KEY`) is configured. Web answers are
verified against the search snippets and labeled as web sourced. Failed
verification, empty search results, and provider errors lead to a safe
abstention.

Generation and verification use independent OpenAI Chat Completions-compatible
model configurations:

- `GENERATION_MODEL`, `GENERATION_API_BASE_URL`, `GENERATION_API_KEY`
- `VERIFICATION_MODEL`, `VERIFICATION_API_BASE_URL`, `VERIFICATION_API_KEY`
- `WEB_GENERATION_MODEL` optionally chooses a model name for web answers.

Verification settings fall back to generation settings. Model/provider changes
are read at backend startup, so restart the backend to apply them. Native
provider APIs that do not implement OpenAI Chat Completions need a compatible
endpoint or a dedicated adapter.

The graph is compiled once per API process and invoked per question. It does
not use a LangGraph checkpointer; conversation history remains managed by the
existing application store. Add a durable checkpointer only if graph execution
needs to pause, resume, or preserve intermediate state across requests.
