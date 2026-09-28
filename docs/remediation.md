# Verification failure remediation

The question endpoint classifies an unverified answer using the strongest
retrieval score:

- Scores below `RETRIEVAL_SCORE_THRESHOLD` (default `0.5`) are
  `retrieval_gap` failures. The system searches SerpAPI, requires at least one
  usable result, and labels a verified answer as `not found in your documents`.
- Healthy scores are `bad_generation` failures. The system regenerates against
  the same chunks with stricter quotation and grounding instructions, then
  reruns Lynx.
- If the selected remediation fails verification or an external service fails,
  the API returns `I could not find reliable information to answer this.`

Configure `SERPAPI_API_KEY` (or the existing `WEB_SEARCH_API_KEY`) in the
backend `.env` file. Adjust the retrieval threshold only after calibrating it
against representative documents.

Web generation uses `GROQ_WEB_MODEL` when configured. If that model is
unavailable or attempts an unsupported tool call, the Groq adapter retries
configured fallbacks and then discovers compatible models from Groq's
`/models` endpoint.
