# Decision record — 2026-10-19

**Issue observed:** A question about how depression was discussed on Reddit became the PubMed query `depression[Title/Abstract] AND talked[Title/Abstract]`, because the application removed the final keyword (`reddit`) when results were scarce. The first eight papers were mostly unrelated clinical studies and none passed Gate 2.

**Search correction:** Extract substantive words from the question and ignore discourse wording (`talked`, `discussed`). Keep any question-stated platform and primary health topic in every relaxed search. For the observed question, both `reddit` and `depression` are required. Search all-fields only after a narrower Title/Abstract search, without silently dropping either anchor. Do not inject additional medical conditions, platforms, or generic synonyms. Retain candidate identifiers and search strings for inspection.

**Pipeline correction:** Adapt the original `Leaf_Gates(3).ipynb` Gate 1 and Gate 2 criteria (v4A-2026-10-07). Each gate is independently evaluated three times; majority vote determines the stage label. Gate 1 evaluates substantive mental/behavioral-health relevance and does **not** apply a social-media constraint. Gate 2 evaluates genuine analyzed platform data rather than contextual platform mentions or survey/recruitment alone. Human labels cannot change AI gate decisions. The original notebook evaluated each paper separately; this web prototype batches papers per pass to keep latency and cost manageable.

**Evidence correction:** Port the notebook's PubMed PMID-to-PMC linkage, article-title matching, preference for sectioned full text, and abstract fallback. Only screening-passed sources feed the final synthesis. Source citations must match actual included PMIDs. No general Perplexity web-search tools.

**Boundary:** The current prototype does not implement the original extraction and RQ taxonomy stages or Xiaoqian's federated social-media knowledge graph. It is not a new validation of the existing 30-paper study. A live rerun with your own API key and a human quality check is necessary before reporting effectiveness to Jina.

**Tests:** Offline tests cover the failing question, anchored query relaxations, source exclusion, PMC preference/fallback, separate gates, citation checks, and graph access.

## October 19: Terminal visibility and citation recovery

- **Problem observed:** The model sometimes produced a two-paragraph answer with a citation in only the first paragraph. The strict client-side scientific validation rejected the whole answer, and the web page showed a citation error without useful terminal progress.
- **Change:** Added timestamps, request IDs, step-specific progress/error logs, and screening vote counts to stdout. Upstream errors show the host and status but not the API token or model input. Enabled optional `LEAF_DEBUG=1` traceback output.
- **Change:** Linked inline `[n]` citations to the matching PubMed source. A malformed or uncited summary triggers one constrained retry using only the same eligible evidence, followed by a clearly labeled, brief extractive fallback if repair fails.
- **Rationale:** Make slow, multi-call pipeline behavior inspectable; preserve source traceability rather than treating citation formatting as an unrecoverable failure. No change to PubMed keyword generation, Gate 1, Gate 2, majority voting, or eligibility criteria.
- **Verification:** Offline deterministic tests, including uncited answer repair, repair failure, and keyword/LEAF screening regression tests. Live API behavior remains to be checked with a local key.

## October 19, 2026 — Broader research question interpretation and retrieval

**Observed failure:** For a question concerning Reddit, women, young girls and mental health, the prior fallback `reddit AND women` retrieved ten papers weighted heavily toward narrow groups (pregnancy, abortion, sexual dysfunction, COVID-19) and lost essential mental-health and age scope. Eligibility screening alone could not distinguish topical research papers from papers that addressed the researcher's actual question. Findings from individual communities appeared broader than the underlying evidence.

**Change:** Introduced explicit question facets (platform, topic, population, age) formed only from question words; multiple complementary queries instead of an all-word prefix fallback; deduplicated pooled retrieval and balanced sampling of named populations (default max 18 papers); a question-relevance assessment after the original Gate 1/Gate 2 screening; and explicit search-coverage warnings. Direct and Partial evidence may support the answer, but Background papers cannot. Screening rules and majority votes are unchanged. The interactive view and terminal expose the separate discovery/relevance decisions.

**Why:** Avoid selecting a narrow, potentially biased slice of available papers, preserve age/population distinctions, and make the uncertainty and limitations visible rather than representing a selected sample as a full review.

**Limitations/next validation:** A broader PubMed keyword search remains non-exhaustive; no independent human assessment of relevance-grade accuracy has been completed; PMC access is partial; no systematic-review methodology or federated query is implemented. Test with diverse human-written questions and manually review both missed papers and misleadingly included papers. Live Perplexity/API behavior requires testing with a private key. These changes are to the **question-answer prototype**, not proof of improvement in the original 30-paper LEAF human-validation kappa metrics.
