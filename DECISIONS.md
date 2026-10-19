# Decision record — 2026-10-19

**Issue observed:** A question about how depression was discussed on Reddit became the PubMed query `depression[Title/Abstract] AND talked[Title/Abstract]`, because the application removed the final keyword (`reddit`) when results were scarce. The first eight papers were mostly unrelated clinical studies and none passed Gate 2.

**Search correction:** Extract substantive words from the question and ignore discourse wording (`talked`, `discussed`). Keep any question-stated platform and primary health topic in every relaxed search. For the observed question, both `reddit` and `depression` are required. Search all-fields only after a narrower Title/Abstract search, without silently dropping either anchor. Do not inject additional medical conditions, platforms, or generic synonyms. Retain candidate identifiers and search strings for inspection.

**Pipeline correction:** Adapt the original `Leaf_Gates(3).ipynb` Gate 1 and Gate 2 criteria (v4A-2026-10-07). Each gate is independently evaluated three times; majority vote determines the stage label. Gate 1 evaluates substantive mental/behavioral-health relevance and does **not** apply a social-media constraint. Gate 2 evaluates genuine analyzed platform data rather than contextual platform mentions or survey/recruitment alone. Human labels cannot change AI gate decisions. The original notebook evaluated each paper separately; this web prototype batches papers per pass to keep latency and cost manageable.

**Evidence correction:** Port the notebook's PubMed PMID-to-PMC linkage, article-title matching, preference for sectioned full text, and abstract fallback. Only screening-passed sources feed the final synthesis. Source citations must match actual included PMIDs. No general Perplexity web-search tools.

**Boundary:** The current prototype does not implement the original extraction and RQ taxonomy stages or Xiaoqian's federated social-media knowledge graph. It is not a new validation of the existing 30-paper study. A live rerun with your own API key and a human quality check is necessary before reporting effectiveness to Jina.

**Tests:** Offline tests cover the failing question, anchored query relaxations, source exclusion, PMC preference/fallback, separate gates, citation checks, and graph access.