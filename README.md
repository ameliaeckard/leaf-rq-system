# LEAF research query prototype

Plain local interface: researcher question → keywords present in the question → **topic + platform anchored PubMed search** → PMC full text (if available) / PubMed abstract → **LEAF v4A Gate 1, then Gate 2** → two paragraphs maximum with citations → optional evidence graph.

The code adapts the method from `Leaf_Gates(3).ipynb`: the Gate 1 substantive mental/behavioral-health criterion, the strict Gate 2 platform-data criterion, 3 independent votes per gate and no human-label interference. Screening is batched for responsiveness. The original notebook's downstream extraction and research-question taxonomy classification are not run for each new live query.

## Run

Use Python 3.10+ (standard library only). Set your **Perplexity** key in your terminal, not in a public source file. For example, in **PowerShell**:

```powershell
$env:PERPLEXITY_API_KEY="YOUR_PRIVATE_API_KEY"
python app.py
```

Open <http://127.0.0.1:8765>. The script uses the Perplexity Agent API and no general web-search tools. Optionally specify `PERPLEXITY_MODEL`; by default it uses the same general reasoning model as the earlier minimal prototype. Avoid models configured to perform their own web searches because LEAF sources must come from PubMed/PMC.

Run offline tests (no API key or network needed):

```powershell
python app.py --test
```

## Example

Question: `How is depression talked about on reddit?`

Extracted keywords: `depression`, `reddit` (not `talked`). PubMed queries **never drop Reddit or depression**. If the abstract query is too restrictive, both terms are searched in wider PubMed fields. The app will not use an unrelated clinical paper to answer this question.

## Research limits

- PubMed/PMC is the retrieval source; full text is used only if accessible in PMC and the title matches. Otherwise the abstract is used and labeled as such. This does **not** provide publisher-paywalled full text.
- The prototype caps the number of screened papers; it does not provide a comprehensive systematic review. Negative results do not prove absence of literature.
- Each paper receives three independent **Gate 1** and three independent **Gate 2** votes. Calls are batched across papers instead of one API request per paper. Vote reasons and the search log are inspectable. The human validation labels remain separate.
- The result is a short synthesis of papers that pass both gates. The final summary cannot introduce outside web sources; every paragraph requires citations to included PMIDs.
- The evidence graph visualizes words/papers/mentioned platforms; mentions do not establish verified dataset use or HiTOP classification. Xiaoqian's original graph is not included. For an authorized aggregate-only graph export, set `LEAF_GRAPH_PATH` to a local JSON containing `nodes` and `links`.
- The 30-paper validation metrics do not automatically validate this new question-specific retrieval and synthesis workflow. Researcher verification is still needed.
- Live runs require internet access, an API key and may incur **9 model calls** (question interpretation, 3 per gate, question relevance, and synthesis) per question, plus PubMed/PMC calls. Exact charges depend on usage.

**Safety:** Never commit `.env`, API keys or private participant-level data to the public repository. The local server binds to `127.0.0.1`, not the public network.

## Watching a question in the terminal

When you run `python app.py`, the terminal prints timestamped operational messages with a request ID, for example:

```text
12:00:01 [Q0001] START        New research question received.
12:00:01 [Q0001] KEYWORDS     Extracted: depression, reddit
12:00:01 [Q0001] SEARCH       PubMed query: depression[Title/Abstract] AND reddit[Title/Abstract]
12:00:02 [Q0001] PUBMED       Found 24 record IDs (up to 36).
12:00:03 [Q0001] EVIDENCE     PMID 12345678: matched PMC full text
12:00:04 [Q0001] SCREEN       GATE1 pass 1/3 started
12:00:13 [Q0001] SCREEN       GATE1 pass 1/3 finished: Yes=8, No=2, Uncertain=0
...
12:01:22 [Q0001] SUMMARY      Synthesizing evidence from 4 screened-in papers.
12:01:35 [Q0001] FINISH       Returned answer mode=model, 4 cited sources.
12:01:35 [Q0001] DONE         Request duration: 94.0s
```

Times and counts above are **illustrative**, not actual results. If an upstream service fails, its hostname and HTTP status are printed without exposing your key. You can enable a complete Python traceback for unexpected errors with `$env:LEAF_DEBUG="1"` before starting. Logs are written to the terminal only, not saved to disk by the app; the search terms will be visible to anyone looking at the terminal.

The answer's numbered references such as `[1]` link directly to the matching paper in the Sources section. Every synthesized paragraph requires at least one citation. If the first Perplexity answer omits citations or references unknown sources, one repair call uses the **same screened-in papers**; if that also fails, LEAF returns clearly labeled short extractive excerpts with citations rather than silently accepting an unsupported answer. When no papers pass both gates, it cannot provide a supported answer and instead exposes the screening log.

A normal question costs 1 question interpretation call + 3 Gate 1 calls + 3 Gate 2 calls + 1 question relevance call + 1 summary call (9 model calls); citation repair may require a tenth. Full-text requests and PubMed queries also take time. This is still a research prototype: neither keyword retrieval nor machine screening alone establishes comprehensive coverage or methodological validation.

## Broad questions and representative evidence (October 19 revision)

The initial keyword search worked poorly for questions such as “How does Reddit social media affect the mental health of women and young girls?” It required every question word, then could fall back to `reddit AND women`, losing the health topic and age distinction. The first ten results then dominated the answer, even when they concerned narrow pregnancy, abortion, or sexual-health communities. Gate 1 and Gate 2 are **eligibility** checks, not checks that a study answers the user's question.

The app now groups **only user-stated terms** into platform, health topic, population, and age facets. It executes and records complementary PubMed queries for (1) platform + health topic + population, (2) platform + population, (3) platform + health topic, and, where relevant, (4) platform + health topic + age. The broader passes are explicitly screening candidates, not automatic evidence for the answer. It retrieves up to 45 PubMed records per pass, deduplicates them, and balances distinct requested populations in its capped sample (default 18, configurable via `LEAF_MAX_PAPERS`, max 24).

After the existing three-vote Gate 1/Gate 2 decisions, **one separate Perplexity question-relevance assessment** grades eligible papers Direct, Partial, or Background. Only Direct and Partial papers are allowed into the summary. This additional grade **does not modify** Gate 1, Gate 2, or the LEAF research corpus. A coverage note warns when a requested group is absent and explicitly distinguishes adult women from girls. Research-specific subgroup evidence must never be presented as representative of an entire platform. The Methods section now shows extracted facets, search attempts, pool size, capped sample, vote audit, relevance grades and limitations.

**Research limitation:** This improves discovery breadth but is still **PubMed-only, keyword-dependent, and capped**, not a systematic or representative synthesis. True “social media affects mental health” questions may require longitudinal, comparison-group or causal evidence, which descriptive content analyses do not establish. Other bibliographic databases, controlled-vocabulary expansion, human assessment of retrieval relevance, and the federated graph remain future work. Each question normally makes nine model calls (one interpretation, six screening, one relevance, one synthesis), possibly ten for citation repair. You can lower costs using `LEAF_MAX_PAPERS=10` but results will be narrower. The model's relevance grades have not been independently validated.

## Interpreting a question before searching

Before PubMed retrieval, Perplexity now classifies a researcher's question using
**the seven LEAFv2 research-question categories**: Content Analysis,
Detection/Prediction, Sentiment/Language Analysis, Stigma/Misinformation,
Behavior/Risk, Social Support, and Other/Uncategorized. It separately identifies
a linguistic question form (Description, Method, Explanation, Comparison, or
Preference) and an evidence goal, such as `describe_content` versus
`causal_effects`. The linguistic question forms are an *adaptation* of Nielsen et
al.'s taxonomy of tutoring questions, and wh-word signals are informed by
Cambridge Grammar. These two sources do **not** validate the LEAF classifier.

The model chooses **priority search words from the question itself** and an
interpretation of its meaning. The app retains named platforms, health topics,
and populations if the model overlooks them. PubMed queries are reordered to
favor the prioritized, anchored concepts; no new diagnoses, demographic groups,
or platform names may be silently added. The screening gates remain separate.
Both the chosen LEAF question category and the exact search words are visible
under **Question interpretation** and in the screening audit. If the AI planner
returns invalid JSON, an explicitly labeled heuristic fallback is used.

Example distinctions:

| User question | Research question type | Evidence goal |
| --- | --- | --- |
| What are people posting about depression on Reddit? | Content Analysis | Describe content |
| Can Reddit posts predict depression? | Detection/Prediction | Predict outcomes |
| How accurate are mental-health claims on TikTok? | Stigma/Misinformation | Content accuracy |
| Does Instagram cause anxiety in teenage girls? | Behavior/Risk | Causal effects (requires stronger study designs) |

This adds **one model call per question**: normally nine total (one question
interpretation, three per screening gate, one question relevance assessment, and
one synthesis), potentially ten with citation repair. No real API run has been
performed without a user's private key. Use `python app.py --test` for local tests.

**References for the interpretation design (not model training):**

- Nielsen, R. D., Buckingham, J., Knoll, G., Marsh, B., & Palen, L. *A Taxonomy of Questions for Question Generation.*
  https://www.cs.memphis.edu/~vrus/questiongeneration/15-NielsenEtAl-QG08.pdf
- Cambridge Dictionary, *Questions: wh-questions.*
  https://dictionary.cambridge.org/us/grammar/british-grammar/questions-wh-questions
