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
- Live runs require internet access, an API key and may incur **7 model calls** (3 per gate plus one synthesis) per question, plus PubMed/PMC calls. Exact charges depend on usage.

**Safety:** Never commit `.env`, API keys or private participant-level data to the public repository. The local server binds to `127.0.0.1`, not the public network.