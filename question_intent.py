"""Question interpretation for LEAF literature retrieval.

The seven LEAFv2 literature categories are the PRIMARY classification. Nielsen
et al.'s educational question types and Cambridge wh-question guidance provide
SECONDARY linguistic cues, not validated labels for this research setting.

Search tokens stay a subset of the researcher's literal question keywords. Never
invent a platform, clinical condition, age range, or population as an anchor.
"""
import json
import re

LEAF_CATEGORIES = (
    "Content Analysis", "Detection/Prediction", "Sentiment/Language Analysis",
    "Stigma/Misinformation", "Behavior/Risk", "Social Support",
    "Other/Uncategorized",
)
QUESTION_FORMS = ("Description", "Method", "Explanation", "Comparison", "Preference")
EVIDENCE_GOALS = (
    "describe_content", "predict_outcomes", "language_patterns", "content_accuracy",
    "associations", "causal_effects", "social_support", "other",
)

PLANNING_RULES = """You interpret research questions BEFORE a literature search.
Use the seven LEAFv2 categories as the PRIMARY research-question taxonomy:
Content Analysis (describing themes/representations in platform content),
Detection/Prediction (ML classification/prediction),
Sentiment/Language Analysis (linguistic patterns, sentiment/topic modeling),
Stigma/Misinformation (content reliability, stigma, misinformation),
Behavior/Risk (social-media behaviors, correlations, temporal trends, or possible effects),
Social Support (peer support, help seeking, community responses),
Other/Uncategorized (not represented clearly by the above).
The historical category counts are corpus frequencies, NOT prediction probabilities.
Independently classify the linguistic question form using the Nielsen et al.
question families: Description, Method, Explanation, Comparison, Preference.
Wh-words (what/who/which/when/where/why/how) are grammatical hints, not
sufficient to decide scientific intent. "How is X discussed" describes content;
"how does exposure change X" asks about effects. Distinguish WHAT posts say,
WHAT people experience, and whether exposure CAUSES outcomes.
Choose an evidence goal from: describe_content, predict_outcomes,
language_patterns, content_accuracy, associations, causal_effects,
social_support, other. If multiple meanings materially change the evidence,
mark ambiguous and explain the interpretation chosen; never silently conflate them.
Select priority_keywords ONLY from the supplied allowed literal question keywords.
Do not invent synonyms, diagnoses, populations, platforms, or age ranges.
Keep both "mental" and "health" when the question says "mental health".
Do not treat adult women as adolescent girls or platform posts as causal studies.
Return ONE JSON object with exactly these fields:
{"category":"Content Analysis|Detection/Prediction|Sentiment/Language Analysis|Stigma/Misinformation|Behavior/Risk|Social Support|Other/Uncategorized",
"secondary_categories":[],
"question_form":"Description|Method|Explanation|Comparison|Preference",
"evidence_goal":"describe_content|predict_outcomes|language_patterns|content_accuracy|associations|causal_effects|social_support|other",
"interpretation":"short, specific paraphrase of what is being asked",
"priority_keywords":["literal", "keywords"],
"ambiguous":false,
"clarification_note":"if ambiguous, say what other interpretation is possible; otherwise empty"}
No outside web search and no claims about retrieved papers.
"""


def heuristic_intent(question: str, terms: list[str], facets: dict) -> dict:
    """Clearly labeled fallback, not a validated classifier."""
    q = question.lower()
    if re.search(r"\b(predict|detect|classify|identify.*risk|machine learning|prediction)\b", q):
        category, goal = "Detection/Prediction", "predict_outcomes"
    elif re.search(r"\b(misinformation|misleading|stigma|stigmatiz|accuracy|accurate|reliable)\w*\b", q):
        category, goal = "Stigma/Misinformation", "content_accuracy"
    elif re.search(r"\b(sentiment|language patterns|linguistic|word choice|words used)\b", q):
        category, goal = "Sentiment/Language Analysis", "language_patterns"
    elif re.search(r"\b(peer support|help.seek|support communit|crisis support|responses? to pleas)\w*\b", q):
        category, goal = "Social Support", "social_support"
    elif re.search(r"\b(caus(e|es|ed|ing)|impact|effect|affect|influence|risk factor|correlat|associat|relationship|link between)\w*\b", q):
        category = "Behavior/Risk"
        goal = "causal_effects" if re.search(r"\b(caus(e|es|ed|ing)|impact|effect|affect|influence)\w*\b", q) else "associations"
    else:
        category, goal = "Content Analysis", "describe_content"
    if re.search(r"\b(compare|comparison|versus| vs\.? |difference|differ)\w*\b", q):
        form = "Comparison"
    elif re.search(r"\b(why|cause[sd]?|leads? to|what explains)\b", q):
        form = "Explanation"
    elif re.search(r"\b(how (do|to) (i|we|you)\b|steps to|procedure for)\b", q):
        form = "Method"
    else:
        form = "Description"
    ambiguity = bool(re.search(r"\b(what does|what do) (instagram|reddit|tiktok|facebook|social media) say\b", q))
    priority = list(dict.fromkeys(facets["platform"] + facets["topic"] + facets["population"] + facets["age"]))
    priority = [x for x in priority if x in terms]
    return {
        "category":category, "secondary_categories":[], "question_form":form,
        "evidence_goal":goal,
        "interpretation":question.strip()[:240], "priority_keywords":priority[:8],
        "ambiguous":ambiguity,
        "clarification_note":"This may refer to what users post rather than what the platform itself asserts or what exposure causes." if ambiguity else "",
        "method":"heuristic_fallback",
    }


def interpret_question(question: str, terms: list[str], facets: dict, model_call, log) -> dict:
    """Obtain an AI interpretation or disclose a bounded heuristic fallback."""
    fallback = heuristic_intent(question, terms, facets)
    try:
        response, _ = model_call(instructions=PLANNING_RULES,
            input_text=json.dumps({"question":question, "allowed_keywords":terms,
                                   "literal_facets":facets}, ensure_ascii=False), max_tokens=750)
        raw = response.strip()
        fence = chr(96) * 3
        if raw.startswith(fence):
            raw = re.sub(r"^" + re.escape(fence) + r"(?:json)?\s*|\s*" + re.escape(fence) + r"$", "", raw).strip()
        plan = json.loads(raw)
        if not isinstance(plan, dict): raise ValueError("not an object")
        if plan.get("category") not in LEAF_CATEGORIES: raise ValueError("unknown LEAF category")
        if plan.get("question_form") not in QUESTION_FORMS: raise ValueError("unknown question form")
        if plan.get("evidence_goal") not in EVIDENCE_GOALS: raise ValueError("unknown evidence goal")
        extras = plan.get("secondary_categories", [])
        selected = plan.get("priority_keywords", [])
        if not isinstance(extras,list) or len(extras)>3 or any(x not in LEAF_CATEGORIES for x in extras):
            raise ValueError("invalid secondary categories")
        if not isinstance(selected,list) or len(selected)>9 or any(not isinstance(x,str) or x not in terms for x in selected):
            raise ValueError("nonliteral keyword detected")
        if not isinstance(plan.get("ambiguous"),bool): raise ValueError("invalid ambiguity flag")
        if any(not isinstance(plan.get(k),str) or len(plan[k])>350 for k in ("interpretation","clarification_note")):
            raise ValueError("invalid interpretation text")
        # Retain anchors even if the planner omitted them. A selected search term
        # can never authorize dropping a named platform, population, or health topic.
        anchors = list(dict.fromkeys(facets["platform"] + facets["topic"] + facets["population"]))
        chosen = list(dict.fromkeys(anchors + selected))
        plan["priority_keywords"] = [x for x in chosen if x in terms][:9]
        plan["secondary_categories"] = list(dict.fromkeys(x for x in extras if x != plan["category"]))
        plan["method"] = "perplexity_question_interpretation"
        log("QUESTION_TYPE", f"{plan['category']} | {plan['question_form']} | goal={plan['evidence_goal']}")
        log("LOOKUP_WORDS", ", ".join(plan["priority_keywords"]))
        return plan
    except (ValueError,TypeError,KeyError,RuntimeError,AttributeError) as error:
        log("QUESTION_TYPE", f"AI plan unavailable or invalid ({type(error).__name__}); using labeled heuristic fallback.")
        return fallback


def prioritize_queries(queries: list[str], plan: dict, facets: dict) -> list[str]:
    """Reorder existing anchored queries, never insert invented search concepts."""
    keywords = plan.get("priority_keywords") or []
    def score(query: str) -> tuple[int,int]:
        q=query.lower()
        score_count = sum(1 for term in keywords if f"{term.lower()}[" in q)
        # For descriptive content searches, platform + topic should precede a
        # looser platform + population result. Causal questions retain topic too.
        topic_hits = sum(1 for term in facets["topic"] if f"{term.lower()}[" in q)
        return (topic_hits,score_count)
    # Python sort is stable: ties retain the original strategy ordering.
    return sorted(queries, key=score, reverse=True)
