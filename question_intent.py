"""Question interpretation for LEAF literature retrieval.

The seven LEAFv2 literature categories are the PRIMARY classification. Nielsen
et al.'s educational question types and Cambridge wh-question guidance provide
SECONDARY linguistic cues, not validated labels for this research setting.

Search tokens stay a subset of the researcher's literal question keywords. Never
invent a platform, clinical condition, age range, or population as an anchor.
"""
import json
import re

QUESTION_CATEGORIES = {
    "Explicative": ("What makes P be P?", "Definitions, concepts, identifying attributes"),
    "Descriptive": ("What are the characteristics?", "Description, prevalence, patterns, characteristics"),
    "Exploratory": ("How do factors relate?", "Associations, relationships, discovery of connections/hypotheses"),
    "Explanatory": ("Why does P cause Q?", "Causes, mechanisms, causal chains"),
    "Theory Research": ("What rules apply to all P?", "General theories, propositions, predictive/generalizable models"),
    "Translational": ("How can knowledge be applied?", "Practical methods, tools, interventions, techniques"),
    "Evaluative": ("Does the solution work?", "Effectiveness, implementation, consequences, program/intervention evaluation"),
}
LEAF_CATEGORIES = tuple(QUESTION_CATEGORIES)
LEAFV2_METHOD_TAGS = (
    "Content Analysis", "Detection/Prediction", "Sentiment/Language Analysis",
    "Stigma/Misinformation", "Behavior/Risk", "Social Support", "Other/Uncategorized",
)
QUESTION_FORMS = ("Description", "Method", "Explanation", "Comparison", "Preference")
EVIDENCE_GOALS = (
    "describe_content", "predict_outcomes", "language_patterns", "content_accuracy",
    "associations", "causal_effects", "social_support", "other",
)

PLANNING_RULES = """Interpret a research question BEFORE searching for papers.
Use exactly these seven PRIMARY research-PURPOSE categories from the supplied table:
Explicative: What makes P be P? Definitions, concepts, identifying attributes.
Descriptive: What are the characteristics? Description, prevalence, patterns.
Exploratory: How do factors relate? Associations, relationships, emerging hypotheses.
Explanatory: Why does P cause Q? Causes, mechanisms, causal chains.
Theory Research: What rules apply to all P? General theories, propositions, predictive/generalizable models.
Translational: How can knowledge be applied? Practical methods, tools, interventions, techniques.
Evaluative: Does the solution work? Effectiveness, implementation, consequences, program evaluation.
Explicative means defining something; Explanatory means explaining causality.
Exploratory association evidence is NOT enough for an Explanatory causal answer.
Proposed practical applications are NOT proof an intervention works.
The older LEAFv2 Content Analysis, Detection/Prediction, Sentiment/Language Analysis,
Stigma/Misinformation, Behavior/Risk, Social Support, Other/Uncategorized are
OPTIONAL SECONDARY paper method/content tags, not primary question categories.
The old 256-paper counts are NOT probabilities for the new purpose classes.
Wh-words are grammatical cues only; classify the actual requested evidence.
Also mark a secondary linguistic question_form (Description, Method, Explanation,
Comparison, Preference) and evidence_goal (describe_content, predict_outcomes,
language_patterns, content_accuracy, associations, causal_effects, social_support, other).
Return 0-3 older LEAFv2 tags in secondary_categories if useful.
Select priority_keywords ONLY from allowed literal question keywords. Do not
invent synonyms, illnesses, age ranges, platforms or populations. Preserve user
named platforms, mental-health topics and populations in search.
If ambiguous, explain the chosen focus and competing meaning.
Return ONE JSON object:
{"category":"Explicative|Descriptive|Exploratory|Explanatory|Theory Research|Translational|Evaluative",
"secondary_categories":["optional old LEAFv2 labels"],
"question_form":"Description|Method|Explanation|Comparison|Preference",
"evidence_goal":"describe_content|predict_outcomes|language_patterns|content_accuracy|associations|causal_effects|social_support|other",
"interpretation":"short specific paraphrase",
"priority_keywords":["only terms from allowed_keywords"],
"ambiguous":false,
"clarification_note":"alternate interpretation if ambiguous; otherwise empty"}
No external web search and no claims about unsupplied papers.
"""


def heuristic_intent(question: str, terms: list[str], facets: dict) -> dict:
    """Transparent, unvalidated fallback classifier for research PURPOSE."""
    q=question.lower()
    def match(pattern):
        return re.search(pattern,q) is not None
    if match(r"\b(does|did|has|have)\b.{0,90}\b(work|effective|improve|reduce|help|successful)\b|\b(evaluat\w*|effectiveness|efficacy|intervention outcome|how well)\b"):
        category="Evaluative"
    elif match(r"\b(how can|how should|how to|ways to|apply|implement|build|design|develop)\b.{0,120}\b(tool|program|intervention|practice|moderation|methods?|research|knowledge|finding|support|mental|social|sources?)\b"):
        category="Translational"
    elif match(r"\b(what (defines|counts as|makes|distinguishes|constitutes|is meant by)|definition of|criteria for|how (is|are) .{0,45} defined)\b"):
        category="Explicative"
    elif match(r"\b(why|what causes|what explains|underlying mechanism|causal (pathway|chain)|how does .{0,55} (cause|lead to|produce)|does .{0,50} make .{0,50} feel)\b"):
        category="Explanatory"
    elif match(r"\b(generaliz\w*|universal|theor\w*|predictive model|general model|predict across|model that predicts|rules apply|across all populations)\b|\bcan .{0,60} (predict|forecast)\b"):
        category="Theory Research"
    elif match(r"\b(associated|association|correlat\w*|relationship|relat\w* to|linked|links? between|connections? between|factors? relate|risk factor|predictors of)\b"):
        category="Exploratory"
    else:
        category="Descriptive"
    if match(r"\b(compare|comparison|versus|difference|differ\w*)\b"):form="Comparison"
    elif match(r"\b(why|causes?|leads? to|what explains)\b"):form="Explanation"
    elif match(r"\b(how (do|to) (i|we|you)\b|steps to|procedure for|apply|implement|design)\b"):form="Method"
    else:form="Description"
    if category=="Theory Research":goal="predict_outcomes"
    elif category=="Explanatory":goal="causal_effects"
    elif category=="Exploratory":goal="associations"
    elif category=="Evaluative":goal="causal_effects" if match(r"\b(effect|effective|impact|improv\w*|reduc\w*)\b") else "other"
    elif match(r"\b(misinformation|misleading|stigma|stigmatiz\w*|accuracy|accurate|reliable)\b"):goal="content_accuracy"
    elif match(r"\b(sentiment|language patterns|linguistic|word choice)\b"):goal="language_patterns"
    elif match(r"\b(peer support|help.seek\w*|support communit\w*)\b"):goal="social_support"
    else:goal="describe_content" if category=="Descriptive" else "other"
    tags=[]
    for regex,label in [
        (r"misinformation|misleading|stigma|accuracy|accurate","Stigma/Misinformation"),
        (r"predict|detect|classif|model","Detection/Prediction"),
        (r"sentiment|language|linguistic","Sentiment/Language Analysis"),
        (r"peer support|help.seek|support communit","Social Support"),
        (r"effect|affect|risk|correlat|association|behavior","Behavior/Risk")
    ]:
        if match(regex):tags.append(label)
    if not tags and category=="Descriptive":tags.append("Content Analysis")
    ambiguity=match(r"\b(what does|what do) (instagram|reddit|tiktok|facebook|social media) say\b")
    priority=list(dict.fromkeys(facets["platform"]+facets["topic"]+facets["population"]+facets["age"]))
    return {"category":category,"secondary_categories":tags[:3],"question_form":form,
        "evidence_goal":goal,"interpretation":question.strip()[:240],
        "priority_keywords":[x for x in priority if x in terms][:9],
        "ambiguous":ambiguity,"clarification_note":"May refer to what people post, not what the platform claims or exposure causes." if ambiguity else "",
        "method":"heuristic_fallback"}


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
        if not isinstance(extras,list) or len(extras)>3 or any(x not in LEAFV2_METHOD_TAGS for x in extras):
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
        plan["secondary_categories"] = list(dict.fromkeys(extras))
        plan["method"] = "perplexity_question_interpretation"
        log("QUESTION_TYPE", f"{plan['category']} (research purpose) | older LEAFv2 tags={plan['secondary_categories']} | goal={plan['evidence_goal']}")
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
