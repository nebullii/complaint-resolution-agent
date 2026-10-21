"""LangGraph workflow: classify -> (human triage) -> retrieve -> policy agent ->
drafting agent -> human approval -> finalize.

The graph receives text that is already PII-masked (see src/service.py), so raw
personal data never enters graph state, checkpoints, traces or the audit log.
Two nodes pause for a person with `interrupt`:
  - triage:    the classifier was not confident enough to pick the product alone
  - approval:  every draft is approved, edited or rejected by a reviewer
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents import prompts
from src.labels import LABELS
from src.retrieval.regulations import PRODUCT_PARTS

CITATION = re.compile(r"\[([^\[\]]+)\]")


class State(TypedDict, total=False):
    complaint_id: str
    text: str
    label: str
    confidence: float
    model_version: str
    needs_triage: bool
    passages: list[dict]
    obligations: list[dict]
    policy_reasoning: str
    draft: str | None
    citations: list[str]
    grounded: bool
    decision: str
    reviewer: str
    final_reply: str | None
    status: str


@dataclass
class Deps:
    classifier: Any  # .predict(text) -> Prediction
    store: Any       # .search(query, parts) -> list[Passage]
    llm: Any         # .complete_json(system, user, schema_name, schema) -> (dict, Usage)
    audit: Any       # .log(complaint_id, step, payload, actor=..., model_version=...)


def check_grounding(reply, citations, allowed):
    """A draft is grounded when every citation it uses, in the text or the list,
    is one of the obligations it was given, and it cites at least one."""
    used = set(citations) | set(CITATION.findall(reply))
    unknown = sorted(used - set(allowed))
    return bool(used) and not unknown, unknown


def build_graph(deps, checkpointer):
    def log(state, step, payload, **kw):
        deps.audit.log(state["complaint_id"], step, payload, **kw)

    def classify(state):
        p = deps.classifier.predict(state["text"])
        log(state, "classify",
            {"text": state["text"], "label": p.label, "confidence": round(p.confidence, 4),
             "needs_human": p.needs_human},
            model_version=p.model_version)
        return {"label": p.label, "confidence": p.confidence,
                "model_version": p.model_version, "needs_triage": p.needs_human}

    def triage(state):
        answer = interrupt({"type": "triage", "predicted": state["label"],
                            "confidence": state["confidence"], "labels": LABELS})
        label = answer["label"]
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        log(state, "triage", {"predicted": state["label"], "chosen": label},
            actor=answer["reviewer"])
        return {"label": label, "reviewer": answer["reviewer"]}

    def retrieve(state):
        passages = deps.store.search(state["text"], PRODUCT_PARTS[state["label"]])
        passages = [asdict(p) for p in passages]
        log(state, "retrieve", {"parts": PRODUCT_PARTS[state["label"]],
                                "hits": [{"citation": p["citation"], "score": round(p["score"], 3)}
                                         for p in passages]})
        return {"passages": passages}

    def policy_agent(state):
        by_id = {p["id"]: p for p in state["passages"]}
        listing = "\n\n".join(f"[{p['id']}] {p['citation']} ({p['heading']})\n{p['text']}"
                              for p in state["passages"])
        result, usage = deps.llm.complete_json(
            prompts.POLICY_SYSTEM,
            prompts.POLICY_USER.format(label=state["label"], complaint=state["text"], passages=listing),
            "policy", prompts.POLICY_SCHEMA)

        # Keep only obligations whose citation belongs to a retrieved passage the agent chose.
        chosen = [by_id[i] for i in result["applicable_ids"] if i in by_id]
        allowed = {p["citation"] for p in chosen}
        obligations = [o for o in result["obligations"] if o["citation"] in allowed]
        log(state, "policy_agent",
            {"applicable": sorted(allowed), "obligations": obligations,
             "dropped": len(result["obligations"]) - len(obligations),
             "cost_usd": round(usage.cost_usd, 5)})
        return {"obligations": obligations, "policy_reasoning": result["reasoning"]}

    def draft_agent(state):
        allowed = [o["citation"] for o in state["obligations"]]
        listing = "\n".join(f"[{o['citation']}] {o['requirement']}" for o in state["obligations"])
        feedback, cost = "", 0.0
        for attempt in range(2):
            result, usage = deps.llm.complete_json(
                prompts.DRAFT_SYSTEM,
                prompts.DRAFT_USER.format(label=state["label"], complaint=state["text"],
                                          obligations=listing, feedback=feedback),
                "draft", prompts.DRAFT_SCHEMA)
            cost += usage.cost_usd
            grounded, unknown = check_grounding(result["reply"], result["citations"], allowed)
            if grounded:
                break
            feedback = (f"\nYour previous draft cited rules you were not given: {unknown}. "
                        "Cite only the obligations listed above, and cite at least one.")
        log(state, "draft_agent", {"citations": result["citations"], "grounded": grounded,
                                   "attempts": attempt + 1, "cost_usd": round(cost, 5)})
        return {"draft": result["reply"], "citations": result["citations"], "grounded": grounded}

    def approval(state):
        answer = interrupt({"type": "approval", "label": state["label"],
                            "draft": state.get("draft"), "citations": state.get("citations", []),
                            "grounded": state.get("grounded", False),
                            "obligations": state.get("obligations", [])})
        decision = answer["decision"]
        if decision not in ("approve", "edit", "reject"):
            raise ValueError(f"unknown decision {decision!r}")
        if decision == "edit" and not answer.get("edited_reply"):
            raise ValueError("an edit needs edited_reply")
        final = {"approve": state.get("draft"), "edit": answer.get("edited_reply"),
                 "reject": None}[decision]
        log(state, "approval", {"decision": decision, "final_reply": final},
            actor=answer["reviewer"])
        return {"decision": decision, "reviewer": answer["reviewer"], "final_reply": final,
                "status": "rejected" if decision == "reject" else "approved"}

    def after_classify(state):
        return "triage" if state["needs_triage"] else "retrieve"

    def after_policy(state):
        # No applicable rule: skip drafting, a reviewer writes the reply from scratch.
        return "draft_agent" if state["obligations"] else "approval"

    g = StateGraph(State)
    for name, fn in [("classify", classify), ("triage", triage), ("retrieve", retrieve),
                     ("policy_agent", policy_agent), ("draft_agent", draft_agent),
                     ("approval", approval)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", after_classify, ["triage", "retrieve"])
    g.add_edge("triage", "retrieve")
    g.add_edge("retrieve", "policy_agent")
    g.add_conditional_edges("policy_agent", after_policy, ["draft_agent", "approval"])
    g.add_edge("draft_agent", "approval")
    g.add_edge("approval", END)
    return g.compile(checkpointer=checkpointer)
