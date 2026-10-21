import pytest

from src.agents.graph import check_grounding
from tests.conftest import DRAFT_OK, POLICY_OK, FakeClassifier

COMPLAINT = "A collector keeps calling me about a debt I do not owe. My email is jo@example.com."


def test_confident_complaint_goes_straight_to_approval(make_service):
    service, llm = make_service()
    result = service.submit(COMPLAINT)

    assert result["state"]["status"] == "awaiting_approval"
    assert result["pending"]["draft"] == DRAFT_OK["reply"]
    assert result["pending"]["grounded"] is True
    assert [name for name, _ in llm.calls] == ["policy", "draft"]


def test_pii_never_reaches_the_llm_or_audit_log(make_service, audit):
    service, llm = make_service()
    result = service.submit(COMPLAINT)

    assert all("jo@example.com" not in prompt for _, prompt in llm.calls)
    assert "jo@example.com" not in str(audit.events(result["complaint_id"]))
    assert "[EMAIL]" in result["state"]["text"]


def test_low_confidence_pauses_for_triage_then_uses_reviewer_label(make_service):
    service, _ = make_service(classifier=FakeClassifier(label="credit_reporting", confidence=0.3))
    result = service.submit(COMPLAINT)
    assert result["pending"]["type"] == "triage"

    result = service.review(result["complaint_id"], {"label": "debt_collection", "reviewer": "neha"})
    assert result["state"]["label"] == "debt_collection"
    assert result["pending"]["type"] == "approval"
    # Retrieval searched Regulation F, the reviewer's product, not Regulation V.
    assert service.deps.store.queries[-1][1] == ["1006"]


def test_approve_and_edit_and_reject(make_service, audit):
    for decision, extra, status, final in [
        ("approve", {}, "approved", DRAFT_OK["reply"]),
        ("edit", {"edited_reply": "Edited [12 CFR 1006.34(a)]."}, "approved", "Edited [12 CFR 1006.34(a)]."),
        ("reject", {}, "rejected", None),
    ]:
        service, _ = make_service()
        cid = service.submit(COMPLAINT)["complaint_id"]
        result = service.review(cid, {"decision": decision, "reviewer": "neha", **extra})
        assert result["state"]["status"] == status
        assert result["state"]["final_reply"] == final
        assert result["pending"] is None
        steps = [e["step"] for e in audit.events(cid)]
        assert steps == ["received", "classify", "retrieve", "policy_agent", "draft_agent", "approval"]


def test_policy_agent_cannot_cite_passages_it_was_not_given(make_service):
    invented = {**POLICY_OK, "obligations": POLICY_OK["obligations"]
                + [{"citation": "12 CFR 9999.1", "requirement": "Made up."}]}
    service, _ = make_service(llm_responses=(invented, DRAFT_OK))
    result = service.submit(COMPLAINT)
    citations = [o["citation"] for o in result["state"]["obligations"]]
    assert citations == ["12 CFR 1006.34(a)"]


def test_ungrounded_draft_is_retried_once_then_flagged(make_service):
    bad = {"reply": "Under 12 CFR 1026.13 [12 CFR 1026.13] we will refund you.",
           "citations": ["12 CFR 1026.13"]}
    service, llm = make_service(llm_responses=(POLICY_OK, bad, bad))
    result = service.submit(COMPLAINT)
    assert [name for name, _ in llm.calls] == ["policy", "draft", "draft"]
    assert "were not given" in llm.calls[-1][1]
    assert result["pending"]["grounded"] is False


def test_no_applicable_rule_skips_drafting(make_service):
    none = {"applicable_ids": [], "obligations": [], "reasoning": "Nothing applies."}
    service, llm = make_service(llm_responses=(none,))
    result = service.submit(COMPLAINT)
    assert [name for name, _ in llm.calls] == ["policy"]
    assert result["pending"]["type"] == "approval"
    assert result["pending"]["draft"] is None


def test_review_without_pending_question_is_refused(make_service):
    service, _ = make_service()
    cid = service.submit(COMPLAINT)["complaint_id"]
    service.review(cid, {"decision": "approve", "reviewer": "neha"})
    with pytest.raises(LookupError):
        service.review(cid, {"decision": "approve", "reviewer": "neha"})


@pytest.mark.parametrize("reply,citations,ok", [
    ("Text [12 CFR 1006.34(a)].", ["12 CFR 1006.34(a)"], True),
    ("Text [12 CFR 1006.34(a)] and [12 CFR 1026.13].", ["12 CFR 1006.34(a)"], False),
    ("No citations at all.", [], False),
])
def test_check_grounding(reply, citations, ok):
    assert check_grounding(reply, citations, ["12 CFR 1006.34(a)"])[0] is ok
