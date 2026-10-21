"""Fakes for the classifier, vector store and LLM, so the workflow is tested
without model files, network access or API spend."""

import pytest

from src.agents.graph import Deps
from src.audit import AuditLog
from src.classifier import Prediction
from src.llm import Usage
from src.retrieval.store import Passage
from src.service import ComplaintService


class FakeClassifier:
    def __init__(self, label="debt_collection", confidence=0.9, threshold=0.5):
        self.label, self.confidence, self.threshold = label, confidence, threshold

    def predict(self, text):
        return Prediction(self.label, self.confidence, "fake-v1", self.confidence < self.threshold)


class FakeStore:
    def __init__(self):
        self.queries = []

    def search(self, query, parts, k=5):
        self.queries.append((query, parts))
        return [
            Passage(id="1006.34-0", citation="12 CFR 1006.34(a)", heading="Validation notice",
                    text="A debt collector must provide validation information...", part="1006",
                    source="regulation", as_of="2026-10-19", score=0.8),
            Passage(id="1006.14-0", citation="12 CFR 1006.14(b)", heading="Harassment",
                    text="A debt collector must not call repeatedly...", part="1006",
                    source="regulation", as_of="2026-10-19", score=0.7),
        ]


class FakeLLM:
    """Returns scripted responses in order and records every prompt it receives."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system, user, schema_name, schema):
        self.calls.append((schema_name, user))
        return self.responses.pop(0), Usage(100, 20)


POLICY_OK = {
    "applicable_ids": ["1006.34-0"],
    "obligations": [{"citation": "12 CFR 1006.34(a)", "requirement": "Send validation information."}],
    "reasoning": "Consumer disputes the debt.",
}
DRAFT_OK = {"reply": "We will send validation information [12 CFR 1006.34(a)].",
            "citations": ["12 CFR 1006.34(a)"]}


@pytest.fixture
def audit(tmp_path):
    return AuditLog(f"sqlite:///{tmp_path / 'audit.db'}")


@pytest.fixture
def make_service(audit):
    def _make(classifier=None, llm_responses=(POLICY_OK, DRAFT_OK)):
        llm = FakeLLM(llm_responses)
        deps = Deps(classifier=classifier or FakeClassifier(), store=FakeStore(), llm=llm, audit=audit)
        return ComplaintService(deps), llm
    return _make
