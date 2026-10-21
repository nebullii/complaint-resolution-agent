"""Real classifier and real Chroma store; only the LLM is scripted.
Skipped until `scripts.train_baseline` and `scripts.ingest_regulations` have run."""

import re

import pytest

from src import config
from src.agents.graph import Deps
from src.llm import Usage
from src.service import ComplaintService

pytestmark = pytest.mark.skipif(
    not (config.CLASSIFIER_PATH.exists() and config.CHROMA_DIR.exists()),
    reason="needs models/tfidf_lr.joblib and data/chroma")

COMPLAINT = ("A debt collector keeps calling me several times a day about a medical debt I already "
             "paid. They never sent me a validation notice. Call me back at 555-201-3344.")


class TopPassageLLM:
    """Policy step picks the first retrieved passage; draft step cites it."""

    def complete_json(self, system, user, schema_name, schema):
        if schema_name == "policy":
            # Passages are listed as "[<id>] <citation> (<heading>)"
            self.id, self.citation = re.search(r"^\[(\S+)\] (.+?) \(", user, re.M).groups()
            return {"applicable_ids": [self.id],
                    "obligations": [{"citation": self.citation, "requirement": "x"}],
                    "reasoning": "top hit"}, Usage(0, 0)
        return {"reply": f"We are reviewing this [{self.citation}].",
                "citations": [self.citation]}, Usage(0, 0)


def test_real_classifier_and_store(audit):
    from src.classifier import TfidfClassifier
    from src.retrieval.store import RegulationStore

    service = ComplaintService(Deps(TfidfClassifier(), RegulationStore(), TopPassageLLM(), audit))
    result = service.submit(COMPLAINT)
    if result["pending"]["type"] == "triage":
        result = service.review(result["complaint_id"], {"label": "debt_collection", "reviewer": "t"})

    state = result["state"]
    assert state["label"] == "debt_collection"
    assert {p["part"] for p in state["passages"]} == {"1006"}
    assert "555-201-3344" not in str(audit.events(result["complaint_id"]))
    assert result["pending"]["type"] == "approval" and result["pending"]["grounded"]
