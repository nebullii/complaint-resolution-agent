"""HTTP API.

    uvicorn src.api.main:app --reload

POST /classify                      product + confidence only (fast path, no LLM)
POST /complaints                    full workflow; pauses for triage or approval
GET  /complaints/{id}               current state and any pending question
POST /complaints/{id}/review        answer the pending question
GET  /complaints/{id}/audit         every recorded step
"""

import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import pii
from src.llm import SpendCapExceeded


class ComplaintIn(BaseModel):
    text: str = Field(min_length=20, max_length=20000)


class ClassifyOut(BaseModel):
    label: str
    confidence: float
    needs_human: bool
    model_version: str
    latency_ms: float


class TriageIn(BaseModel):
    reviewer: str
    label: str


class ApprovalIn(BaseModel):
    reviewer: str
    decision: Literal["approve", "edit", "reject"]
    edited_reply: str | None = None


def default_service():
    from src.agents.graph import Deps
    from src.audit import AuditLog
    from src.classifier import TfidfClassifier
    from src.llm import LLM
    from src.retrieval.store import RegulationStore
    from src.service import ComplaintService

    deps = Deps(classifier=TfidfClassifier(), store=RegulationStore(), llm=LLM(), audit=AuditLog())
    return ComplaintService(deps)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        svc()  # load the classifier and embedding model once, at startup
        yield

    app = FastAPI(title="Complaint Resolution Assistant", lifespan=lifespan)
    app.state.service = service

    def svc():
        if app.state.service is None:
            app.state.service = default_service()
        return app.state.service

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/classify", response_model=ClassifyOut)
    def classify(body: ComplaintIn):
        service = svc()  # load models before the timer starts
        start = time.perf_counter()
        masked, counts = pii.mask(body.text)
        p = service.deps.classifier.predict(masked)
        latency = (time.perf_counter() - start) * 1000
        service.deps.audit.log("classify-only", "classify",
                               {"text": masked, "pii_masked": counts, "label": p.label,
                                "confidence": round(p.confidence, 4), "needs_human": p.needs_human},
                               model_version=p.model_version)
        return ClassifyOut(label=p.label, confidence=p.confidence, needs_human=p.needs_human,
                           model_version=p.model_version, latency_ms=round(latency, 2))

    @app.post("/complaints")
    def submit(body: ComplaintIn):
        try:
            return svc().submit(body.text)
        except SpendCapExceeded as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/complaints/{complaint_id}")
    def get(complaint_id: str):
        try:
            return svc().get(complaint_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/complaints/{complaint_id}/review")
    def review(complaint_id: str, body: TriageIn | ApprovalIn):
        try:
            pending = svc().get(complaint_id)["pending"]
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))
        if pending is None:
            raise HTTPException(status_code=409, detail="nothing is waiting for review")
        expected = TriageIn if pending["type"] == "triage" else ApprovalIn
        if not isinstance(body, expected):
            raise HTTPException(status_code=422, detail=f"expected a {pending['type']} answer")
        try:
            return svc().review(complaint_id, body.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except SpendCapExceeded as e:
            raise HTTPException(status_code=503, detail=str(e))

    @app.get("/complaints/{complaint_id}/audit")
    def audit(complaint_id: str):
        return svc().deps.audit.events(complaint_id)

    return app


app = create_app()
