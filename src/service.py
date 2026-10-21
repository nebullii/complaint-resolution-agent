"""Entry point used by the API: masks PII, runs the graph, resumes it after human review."""

import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src import pii
from src.agents.graph import build_graph


class ComplaintService:
    def __init__(self, deps, checkpointer=None):
        self.deps = deps
        # In-memory checkpoints are lost on restart. For production, swap in
        # langgraph-checkpoint-postgres against the same database as the audit log.
        self.graph = build_graph(deps, checkpointer or InMemorySaver())

    def _config(self, complaint_id):
        return {"configurable": {"thread_id": complaint_id}}

    def submit(self, text):
        complaint_id = uuid.uuid4().hex
        masked, counts = pii.mask(text)
        self.deps.audit.log(complaint_id, "received", {"pii_masked": counts, "chars": len(text)})
        self.graph.invoke({"complaint_id": complaint_id, "text": masked, "status": "in_progress"},
                          self._config(complaint_id))
        return self.get(complaint_id)

    def review(self, complaint_id, answer):
        """Resume a paused complaint with the reviewer's answer to the pending question."""
        if self.pending(complaint_id) is None:
            raise LookupError(f"complaint {complaint_id} is not waiting for review")
        self.graph.invoke(Command(resume=answer), self._config(complaint_id))
        return self.get(complaint_id)

    def pending(self, complaint_id):
        snapshot = self.graph.get_state(self._config(complaint_id))
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        return interrupts[0].value if interrupts else None

    def get(self, complaint_id):
        snapshot = self.graph.get_state(self._config(complaint_id))
        if not snapshot.values:
            raise LookupError(f"unknown complaint {complaint_id}")
        values = dict(snapshot.values)
        pending = self.pending(complaint_id)
        if pending:
            values["status"] = f"awaiting_{pending['type']}"
        return {"complaint_id": complaint_id, "state": values, "pending": pending}
