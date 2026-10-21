"""LLM access with a hard spend cap and optional LangSmith tracing.

Every call goes through `LLM.complete_json`, which:
  1. refuses to run once the spend ledger has reached LLM_SPEND_CAP_USD,
  2. asks for JSON that matches a schema (structured output),
  3. records the cost of the call in the ledger.
"""

import json
import os
import threading
from dataclasses import dataclass

from src import config


class SpendCapExceeded(RuntimeError):
    pass


class SpendLedger:
    """Running total of LLM spend, kept in a small JSON file so it survives restarts."""

    _lock = threading.Lock()

    def __init__(self, path=config.SPEND_LEDGER_PATH, cap_usd=config.LLM_SPEND_CAP_USD):
        self.path = path
        self.cap_usd = cap_usd

    def total(self):
        if not self.path.exists():
            return 0.0
        return json.loads(self.path.read_text())["total_usd"]

    def check(self):
        if self.total() >= self.cap_usd:
            raise SpendCapExceeded(f"LLM spend cap of ${self.cap_usd:.2f} reached")

    def add(self, usd):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"total_usd": self.total() + usd}))


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self):
        return (self.input_tokens * config.LLM_INPUT_PRICE_PER_M
                + self.output_tokens * config.LLM_OUTPUT_PRICE_PER_M) / 1e6


class LLM:
    def __init__(self, model=config.LLM_MODEL, ledger=None):
        self.model = model
        self.ledger = ledger or SpendLedger()
        self._client = None

    @property
    def client(self):
        # Created on first use, so the API can serve /classify without an OpenAI key.
        if self._client is None:
            from openai import OpenAI

            client = OpenAI()
            if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
                from langsmith.wrappers import wrap_openai
                client = wrap_openai(client)
            self._client = client
        return self._client

    def complete_json(self, system, user, schema_name, schema):
        """Return (parsed_dict, Usage). Raises SpendCapExceeded before calling if over cap."""
        self.ledger.check()
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
            text={"format": {"type": "json_schema", "name": schema_name,
                             "schema": schema, "strict": True}},
        )
        usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
        self.ledger.add(usage.cost_usd)
        return json.loads(response.output_text), usage
