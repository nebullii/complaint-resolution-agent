# Complaint Resolution Assistant

Classifies bank customer complaints (CFPB data), retrieves the applicable regulation, and drafts a cited response for human approval.

```
complaint ─► PII mask ─► classify ──(low confidence)──► human triage
                            │                               │
                            ▼                               ▼
                    retrieve regulation (Chroma, filtered by product)
                            ▼
                    policy agent: which passages apply, what they require
                            ▼            └──(none apply)──┐
                    drafting agent: reply citing only those rules
                            ▼                             ▼
                    human approval: approve / edit / reject
```

Every step is written to an append-only audit log with the model version. Only masked text is stored.

## Results so far

Classification, 7 products, frozen test set of 1,635 complaints (`python -m scripts.evaluate --compare`):

| Approach | Macro F1 | Debt collection recall | Mortgage recall | p95 latency | Cost / 1k | Deterministic |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | **0.831** | **0.80** | 0.90 | **3 ms** | ~$0 | yes |
| bge-small embeddings + LR | 0.825 | 0.80 | 0.92 | 191 ms | ~$0 | yes |
| Zero-shot LLM (gpt-5.6-luna) | 0.794 | 0.47 | 0.91 | 1,936 ms | $1.30* | no |
| TF-IDF, LLM fallback when unsure | 0.851 | 0.75 | 0.94 | | | no |
| Fine-tuned ModernBERT | run `notebooks/05` on Colab | | | | | |

\* at an assumed $2 / $12 per 1M input / output tokens; set the real prices in `.env`.

Findings:
- The zero-shot LLM sent more than half of debt collection complaints to credit reporting.
- The LLM fallback raised macro F1 by 0.02 but lowered debt collection recall, so unsure complaints go to a human instead.
- The main confusion, bank account vs money transfer, comes from consumer-chosen labels (label noise), not the model.
- With the confidence threshold set on train (20% target), 16.7% of test complaints go to a human; the rest are 89.8% accurate.

Retrieval: recall@5 of 0.87 on 15 hand-written cases (`python -m scripts.eval_retrieval`).

## Setup

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY and the real model prices
python -m scripts.train_baseline        # trains, evaluates, logs to MLflow, saves models/tfidf_lr.joblib
python -m scripts.ingest_regulations    # downloads eCFR Title 12 parts, builds data/chroma
python -m pytest
uvicorn src.api.main:app --reload
```

Optional: `docker compose up -d` and set `DATABASE_URL` for Postgres; `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse runs; `LANGSMITH_TRACING=true` to trace LLM calls.

## Layout

| Path | What |
|---|---|
| `notebooks/01-05` | exploration, baseline, zero-shot LLM, cascade, fine-tuning (Colab) |
| `src/labels.py` | the 7 labels and the old-to-new CFPB product name map |
| `src/classifier.py` | serving wrapper: label, confidence, needs_human, model version |
| `src/pii.py` | rule-based masking of emails, SSNs, card, phone and account numbers |
| `src/retrieval/` | eCFR download and parsing into cited chunks, Chroma store |
| `src/agents/graph.py` | LangGraph workflow with two human interrupts |
| `src/llm.py` | OpenAI calls with structured output and a hard spend cap |
| `src/audit.py` | audit log (Postgres or SQLite) |
| `src/api/main.py` | FastAPI endpoints |
| `scripts/` | train, evaluate, compare, retrieval eval, drift check, 100-complaint review |

## API

```bash
curl -X POST localhost:8000/classify -H 'content-type: application/json' -d '{"text": "..."}'
curl -X POST localhost:8000/complaints -H 'content-type: application/json' -d '{"text": "..."}'
curl -X POST localhost:8000/complaints/<id>/review -H 'content-type: application/json' \
     -d '{"reviewer": "neha", "decision": "approve"}'      # or {"reviewer": ..., "label": ...} for triage
curl localhost:8000/complaints/<id>/audit
```

## Known limits

- Regulations: Reg E, F, V, X, Z and DD only. Federal student loans (Title 34) are not covered.
- PII masking is rule-based; names and street addresses are not detected.
- Workflow checkpoints are in memory and lost on restart; use the LangGraph Postgres checkpointer in production.
- Retrieval eval has 15 cases; misses on validation notices (1006.34) and time-barred debt (1006.26).
- CFPB data has no real bank replies, so draft quality has no ground truth yet; it is checked for citation grounding only.
