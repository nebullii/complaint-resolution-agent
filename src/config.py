"""Settings read from environment variables (and .env). One place for every knob."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# Classifier
CLASSIFIER_PATH = Path(os.getenv("CLASSIFIER_PATH", MODELS_DIR / "tfidf_lr.joblib"))
# Share of complaints we are willing to send to a human for triage.
# The confidence threshold is derived from this on train data, never on test.
TARGET_ESCALATION_RATE = float(os.getenv("TARGET_ESCALATION_RATE", "0.2"))

# LLM
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.6-luna")
LLM_INPUT_PRICE_PER_M = float(os.getenv("LLM_INPUT_PRICE_PER_M", "2.0"))
LLM_OUTPUT_PRICE_PER_M = float(os.getenv("LLM_OUTPUT_PRICE_PER_M", "12.0"))
# Hard stop: no LLM call is made once the ledger reaches this amount.
LLM_SPEND_CAP_USD = float(os.getenv("LLM_SPEND_CAP_USD", "5.0"))
SPEND_LEDGER_PATH = Path(os.getenv("SPEND_LEDGER_PATH", DATA_DIR / "llm_spend.json"))

# Retrieval
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

# Audit log. Postgres in docker-compose; SQLite fallback for local runs and tests.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'audit.db'}")
