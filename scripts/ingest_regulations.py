"""Download the regulations from eCFR and (re)build the Chroma store.

    python -m scripts.ingest_regulations
"""

from collections import Counter

from src.retrieval.ecfr import load_chunks, to_dicts
from src.retrieval.regulations import REGULATIONS
from src.retrieval.store import RegulationStore


def main():
    chunks = to_dicts(load_chunks(list(REGULATIONS)))
    ids = [c["id"] for c in chunks]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        raise SystemExit(f"duplicate chunk ids: {duplicates[:5]}")

    for part, n in sorted(Counter(c["part"] for c in chunks).items()):
        print(f"{part} {REGULATIONS[part]}: {n} chunks")
    count = RegulationStore().build(chunks)
    print(f"stored {count} chunks, as of {chunks[0]['as_of']}")


if __name__ == "__main__":
    main()
