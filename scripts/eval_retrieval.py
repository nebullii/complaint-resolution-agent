"""Retrieval eval: does the governing section come back in the top k?

Each case is a short complaint-style query, its product and the section that
governs it. A hit is any chunk from that section, either the regulation text
or its official interpretation in Supplement I. Extend CASES as you review
real complaints; 15 cases is a smoke test, not a benchmark.

    python -m scripts.eval_retrieval
"""

import json

from src import config
from src.retrieval.regulations import PRODUCT_PARTS
from src.retrieval.store import RegulationStore

CASES = [
    ("debt_collection", "The collector never sent me any notice explaining the debt or how to dispute it", "1006.34"),
    ("debt_collection", "They call me seven or eight times a day about this debt and will not stop", "1006.14"),
    ("debt_collection", "The debt collector keeps calling me at work even though I told them my employer prohibits it", "1006.6"),
    ("debt_collection", "The collector threatened to have me arrested if I do not pay", "1006.18"),
    ("debt_collection", "They are suing me over a debt that is too old to be collected in court", "1006.26"),
    ("bank_account", "Someone made unauthorized debit card withdrawals from my checking account and the bank says I am liable", "1005.6"),
    ("bank_account", "I reported an error on my account statement and the bank never investigated it", "1005.11"),
    ("money_transfer", "I sent an international remittance and the recipient never received the money", "1005.33"),
    ("credit_card", "I disputed a billing error on my credit card statement and the issuer ignored it", "1026.13"),
    ("credit_card", "My credit card was stolen and I am being held responsible for the fraudulent charges", "1026.12"),
    ("credit_reporting", "I disputed inaccurate information directly with the company that reported it and nothing happened", "1022.43"),
    ("mortgage", "I sent my mortgage servicer a notice of error and they never responded", "1024.35"),
    ("mortgage", "My servicer moved ahead with foreclosure while my loan modification application was pending", "1024.41"),
    ("mortgage", "My escrow account shortage made my monthly payment jump without explanation", "1024.17"),
    ("bank_account", "The bank charged overdraft fees that were never disclosed when I opened the savings account", "1030.4"),
]


def section_of(passage):
    """'12 CFR 1006.34(a)' -> '1006.34'; 'Part 1006, Supp. I, 34(c)' -> '1006.34'."""
    if passage.source == "interpretation":
        code = passage.citation.split("Supp. I, ", 1)[-1].split("(")[0]
        return f"{passage.part}.{code}" if code.isdigit() else passage.part
    return passage.citation.split()[-1].split("(")[0]


def main(k=config.RETRIEVAL_TOP_K):
    store = RegulationStore()
    hits, rows = 0, []
    for label, query, expected in CASES:
        passages = store.search(query, PRODUCT_PARTS[label], k=k)
        found = [section_of(p) for p in passages]
        hit = expected in found
        hits += hit
        rows.append({"expected": expected, "hit": hit, "retrieved": [p.citation for p in passages]})
        print(f"{'HIT ' if hit else 'MISS'} {expected:9} <- {', '.join(found)}")

    recall = hits / len(CASES)
    print(f"\nrecall@{k}: {recall:.2f} ({hits}/{len(CASES)})")
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "retrieval_eval.json").write_text(
        json.dumps({"k": k, "recall": recall, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
