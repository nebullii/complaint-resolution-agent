import pytest

from src import pii
from src.labels import LABELS, PRODUCT_MAP, normalize_product
from src.llm import SpendCapExceeded, SpendLedger
from src.retrieval.ecfr import parse_part
from src.retrieval.regulations import PRODUCT_PARTS, REGULATIONS


@pytest.mark.parametrize("text,kind", [
    ("write to me at jo.doe+cfpb@mail.example.com please", "EMAIL"),
    ("my SSN is 123-45-6789", "SSN"),
    ("card 4111 1111 1111 1111 was charged", "CARD"),
    ("call (555) 123-4567 or 555.123.4567", "PHONE"),
    ("account number 000123456789", "ACCOUNT"),
])
def test_pii_masked(text, kind):
    masked, counts = pii.mask(text)
    assert f"[{kind}]" in masked and counts[kind] >= 1


def test_pii_leaves_ordinary_text_and_cfpb_masks_alone():
    text = "On XX/XX/2023 I paid {$200.00} to XXXX for loan 12 of 2023."
    assert pii.mask(text) == (text, {})


def test_number_failing_luhn_is_not_called_a_card():
    masked, counts = pii.mask("reference 1234567812345678")
    assert "CARD" not in counts and counts == {"ACCOUNT": 1}


def test_every_label_has_regulations_and_every_mapped_product_is_known():
    assert set(PRODUCT_PARTS) == set(LABELS)
    assert all(p in REGULATIONS for parts in PRODUCT_PARTS.values() for p in parts)
    assert set(v for v in PRODUCT_MAP.values() if v) == set(LABELS)


def test_mini_dataset_product_names_map_to_their_label():
    assert normalize_product("Prepaid card") == "credit_card"
    assert normalize_product("Credit reporting or other personal consumer reports") == "credit_reporting"
    assert normalize_product("Payday loan") is None
    with pytest.raises(KeyError):
        normalize_product("Brand new CFPB product")


def test_spend_cap_blocks_calls(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=0.01)
    ledger.check()
    ledger.add(0.006)
    ledger.add(0.006)
    with pytest.raises(SpendCapExceeded):
        ledger.check()


ECFR_XML = b"""<DIV5 N="1006" TYPE="PART">
<DIV8 N="1006.34" TYPE="SECTION"><HEAD>\xc2\xa7 1006.34 Notice for validation of debts.</HEAD>
<P>(a) <I>Validation information required.</I> A debt collector must provide it.</P>
<P>(1) In general.</P>
<P>(b) <I>Definitions.</I> For purposes of this section:</P>
</DIV8>
<DIV9 N="Supplement I to Part 1006" TYPE="APPENDIX">
<HD3>34(c) Validation information</HD3><P>1. Itemization date.</P>
</DIV9></DIV5>"""


def test_ecfr_parse_keeps_citations():
    chunks = parse_part(ECFR_XML, "1006", "2026-10-19")
    assert [c.citation for c in chunks] == ["12 CFR 1006.34(a)", "12 CFR Part 1006, Supp. I, 34(c)"]
    assert "(b) Definitions" in chunks[0].text


def test_ecfr_paragraph_i_after_h_is_a_letter_not_roman():
    paras = "".join(f"<P>({c}) Para {c}. {'x' * 1400}</P>" for c in "abcdefghi")
    xml = f'<DIV5><DIV8 N="1005.6" TYPE="SECTION"><HEAD>S</HEAD>{paras}</DIV8></DIV5>'.encode()
    chunks = parse_part(xml, "1005", "2026-10-19", interpretations=False)
    assert chunks[-1].citation == "12 CFR 1005.6(i)"
