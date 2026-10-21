"""Download regulation text from the official eCFR API and split it into cited chunks.

Each chunk keeps a citation a reviewer can look up, e.g. "12 CFR 1006.34(c)"
for regulation text or "12 CFR Part 1006, Supp. I, 34(c)" for official
interpretations.
"""

import gzip
import json
import re
import string
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

from src import config

API = "https://www.ecfr.gov/api/versioner/v1"
MAX_CHARS = 1500
LETTERS = string.ascii_lowercase


@dataclass
class Chunk:
    id: str
    part: str
    citation: str
    heading: str
    text: str
    source: str  # "regulation" or "interpretation"
    as_of: str


def _get(url):
    # The eCFR API refuses uncompressed responses; urllib does not decompress, so ask for gzip.
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip",
                                               "User-Agent": "complaint-resolution-agent"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
    return data


def latest_date(title=12):
    titles = json.loads(_get(f"{API}/titles.json"))["titles"]
    return next(t for t in titles if t["number"] == title)["up_to_date_as_of"]


def fetch_part(part, date=None):
    """Return (xml_bytes, date), using a cached copy under data/ecfr/ when present."""
    date = date or latest_date()
    cache = config.DATA_DIR / "ecfr" / f"part-{part}-{date}.xml"
    if cache.exists():
        return cache.read_bytes(), date
    xml = _get(f"{API}/full/{date}/title-12.xml?part={part}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(xml)
    return xml, date


def _text(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _top_marker(paragraph, expected):
    """Return the paragraph letter if it opens the next top-level paragraph, e.g. "(c)".

    "(i)" is ambiguous (letter i or roman one). It counts as a letter only when
    "(h)" came just before it.
    """
    m = re.match(r"\(([a-z])\)", paragraph)
    if m and m.group(1) == expected:
        return m.group(1)
    return None


def _pack(paragraphs, max_chars=MAX_CHARS):
    """Group consecutive (marker, text) paragraphs into chunks under max_chars."""
    groups, current, size = [], [], 0
    for marker, text in paragraphs:
        if current and size + len(text) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append((marker, text))
        size += len(text)
    if current:
        groups.append(current)
    return groups


def _section_chunks(div, part, as_of):
    number = div.get("N")
    heading = _text(div.find("HEAD"))
    paragraphs, marker, expected = [], None, 0
    for p in div.iter("P"):
        text = _text(p)
        if not text:
            continue
        if expected < len(LETTERS) and _top_marker(text, LETTERS[expected]):
            marker = LETTERS[expected]
            expected += 1
        paragraphs.append((marker, text))

    chunks = []
    for i, group in enumerate(_pack(paragraphs)):
        first = group[0][0]
        citation = f"12 CFR {number}" + (f"({first})" if first else "")
        chunks.append(Chunk(
            id=f"{number}-{i}", part=part, citation=citation, heading=heading,
            text="\n".join(t for _, t in group), source="regulation", as_of=as_of))
    return chunks


# Supplement I headings name the provision in several styles:
#   "34(c) Validation information", "Paragraph 14(b)(1)", "Section 1026.13—Billing Error Resolution",
#   "§ 1024.35—Error Resolution Procedures". All reduce to a comment code like "34(c)" or "13".
COMMENT_CODE = re.compile(r"^(?:Section|§|Paragraph)?\s*(?:\d{3,4}\.)?(\d+[a-z]?(?:\([0-9A-Za-z]+\))*)")


def comment_code(heading):
    m = COMMENT_CODE.match(heading)
    return m.group(1) if m else None


def _interpretation_chunks(div, part, as_of):
    """Official interpretations: group paragraphs under each comment heading, e.g. "34(c)"."""
    chunks, label, paragraphs, n = [], None, [], 0

    def flush():
        code = comment_code(label)
        cite = f"12 CFR Part {part}, Supp. I, " + (code or label)
        for i, group in enumerate(_pack([(None, t) for t in paragraphs])):
            chunks.append(Chunk(
                id=f"{part}-supp-{n}-{i}", part=part, citation=cite, heading=label,
                text="\n".join(t for _, t in group), source="interpretation", as_of=as_of))

    for el in div:
        if el.tag in ("HD1", "HD2", "HD3"):
            if label and paragraphs:
                flush()
                n += 1
            label, paragraphs = _text(el), []
        elif el.tag == "P" and label:
            paragraphs.append(_text(el))
    if label and paragraphs:
        flush()
    return chunks


def parse_part(xml, part, as_of, interpretations=True):
    root = ET.fromstring(xml)
    chunks = []
    for div in root.iter("DIV8"):
        if div.get("TYPE") == "SECTION":
            chunks.extend(_section_chunks(div, part, as_of))
    if interpretations:
        for div in root.iter("DIV9"):
            if div.get("N", "").startswith("Supplement I"):
                chunks.extend(_interpretation_chunks(div, part, as_of))
    return chunks


def load_chunks(parts, date=None, interpretations=True):
    chunks = []
    for part in parts:
        xml, as_of = fetch_part(part, date)
        chunks.extend(parse_part(xml, part, as_of, interpretations))
    return chunks


def to_dicts(chunks):
    return [asdict(c) for c in chunks]
