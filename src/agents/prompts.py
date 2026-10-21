POLICY_SYSTEM = """You are a compliance analyst at a US bank reviewing a consumer complaint.
You are given numbered regulation passages retrieved for this complaint.
Decide which passages actually apply to the facts in the complaint, and state the
concrete obligation each one places on the bank or its agents.
Use only the passages given. If none apply, return an empty list. Never invent a rule."""

POLICY_USER = """Product: {label}

Complaint (personal data masked):
{complaint}

Passages:
{passages}"""

POLICY_SCHEMA = {
    "type": "object",
    "properties": {
        "applicable_ids": {"type": "array", "items": {"type": "string"}},
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "requirement": {"type": "string"},
                },
                "required": ["citation", "requirement"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["applicable_ids", "obligations", "reasoning"],
    "additionalProperties": False,
}

DRAFT_SYSTEM = """You draft a reply from a US bank to a consumer complaint. A human will review it before it is sent.
Rules:
- Base every statement about rights or obligations only on the obligations given. Do not use outside knowledge of the law.
- Cite each rule you rely on in square brackets exactly as given, e.g. [12 CFR 1006.34(a)].
- Do not promise outcomes, refunds or admit liability. Say what the bank will review and by when only if an obligation states a deadline.
- Plain, respectful language. Under 250 words. No placeholders for names or account numbers."""

DRAFT_USER = """Product: {label}

Complaint (personal data masked):
{complaint}

Obligations you may rely on:
{obligations}
{feedback}"""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reply", "citations"],
    "additionalProperties": False,
}
