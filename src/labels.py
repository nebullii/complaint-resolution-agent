"""Label set and the map from CFPB product names (old and new) to our labels."""

LABELS = [
    "bank_account",
    "credit_card",
    "credit_reporting",
    "debt_collection",
    "money_transfer",
    "mortgage",
    "student_loan",
]

# Misses on these classes carry regulatory risk, so we track their recall separately.
HIGH_RISK = ["debt_collection", "mortgage"]

# The CFPB renamed its product categories in 2017 and again in 2023.
# Every name that has appeared in the public database maps to one label.
# Products outside our 7 classes (payday loans, vehicle loans, ...) map to None.
PRODUCT_MAP = {
    # bank_account
    "Bank account or service": "bank_account",
    "Checking or savings account": "bank_account",
    # credit_card
    "Credit card": "credit_card",
    "Prepaid card": "credit_card",
    "Credit card or prepaid card": "credit_card",
    # credit_reporting
    "Credit reporting": "credit_reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "credit_reporting",
    "Credit reporting or other personal consumer reports": "credit_reporting",
    # debt_collection
    "Debt collection": "debt_collection",
    # money_transfer
    "Money transfers": "money_transfer",
    "Virtual currency": "money_transfer",
    "Money transfer, virtual currency, or money service": "money_transfer",
    # mortgage
    "Mortgage": "mortgage",
    # student_loan
    "Student loan": "student_loan",
    # out of scope
    "Consumer Loan": None,
    "Vehicle loan or lease": None,
    "Payday loan": None,
    "Payday loan, title loan, or personal loan": None,
    "Payday loan, title loan, personal loan, or advance loan": None,
    "Other financial service": None,
    "Debt or credit management": None,
}


def normalize_product(product):
    """Map a raw CFPB product name to one of LABELS, or None if out of scope.

    Raises KeyError for a name we have never seen, so a new CFPB rename
    fails loudly instead of being silently dropped.
    """
    return PRODUCT_MAP[product.strip()]
